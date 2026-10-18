"""Swap: a track of yours split into its parts, any part swapped for the openly licensed loop nearest your own part or
nearest an artist's version of it, laid where your part plays, and the whole track rendered to download.

Where each step runs (worker/modal_app.py wires them to the swap endpoints):
  read_track(path)    the parts (htdemucs, as every record was split), a steady beat grid and where bars start, each
                      part's level bar by bar, its measures on a listening window (the corpus measured two-minute
                      previews, so the window is measured the same way), the licensed loops and the artists nearest
                      each part, and the window of each part as MP3 for the page to play.
  loop_for(...)       the nearest clean licensed loops (five) for one part and one target (your part, the average of an
                      artist's separated records' same part, or a point between the two), each stretched to your tempo
                      and moved to your key with Rubber Band and cut to whole bars; the first comes with its audio, the
                      others' audio on demand (loop_audio). No audio of yours is involved.
  render(path, plan)  separates again (nothing is kept between calls) and lays the chosen loops by the plan the page
                      played: the same bars, the same gains, the same loop processing, a layer per section where the
                      page chose differently by section, your own part kept under a swap at the share chosen, cuts and
                      rolls at phrase ends, then loudness matched to your track and limited.
The sounds come only from the loop library (Freesound, CC0 and CC BY), never from another record.
"""
import os, io, re, json, math, time, hmac, hashlib, base64, threading, subprocess, tempfile, shutil, urllib.request
import numpy as np

SR = 44100
MAX_S = 480                     # the longest stretch of a track read: eight minutes
PARTS = ("drums", "bass", "other", "vocals")
FAM = {"drums": "drums", "bass": "bass", "other": "melody", "vocals": "vocals"}
SC = ("level", "crest", "dynamic_span", "centroid_hz", "rolloff_hz", "flatness", "onsets_per_s", "share_of_energy")
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- the token the site signs (api/swap.js)
def token_scope(t, key):
    """'<expiry>.<nonce>.<scope>.<signature>', signed by the site with the worker's shared key, valid until its expiry.
    Returns the scope ('t' may split a track; 'e' may choose loops and render the example only), or None."""
    try:
        exp, nonce, scope, sig = str(t or "").split(".")
        if int(exp) < time.time() or not key or scope not in ("t", "e"): return None
        want = hmac.new(key.encode(), f"swap|{exp}|{nonce}|{scope}".encode(), hashlib.sha256).hexdigest()[:32]
        return scope if hmac.compare_digest(want, sig) else None
    except Exception:
        return None


EXAMPLE_URL = "https://www.earlysignal.live/data/swap-example/source.mp3"   # the page's example (worker/swap_example.py)


# ---------------------------------------------------------------- audio in and out
def decode_to_wav(src, dst, max_s=MAX_S):
    """Any file ffmpeg reads, as 44.1 kHz stereo 16-bit WAV, at most max_s seconds."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-t", str(max_s), "-vn", "-ac", "2", "-ar", str(SR),
                    "-c:a", "pcm_s16le", dst], check=True, timeout=300)


def read_wav(path):
    import soundfile as sf
    x, sr = sf.read(path, always_2d=True, dtype="float32")
    if x.shape[1] == 1: x = np.repeat(x, 2, 1)
    return x[:, :2], sr


def decode_bytes(data, sr=SR):
    """Compressed audio bytes (a loop preview) as float32 stereo at sr, through ffmpeg."""
    p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "2", "-ar", str(sr), "pipe:1"],
                       input=data, capture_output=True, check=True, timeout=120)
    return np.frombuffer(p.stdout, np.float32).reshape(-1, 2).copy()


def mp3_bytes(x, sr=SR, kbps=160, meta=None, af=None):
    """float32 stereo -> MP3 bytes (libmp3lame, constant bit rate), through the ffmpeg audio filter af if given (the
    render's limiter). Without a filter the audio is clipped to full scale first, as 16-bit; with one it goes in as
    float, so the filter sees the peaks."""
    if af:
        pcm = np.ascontiguousarray(x, dtype="<f4").tobytes(); fmt = "f32le"
    else:
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes(); fmt = "s16le"
    cmd = ["ffmpeg", "-loglevel", "error", "-f", fmt, "-ar", str(sr), "-ac", "2", "-i", "pipe:0"] + (["-af", af] if af else []) + ["-c:a", "libmp3lame", "-b:a", f"{kbps}k"]
    for k, v in (meta or {}).items(): cmd += ["-metadata", f"{k}={v}"]
    p = subprocess.run(cmd + ["-f", "mp3", "pipe:1"], input=pcm, capture_output=True, check=True, timeout=600)
    return p.stdout


def loudness_lufs(x, sr=SR):
    """Integrated loudness of float32 stereo audio (EBU R128, ffmpeg's ebur128), and its true peak in dBFS."""
    pcm = np.ascontiguousarray(x, dtype="<f4").tobytes()
    p = subprocess.run(["ffmpeg", "-loglevel", "info", "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "pipe:0", "-af", "ebur128=framelog=quiet:peak=true", "-f", "null", "-"],
                       input=pcm, capture_output=True, timeout=600)
    txt = p.stderr.decode("utf-8", "replace")
    # the summary's lines, not the running ones an older ffmpeg prints per frame ("... I: 0.0 LUFS ...")
    mI = re.search(r"Integrated loudness:\s*I:\s*(-?[\d.]+) LUFS", txt); mP = re.search(r"True peak:\s*Peak:\s*(-?[\d.]+) dBFS", txt)
    if not mI:
        allI = re.findall(r"I:\s*(-?[\d.]+) LUFS", txt); allP = re.findall(r"Peak:\s*(-?[\d.]+) dBFS", txt)
        return (float(allI[-1]) if allI else None), (float(allP[-1]) if allP else None)
    return float(mI.group(1)), (float(mP.group(1)) if mP else None)


def wav_bytes(x, sr=SR):
    import soundfile as sf
    b = io.BytesIO(); sf.write(b, np.clip(x, -1, 1), sr, format="WAV", subtype="PCM_16"); return b.getvalue()


def probe_mp3():
    """One second of silence with a single click at exactly half a second, encoded as the parts are: the page decodes it
    to learn how far its MP3 decoder shifts audio (encoder delay), and plays every part that much later or earlier."""
    x = np.zeros((SR, 2), np.float32); x[SR // 2] = 0.9
    return mp3_bytes(x)


# ---------------------------------------------------------------- beats and bars
def _fit_line(tp, a, p, tol=0.08):
    """Kick attack times fitted as a + k*p: each attack indexed to its nearest beat, those within tol of a beat kept,
    a straight line fitted; twice."""
    for _ in range(2):
        k = np.round((tp - a) / p); r = tp - (a + k * p); keep = np.abs(r) < tol * p
        if keep.sum() < 8: break
        A = np.vstack([np.ones(int(keep.sum())), k[keep]]).T
        (a, p), *_ = np.linalg.lstsq(A, tp[keep], rcond=None)
    return float(a), float(p)


def _attack_times(y22, tp, sr=22050, low=None):
    """Each attack found in the spectrogram (whose frames lag a hit by several milliseconds) moved to where the drum
    part's level (its band under low Hz when given, filtered both ways so nothing shifts) first reaches half its peak
    near it."""
    from scipy.signal import butter, sosfiltfilt
    lo = np.abs(sosfiltfilt(butter(4, low, "low", fs=sr, output="sos"), y22)) if low else np.abs(y22)
    out = []
    for t in tp:
        i0, i1 = max(0, int((t - 0.04) * sr)), min(len(lo), int((t + 0.03) * sr))
        seg = lo[i0:i1]
        if len(seg) < 8 or seg.max() <= 0: out.append(t); continue
        j = int(np.argmax(seg >= 0.5 * seg.max()))
        out.append((i0 + j) / sr)
    return np.asarray(out)


def _peaks(env, w=3, avg=12, delta=0.03, wait=8):
    """Local maxima of an onset envelope that stand delta above their neighbourhood's mean, at least wait frames apart."""
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    mx = maximum_filter1d(env, size=2 * w + 1, mode="nearest"); av = uniform_filter1d(env, size=2 * avg + 1, mode="nearest")
    out, last = [], -10 ** 9
    for i in np.where((env >= mx) & (env >= av + delta))[0]:
        if i - last > wait: out.append(int(i)); last = i
    return np.asarray(out, int)


def rough_tempo(drums):
    """The tempo the corpus would give the drum part (features/stems.py, rhythm_of_stem: a learned beat tracker on the
    drums, its grid refitted to the kicks), read on the loudest forty seconds of drums (the middle of a track can be a
    breakdown with none), with that tracker's beats in the track's time (a second opinion on where the beats are);
    (None, []) when it cannot say."""
    import soundfile as sf
    from features import stems as S
    n = len(drums); m = int(40 * SR)
    if n > m:
        e = np.sqrt(np.mean(drums.mean(1)[: n // SR * SR].reshape(-1, SR) ** 2, axis=1))   # level second by second
        c = np.convolve(e, np.ones(40), mode="valid"); i0 = int(np.argmax(c)) * SR
    else:
        i0 = 0
    seg = drums[i0: i0 + m]
    w = tempfile.mkdtemp()
    try:
        fp = os.path.join(w, "loud_drums.wav"); sf.write(fp, seg, SR, subtype="PCM_16")
        r = S.rhythm_of_stem(fp) or {}
        T = float(r.get("beats_per_minute") or 0)
        beats = [i0 / SR + float(b) for b in (r.get("_beats") or [])]
        return (T if 50 <= T <= 250 else None), beats
    except Exception:
        return None, []
    finally:
        shutil.rmtree(w, ignore_errors=True)


def beat_grid(drums, mix, dur, T0=None):
    """A steady beat grid for the whole track, from the drum part (the mix when the drums are near silent): the rough
    tempo as the corpus reads it (T0; librosa's tracker when there is none), folded into 86-186 bpm; the exact period
    from every hit's Fourier component at the eighth-note rate, searched within 4% in steps of 0.01% (hits sit on eighth
    notes in nearly every style, so the tempo holds even where the kick is broken); the beats from where the kicks and
    snares land, which is on the beat in four-on-the-floor, garage, breaks and drum and bass alike, then a straight line
    through those attacks, timed in the waveform. Returns (first beat at or after 0 s, period in seconds, how sure: how
    many times more kick and snare the chosen beats find than a phase chosen at random does; about 10 on a steady four
    on the floor, near 1 where the beats are a guess)."""
    import librosa
    y = drums.mean(1)
    if float(np.sqrt(np.mean(y ** 2))) < 2e-3: y = mix.mean(1)
    y22 = librosa.resample(y.astype(np.float32), orig_sr=SR, target_sr=22050)
    if not T0:
        oenv = librosa.onset.onset_strength(y=y22, sr=22050, hop_length=256)
        tempo, _ = librosa.beat.beat_track(onset_envelope=oenv, sr=22050, hop_length=256, start_bpm=124)
        T0 = float(np.atleast_1d(tempo)[0]) or 124.0
    while T0 < 86: T0 *= 2
    while T0 >= 186: T0 /= 2
    hop = 128; fr = 22050 / hop
    X = np.abs(librosa.stft(y22, n_fft=1024, hop_length=hop)); f = librosa.fft_frequencies(sr=22050, n_fft=1024)
    def rise(v): r = np.maximum(0, np.diff(v, prepend=v[:1])); return r / (r.max() + 1e-12)
    full = rise(np.log1p(X).sum(0)); lowmid = rise(X[(f >= 30) & (f < 500)].sum(0))
    t = np.arange(len(full)) / fr
    periods = 60 / (T0 * (1 + np.linspace(-0.04, 0.04, 801)))
    best, ang, pb = -1.0, 0.0, 60 / T0
    for c in range(0, len(periods), 50):
        P = periods[c:c + 50]
        Z = np.exp(-2j * np.pi * np.outer(2 / P, t)) @ full      # the eighth-note rate
        i = int(np.argmax(np.abs(Z)))
        if abs(Z[i]) > best: best, ang, pb = float(abs(Z[i])), float(np.angle(Z[i])), float(P[i])
    p = pb; a8 = (-ang / (2 * np.pi) * (p / 2)) % (p / 2)
    # the beats: where the kicks and snares land (30-500 Hz), which is on the beat in house, techno, garage, breaks and drum
    # and bass alike. The eighth-note phase of every hit cannot say this: loud hats on the e and the a of each beat pull
    # it a sixteenth off (a tech house record read 80 ms early). Each candidate phase, a millisecond apart, scores the
    # kick and snare rises within 12 ms of its beats
    from scipy.ndimage import maximum_filter1d
    M = maximum_filter1d(lowmid, size=5, mode="nearest")
    cand = np.arange(0, p, 0.001)
    k = np.arange(0, int(dur / p) + 1)
    T = cand[:, None] + k[None, :] * p
    idx = np.clip((T * fr).astype(int), 0, len(M) - 1); valid = (T * fr) < len(M)
    E = (M[idx] * valid).sum(1)
    sure = float(E.max() / (E.mean() + 1e-12))   # how much more the best phase finds on its beats than an average one
    if sure > 1.25:
        a = float(cand[int(np.argmax(E))])
        # then a straight line through the kick and snare attacks at each beat: the strongest rise within a tenth of a
        # beat of it (not the strongest rise anywhere, which on a loud hat would skip the kick after it), timed in the
        # waveform
        w = max(1, int(0.1 * p * fr)); pk, pv = [], []
        for kb in range(0, int((dur - a) / p) + 1):
            c0 = int((a + kb * p) * fr)
            if c0 - w < 1 or c0 + w >= len(lowmid) - 1: continue
            j = c0 - w + int(np.argmax(lowmid[c0 - w:c0 + w + 1])); pk.append(j); pv.append(lowmid[j])
        if len(pk) >= 16:
            pk, pv = np.asarray(pk), np.asarray(pv)
            keep = pv >= 0.25 * np.percentile(pv, 90)
            tp = _attack_times(y22, pk[keep] / fr, low=500)
            kk = np.round((tp - a) / p); near = np.abs(tp - (a + kk * p)) < 0.1 * p
            if near.sum() >= 8: a, p = _fit_line(tp[near], a, p, tol=0.1)
    else:
        # no kick or snare on the beats to go by: the eighth-note phase, and beats where the low band falls more often
        a = a8
        def at_beats(a_):
            kk = np.arange(max(0, int(math.ceil(-a_ / p))), int((dur - a_) / p))
            ii = ((a_ + kk * p) * fr).astype(int); ii = ii[(ii >= 1) & (ii < len(lowmid) - 3)]
            return float(sum(lowmid[i - 1:i + 3].max() for i in ii))
        if at_beats(a + p / 2) > at_beats(a): a = a + p / 2
    while 60 / p < 86: p /= 2
    while 60 / p >= 186: p *= 2
    a = a - math.floor(a / p) * p      # the first beat at or after zero
    return float(a), float(p), sure


def bar_phase(stems, a, p, dur):
    """Which beat of four starts a bar. Three cues, each standardised across the four choices and added:
    the backbeat (a clap or snare's rise in 1.5-8 kHz of the drum part on beats two and four), harmony changing
    (chroma of bass and melody, beat to beat), and the music changing (the mix's energy over the next four beats
    against the last four: drops and breakdowns start bars). Returns (phase, margin)."""
    import librosa
    n = int((dur - a) / p)
    if n < 16: return 0, 0.0
    times = a + np.arange(n) * p
    hop = 256; fr = 22050 / hop
    def mono22(x): return librosa.resample(x.mean(1).astype(np.float32), orig_sr=SR, target_sr=22050)
    d22 = mono22(stems["drums"])
    D = np.abs(librosa.stft(d22, n_fft=1024, hop_length=hop)); f = librosa.fft_frequencies(sr=22050, n_fft=1024)
    hi = D[(f >= 1500) & (f < 8000)].sum(0); hr = np.maximum(0, np.diff(hi, prepend=hi[:1]))
    def at(env, t0, t1):
        i0, i1 = int(t0 * fr), max(int(t0 * fr) + 1, int(t1 * fr))
        return float(env[i0:i1].max()) if i1 <= len(env) and i0 >= 0 else 0.0
    clap = np.array([at(hr, t - 0.03, t + 0.06) for t in times])
    hm = mono22(stems["bass"] + stems["other"])
    C = librosa.feature.chroma_stft(y=hm, sr=22050, hop_length=512)
    cf = 22050 / 512
    ch = np.array([C[:, int(t * cf):max(int(t * cf) + 1, int((t + p) * cf))].mean(1) if int((t + p) * cf) <= C.shape[1] else np.zeros(12) for t in times])
    chn = ch / (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9)
    nov = np.r_[0, 1 - np.sum(chn[1:] * chn[:-1], 1)]
    mx = sum(stems[k] for k in PARTS).mean(1)
    e = np.array([float(np.mean(mx[int(t * SR):int((t + p) * SR)] ** 2)) for t in times])
    le = np.log(np.maximum(e, 1e-3 * (np.median(e) + 1e-12)))   # silence floored 30 dB under the typical beat
    jump = np.zeros(n)
    for k in range(1, n - 1): jump[k] = max(0.0, le[k:k + 4].mean() - le[max(0, k - 4):k].mean())
    # each rise votes once, at its peak, where the four beats before are all old and the four after all new
    peak = np.zeros(n)
    for k in range(1, n - 1):
        if jump[k] >= 0.3 and jump[k] == jump[max(0, k - 2):k + 3].max(): peak[k] = jump[k]
    def std4(v):
        v = np.asarray(v, float); s = v.std()
        return (v - v.mean()) / s if s > 1e-9 else np.zeros(4)
    k = np.arange(n)
    bb = [clap[(k % 4 == (ph + 1) % 4) | (k % 4 == (ph + 3) % 4)].mean() - clap[(k % 4 == ph) | (k % 4 == (ph + 2) % 4)].mean() for ph in range(4)]
    hc = [nov[k % 4 == ph].mean() for ph in range(4)]
    ej = [peak[k % 4 == ph].sum() for ph in range(4)]
    S = std4(bb) + std4(hc) + 1.5 * std4(ej)
    o = np.argsort(-S)
    return int(o[0]), float(S[o[0]] - S[o[1]])


def bar_levels(x, g0, bar, nb):
    """A part's level in each bar: root mean square of its mono sum, in dB (-120 when silent)."""
    m = x.mean(1); out = []
    for b in range(nb):
        s0, s1 = int((g0 + b * bar) * SR), int((g0 + (b + 1) * bar) * SR)
        seg = m[s0:s1]
        r = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
        out.append(round(20 * math.log10(r), 1) if r > 1e-6 else -120.0)
    return out


def beat_levels(x, a, p, nbeats):
    """A part's level on each beat of the grid, in dB (-120 when silent): the page sums four of them into a bar, for
    wherever it puts the bar lines."""
    m = x.mean(1); out = []
    for k in range(nbeats):
        s0, s1 = int((a + k * p) * SR), int((a + (k + 1) * p) * SR)
        seg = m[s0:s1]
        r = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
        out.append(round(20 * math.log10(r), 1) if r > 1e-6 else -120.0)
    return out


def listening_window(levels, nb, bar):
    """About ninety seconds in whole multiples of eight bars, where the most parts play the loudest."""
    W = int(min(nb, max(16, 8 * round(90 / bar / 8)), 64))
    if nb <= W: return 0, nb
    sc = np.zeros(nb)
    for p in PARTS:
        lv = 10 ** (np.asarray(levels[p], float) / 20); ref = np.percentile(lv, 90) or 1.0
        sc += np.minimum(1.0, lv / ref)
    best = max(range(0, nb - W + 1, 4), key=lambda s: (sc[s:s + W].sum(), -s))
    return int(best), W


# ---------------------------------------------------------------- reading a track
def read_track(path, name="track"):
    """Everything the page needs to play the track's parts and swap them; the parts' window audio comes back as MP3."""
    import soundfile as sf
    from features import stems as S
    from worker import scene as SCN
    t_start = time.time(); timing = {}
    w = tempfile.mkdtemp()
    try:
        wav = os.path.join(w, "track.wav"); decode_to_wav(path, wav)
        dur = sf.info(wav).duration
        if dur < 20: return {"error": "the track is shorter than twenty seconds"}
        timing["decode"] = round(time.time() - t_start, 1)
        st = S.separate(wav, os.path.join(w, "sep")); timing["separate"] = round(time.time() - t_start, 1)
        X = {k: read_wav(st[k])[0] for k in PARTS if k in st}
        if len(X) < 4: return {"error": "separation came back without all four parts"}
        n = min(len(v) for v in X.values()); X = {k: v[:n] for k, v in X.items()}; dur = n / SR
        mix = sum(X.values())
        T0, tbeats = rough_tempo(X["drums"]); timing["rough_tempo"] = round(time.time() - t_start, 1)
        a, p, strength = beat_grid(X["drums"], mix, dur, T0); timing["grid"] = round(time.time() - t_start, 1)
        # the tracker's beats against the grid: a second opinion on where the beats fall, in milliseconds (median)
        tr_off = None
        if len(tbeats) >= 8:
            o = (np.asarray(tbeats) - a) % p; o = np.where(o > p / 2, o - p, o); tr_off = round(1000 * float(np.median(o)), 1)
        ph, margin = bar_phase(X, a, p, dur); bar = 4 * p
        g0 = a + ph * p
        if g0 >= bar: g0 -= bar
        nb = int((dur - g0) / bar)
        if nb < 8: return {"error": "too short to swap whole bars"}
        levels = {k: bar_levels(X[k], g0, bar, nb) for k in PARTS}
        nbeats = int((dur - a) / p)
        blevels = {k: beat_levels(X[k], a, p, nbeats) for k in PARTS}
        b0, W = listening_window(levels, nb, bar)
        ws, we = int((g0 + b0 * bar) * SR), min(n, int((g0 + (b0 + W) * bar) * SR))
        timing["bars"] = round(time.time() - t_start, 1)
        # what each part plays, sixteenth by sixteenth: notes for bass, melody and voice; kicks and snares for the drums
        from concurrent.futures import ThreadPoolExecutor
        def read_line(k):
            try:
                if k == "drums":
                    kk, ss = hits_of(X[k], g0, bar, nb); return k, {"kick": kk, "snare": ss}
                nts, on = notes_of(X[k], k, g0, bar, nb); return k, {"notes": nts, "onsets": [int(v) for v in on]}
            except Exception as e:
                return k, {"error": type(e).__name__ + ": " + str(e)[:80]}
        with ThreadPoolExecutor(4) as ex: follow = dict(ex.map(read_line, PARTS))
        timing["lines"] = round(time.time() - t_start, 1)
        # the parts' measures on the window, exactly as a reading measures them
        rec = {}
        for k in PARTS:
            fp = os.path.join(w, f"win_{k}.wav"); sf.write(fp, X[k][ws:we], SR, subtype="PCM_16")
            m = S.measure_stem(fp) or {}
            e = S.analyse_stem(fp)
            if e: m.update(e)
            rec[k] = m
        tot = sum((s or {}).get("level", 0) for s in rec.values()) or 1
        for s in rec.values():
            if s: s["share_of_energy"] = round(s.get("level", 0) / tot, 4)
        vec = {}
        for k in PARTS:
            s_ = rec.get(k) or {}; e_ = s_.get("embedding")
            if isinstance(e_, list) and len(e_) == 45:
                vec[k] = [round(float(x), 5) for x in e_] + [round(float(s_.get(c)), 5) if isinstance(s_.get(c), (int, float)) else 0.0 for c in SC]
        timing["measure"] = round(time.time() - t_start, 1)
        tempo = round(60 / p, 2)
        key = None
        try: key = SCN.key_from_parts(rec)
        except Exception: pass
        like = None
        try:
            like = SCN.part_sounds_like(rec, k=5)
            # only artists Sonic can average: three or more separated records under the name the catalogue writes
            if like and like.get("all"):
                RP = record_parts(); AI = artist_ids(); keep = []
                for nm, sc in like["all"]:
                    hit = AI.get(str(nm).lower())
                    if hit and sum(1 for t in hit[1] if t in RP["at"]) >= 3: keep.append([hit[0], sc])
                like["all"] = keep
        except Exception: pass
        nearest = None
        try: nearest = SCN.licensed_parts(rec, tempo, (key or {}).get("key"))
        except Exception: pass
        audio = {k: mp3_bytes(X[k][ws:we]) for k in PARTS}
        audio["probe"] = probe_mp3()
        timing["total"] = round(time.time() - t_start, 1)
        plain = {k: {c: (rec.get(k) or {}).get(c) for c in ("crest", "centroid_hz", "flatness", "onsets_per_s", "share_of_energy")} for k in PARTS}
        return {"name": str(name)[:120], "duration": round(dur, 2), "tempo": tempo, "key": key,
                "grid": {"first_beat": round(a, 5), "beat": round(p, 6), "bar": round(bar, 6), "g0": round(g0, 5), "nb": nb, "phase": ph, "phase_margin": round(margin, 3),
                         "nbeats": nbeats, "rough_tempo": round(T0, 2) if T0 else None, "strength": round(strength, 4), "tracker_offset_ms": tr_off},
                "beat_levels": blevels,
                "window": {"b0": b0, "bars": W, "start_s": round(ws / SR, 5), "end_s": round(we / SR, 5)},
                "levels": levels, "vectors": vec, "plain": plain, "nearest": nearest, "like": like, "follow": follow,
                "model": S.MODEL, "timing": timing, "audio": audio}
    finally:
        shutil.rmtree(w, ignore_errors=True)


# ---------------------------------------------------------------- the loop library and the targets
_LI = None
_AI = None
_RP = None


def loop_index():
    global _LI
    if _LI is None:
        import pickle
        with open(os.path.join(HERE, "loop_index.pkl"), "rb") as f: _LI = pickle.load(f)
    return _LI


def record_parts(path="/embed/record-parts.npz"):
    """Every separated record's four parts as 53 numbers (tools/build_record_parts.py), indexed by id."""
    global _RP
    if _RP is None:
        d = np.load(path, allow_pickle=False)
        _RP = {"at": {t: i for i, t in enumerate(d["ids"].tolist())}, "V": d["V"]}
    return _RP


def artist_ids():
    """Artist name (any case) -> (name as written, record ids), for artists with three or more records."""
    global _AI
    if _AI is None:
        d = json.load(open(os.path.join(HERE, "artist_ids.json")))["artists"]
        _AI = {k.lower(): (k, v) for k, v in d.items()}
    return _AI


def artists_separated(min_n=3):
    """The artists with at least min_n separated records, for the page's search: [[name, records], ...]."""
    RP = record_parts(); out = []
    for k, (nm, ids) in artist_ids().items():
        c = sum(1 for t in ids if t in RP["at"])
        if c >= min_n: out.append([nm, c])
    out.sort(key=lambda r: (-r[1], r[0].lower()))
    return out


def _space(L, v):
    """A raw part vector in the matcher's robust space (tools/add_robust_measure.py): unit length."""
    keep, mu, sd = L["keep"], np.asarray(L["mu"], float), np.asarray(L["sd"], float)
    z = (np.asarray(v, float)[keep] - mu) / sd; z /= np.linalg.norm(z) + 1e-9
    if "W" in L and "Zp" in L:
        pz = z.astype(np.float32) @ np.asarray(L["W"], np.float32); return pz / (np.linalg.norm(pz) + 1e-9), "Zp"
    return z.astype(np.float32), ("Zs" if "Zs" in L else "Z")


def artist_target(part, name):
    """The average of an artist's separated records' same part, and how close each of those records sits to it in the
    matcher's space (to say how close a loop is, against the artist's own records)."""
    hit = artist_ids().get(str(name or "").strip().lower())
    if not hit: return None, {"error": "Sonic has no separated records by that artist"}
    nm, ids = hit; RP = record_parts(); j = PARTS.index(part)
    rows = [RP["at"][t] for t in ids if t in RP["at"]]
    if len(rows) < 3: return None, {"error": f"Sonic has separated {len(rows)} of {nm}'s records; it needs three"}
    V = RP["V"][rows, j].astype(np.float64)
    c = V.mean(0)
    L = loop_index()[FAM[part]]
    cz, _ = _space(L, c)
    own = np.array([float(_space(L, v)[0] @ cz) for v in V])
    return c, {"artist": nm, "records": len(rows), "own": own, "cz": cz}


# ---------------------------------------------------------------- keys
_PCN = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}
_CAM = {"1A": "G#m", "2A": "D#m", "3A": "A#m", "4A": "Fm", "5A": "Cm", "6A": "Gm", "7A": "Dm", "8A": "Am", "9A": "Em", "10A": "Bm", "11A": "F#m", "12A": "C#m",
        "1B": "B", "2B": "F#", "3B": "C#", "4B": "G#", "5B": "D#", "6B": "A#", "7B": "F", "8B": "C", "9B": "G", "10B": "D", "11B": "A", "12B": "E"}


def parse_key(k):
    """'Am', 'C#', 'D#m', 'Bbm' or a Camelot code '8A' -> (pitch class, minor) or None."""
    k = str(k or "").strip()
    if not k: return None
    k = _CAM.get(k.upper(), k)
    minor = k.endswith("m") and not k.lower().endswith("dim"); nm = (k[:-1] if minor else k).strip().upper()
    nm = nm.replace("MAJ", "").replace("MIN", "").strip()
    pc = _PCN.get(nm)
    return (pc, minor) if pc is not None else None


def key_shift(loop_key, track_key):
    """Semitones that put a loop in the track's key (major and minor meet at their relative), between -5 and +6."""
    a, b = parse_key(loop_key), parse_key(track_key)
    if not a or not b: return 0
    lp, lm = a; tp, tm = b
    if lm != tm: lp = (lp + 9) % 12 if not lm else (lp + 3) % 12
    s = (tp - lp) % 12
    return s - 12 if s > 6 else s


# ---------------------------------------------------------------- one loop, chosen and prepared
def choose(part, v, tempo, key=None, exclude=(), top=400):
    """The licensed loop of this part's family nearest the raw vector v, among loops at a tempo within 12% (or half or
    double) of the track's, preferring a small stretch and, for tonal parts, a small transposition."""
    L = loop_index()[FAM[part]]
    q, zname = _space(L, v)
    Zl = np.asarray(L[zname], np.float32); sim = Zl @ q
    ex = {str(x) for x in (exclude or ())}
    cands = []
    for i in np.argsort(-sim)[:top]:
        m = L["meta"][int(i)]; lt = float(m.get("tempo") or 0)
        if not m.get("preview") or not lt or str(m.get("id")) in ex: continue
        f = min((1.0, 2.0, 0.5), key=lambda x: abs(lt * x - tempo))
        r = lt * f / tempo
        if abs(r - 1) > 0.12: continue
        s = key_shift(m.get("key"), key) if (part in ("bass", "other", "vocals") and key) else 0
        score = float(sim[i]) - 0.6 * abs(r - 1) - 0.012 * abs(s)
        cands.append((score, int(i), f, s))
    cands.sort(reverse=True)
    return [{"i": i, "meta": L["meta"][i], "sim": float(sim[i]), "q": q, "f": f, "shift": s, "of": len(L["meta"]),
             "rank": int(np.sum(sim > sim[i])) + 1} for sc, i, f, s in cands]


def _rubberband(x, ratio, shift, w):
    """Time-stretch by ratio (output length over input length) and shift by semitones, with Rubber Band's finer engine;
    librosa when the command is missing."""
    import soundfile as sf
    if abs(ratio - 1) < 1e-4 and not shift: return x
    a, b = os.path.join(w, "rb_in.wav"), os.path.join(w, "rb_out.wav")
    sf.write(a, x, SR, subtype="FLOAT")
    for extra in (["-3"], []):
        try:
            subprocess.run(["rubberband", "-q", *extra, "-t", f"{ratio:.6f}", "-p", f"{shift:.3f}", a, b], check=True, timeout=120,
                           capture_output=True)
            y, _ = read_wav(b); return y
        except Exception:
            continue
    import librosa
    ch = []
    for c in range(2):
        y = librosa.effects.time_stretch(x[:, c], rate=1 / ratio)
        if shift: y = librosa.effects.pitch_shift(y, sr=SR, n_steps=shift)
        ch.append(y)
    n = min(len(c) for c in ch)
    return np.stack([c[:n] for c in ch], 1).astype(np.float32)


def prepare(meta, f, shift, tempo, min_fit=None):
    """A loop as one cycle of whole bars at the track's tempo: its bars read from its length and stated tempo, stretched
    to the track's tempo, moved by shift semitones, cut to exactly the cycle. With min_fit, a loop whose hits sit less
    squarely than that on a straight grid is refused before any stretching. Returns (cycle as float32 stereo, info)."""
    raw = urllib.request.urlopen(meta["preview"], timeout=30).read()
    return prepare_audio(decode_bytes(raw), meta, f, shift, tempo, min_fit)


class NotALoop(ValueError):
    pass


_PREP = {}; _PREP_LOCK = threading.Lock()


def prepare_cached(meta, f, shift, tempo, min_fit=None, keep=48):
    """prepare(), with this container's recent cycles kept (the page asks for a listed loop's audio a moment after the
    list), so a pick usually comes back at once. A cycle is kept only when it prepared; the grid gate is checked again
    on a hit."""
    k = (str(meta.get("id")), round(float(f), 4), int(shift), round(float(tempo), 3))
    with _PREP_LOCK: hit = _PREP.get(k)
    if hit:
        if min_fit is not None and hit[1]["grid_fit"] < min_fit: raise NotALoop("its hits do not sit on a straight grid")
        return hit[0], hit[1]
    y, prep = prepare(meta, f, shift, tempo, min_fit)
    with _PREP_LOCK:
        _PREP[k] = (y, prep, time.time())
        if len(_PREP) > keep:
            for kk in sorted(_PREP, key=lambda z: _PREP[z][2])[: keep // 3]: _PREP.pop(kk, None)
    return y, prep


def loop_bars(d, lt):
    """How many bars a loop of d seconds holds at its stated tempo lt (or half or double it, for loops tagged at the
    other octave): (bars, the multiple of lt, length over that many bars), for lengths from 4% short to 20% long of a
    whole 1, 2, 4, 8 or 16 bars (a tail after the last bar is common), or None. A loop that does not divide into whole
    bars at its tempo drifts off the beat, so it is not used."""
    best = None
    for q in (1.0, 2.0, 0.5):
        bars_f = d * lt * q / 240
        for N in (1, 2, 4, 8, 16):
            r = bars_f / N
            if 0.96 <= r <= 1.20 and (best is None or abs(math.log(r)) < abs(math.log(best[2]))): best = (N, q, r)
    return best


def _lead(x, tempo, max_s=0.06):
    """Where a loop's grid starts: the lead (0 to 60 ms: an MP3's encoder delay, or a late first hit) that puts the most
    of its hits within 15 ms of a straight sixteenth grid, and that share, against the share with no lead at all."""
    m = np.abs(x.mean(1)); hop = 64
    e = m[: len(m) // hop * hop].reshape(-1, hop).max(1); r = np.maximum(0, np.diff(e, prepend=e[:1]))
    tot = float(r.sum())
    if tot <= 0: return 0.0, 0.0, 0.0
    t = (np.arange(len(r)) + 0.5) * hop / SR; q = 60 / tempo / 4
    leads = np.arange(0, max_s + 1e-9, 0.001)
    d = np.abs(((t[None, :] - leads[:, None] + q / 2) % q) - q / 2)
    fits = (r[None, :] * (d <= 0.015)).sum(1) / tot
    # among the leads that keep nearly the most hits on the grid, the one that centres them (the window is 30 ms wide)
    spread = (r[None, :] * np.minimum(d, 0.03)).sum(1) / tot
    ok = np.where(fits >= fits.max() - 0.02)[0]
    k = int(ok[np.argmin(spread[ok])])
    return float(leads[k]), float(fits[k]), float(fits[0])


def loop_shape(x, meta, tempo):
    """A decoded loop made ready to lay: its lead trimmed when that puts its hits squarely on the grid, cut to whole
    bars, with the tempo its length says (or its stated tempo when it has a tail), whether it plays at half or double
    time against the track, and how squarely its hits sit on a straight sixteenth grid. Returns
    (audio, bars, tempo, f, fit, lead in seconds); raises NotALoop when it does not divide into whole bars or is too far
    from the track's tempo to stretch."""
    lt = float(meta.get("tempo") or tempo)
    lb = loop_bars(len(x) / SR, lt)
    if not lb: raise NotALoop("its length is not a whole number of bars at its tempo")
    N, q, r = lb; lq = lt * q
    lead, fit, fit0 = _lead(x, lq)
    if fit < 0.3 or fit < fit0 - 0.02: lead, fit = 0.0, fit0   # trim only where the hits show a grid, and never to its cost
    x = x[int(round(lead * SR)):]
    if r <= 1.04:
        lt_eff = N * 240 / (len(x) / SR)                  # a clean loop: its length says its exact tempo
    else:
        lt_eff = lq; x = x[:int(round(N * 240 / lt_eff * SR))]   # a tail after the last bar: cut at its stated tempo
    f = min((1.0, 2.0, 0.5), key=lambda k: abs(lt_eff * k - tempo))   # half or double time where that is nearer
    if abs(lt_eff * f / tempo - 1) > 0.15: raise NotALoop("too far from the track's tempo to stretch")
    return x, N, lt_eff, f, fit, lead


def prepare_audio(x, meta, f, shift, tempo, min_fit=None):
    x, N, lt_eff, f, fit, lead = loop_shape(x, meta, tempo)
    if min_fit is not None and fit < min_fit: raise NotALoop("its hits do not sit on a straight grid")
    cb = N * f                                    # the track's bars in one cycle
    C = int(round(cb * 240 / tempo * SR))         # the cycle in samples at the track's tempo
    ratio = C / len(x)                            # output length over input length
    # three cycles in a row, stretched, and the middle one kept: no edge for the stretcher to smear, and its end runs
    # into its own start exactly as it will when it repeats
    w = tempfile.mkdtemp()
    try:
        y = _rubberband(np.concatenate([x, x, x]), ratio, shift, w)
    finally:
        shutil.rmtree(w, ignore_errors=True)
    if len(y) >= 2 * C:
        out = y[C:2 * C].copy()
    else:
        out = np.zeros((C, 2), np.float32); m = min(C, len(y)); out[:m] = y[:m]
    pk = float(np.abs(out).max()) or 1.0
    if pk > 0.99: out *= 0.99 / pk
    m = out.mean(1); rms = float(np.sqrt(np.mean(m ** 2)))
    return out.astype(np.float32), {"bars": N, "f": f, "cycle_bars": cb, "ratio": round(ratio, 6), "shift": shift, "loop_tempo": round(lt_eff, 3),
                                    "samples": C, "rms_db": round(20 * math.log10(rms), 2) if rms > 1e-6 else -120.0, "grid_fit": round(fit, 3),
                                    "lead_ms": round(1000 * lead, 1)}


GRID_FIT_MIN = {"drums": 0.3}   # drums only: their hits are sharp enough to judge (about three in five clean drum loops pass)


def grid_fit(y, tempo):
    """How squarely a prepared cycle's attacks sit on a straight sixteenth-note grid that starts at its first sample:
    the share of the rise in its level that falls within 15 ms of a sixteenth. A quantised loop that starts on its first
    beat scores near 1; one that starts late, drifts or is played loose scores near a quarter, what chance gives; a
    swung loop, whose offbeat sixteenths are late, scores about a half."""
    m = np.abs(y.mean(1)); hop = 64
    e = m[: len(m) // hop * hop].reshape(-1, hop).max(1)
    r = np.maximum(0, np.diff(e, prepend=e[:1]))
    if r.sum() <= 0: return 0.0
    t = (np.arange(len(r)) + 0.5) * hop / SR
    q = 60 / tempo / 4
    d = np.abs(((t + q / 2) % q) - q / 2)
    return float(r[d <= 0.015].sum() / r.sum())


def _licence(u):
    u = str(u or "")
    if "publicdomain/zero" in u: return "CC0"
    import re
    m = re.search(r"licenses/([a-z-]+)/([\d.]+)", u)
    return f"CC {m.group(1).upper()} {m.group(2)}" if m else u


def _loop_entry(part, c, prep, info):
    m = c["meta"]
    e = {"loop": {"id": m.get("id"), "name": m.get("name"), "user": m.get("user"), "licence": _licence(m.get("license")),
                  "licence_url": m.get("license"), "page": f"https://freesound.org/s/{m.get('id')}/", "tempo": m.get("tempo"), "key": m.get("key")},
         "prep": prep, "closeness": round(c["sim"], 3), "rank": c["rank"], "of": c["of"]}
    if info.get("cz") is not None:   # against the artist's average itself (whatever the target was steered to), as their own records are
        L = loop_index()[FAM[part]]; z = np.asarray(L["Zp" if "Zp" in L else ("Zs" if "Zs" in L else "Z")][c["i"]], np.float32)
        sim_art = float(z @ info["cz"]); e["closer_than"] = round(float(np.mean(info["own"] < sim_art)), 3); e["closeness_to_artist"] = round(sim_art, 3)
    return e


def _target_vector(part, target, info):
    """The raw part vector the loops are ranked against: your part, an artist's average, or a point between the two
    (w: 1 is their average, 0 is yours)."""
    t = target or {}
    if t.get("kind") == "artist":
        v, info_ = artist_target(part, t.get("name")); info.update(info_)
        if v is None: return None
        y = t.get("vector"); w = t.get("w")
        if w is not None and isinstance(y, list) and len(y) == 53:
            w = min(1.0, max(0.0, float(w))); v = (1 - w) * np.asarray(y, float) + w * v; info["w"] = w
        else:
            info["w"] = 1.0
        return v
    v = t.get("vector")
    return np.asarray(v, float) if isinstance(v, list) and len(v) == 53 else None


def loop_for(part, target, tempo, key=None, exclude=(), n=5):
    """The nearest clean licensed loops for one part: target {"kind": "yours", "vector": [53]}, {"kind": "artist",
    "name": ...} or {"kind": "artist", "name": ..., "vector": [53], "w": 0.5} (a point between your part and the
    artist's average). Returns up to n in rank order, the first with its prepared cycle as WAV (base64), the others
    with their credit, closeness and preparation only (loop_audio gives their audio)."""
    if part not in PARTS: return {"error": "unknown part"}
    tempo = float(tempo or 0)
    if not (60 <= tempo <= 200): return {"error": "no tempo"}
    info = {}
    v = _target_vector(part, target, info)
    if v is None: return {"error": info.get("error") or "no part measures"}
    cands = choose(part, v, tempo, key, exclude)
    if not cands: return {"error": "no licensed loop at a tempo that stretches to this one"}
    gate = GRID_FIT_MIN.get(part); found = []; skipped = 0
    from concurrent.futures import ThreadPoolExecutor
    def prep_one(c):
        try: return c, prepare_cached(c["meta"], c["f"], c["shift"] if part != "drums" else 0, tempo, gate)
        except Exception: return c, None
    with ThreadPoolExecutor(4) as ex:   # eight at a time, in rank order, until n are clean (at most 24 tried)
        for lo in range(0, min(len(cands), 24), 8):
            for c, r in ex.map(prep_one, cands[lo:lo + 8]):
                if r is None: skipped += 1; continue
                found.append((c, r[0], r[1]))
                if len(found) >= n: break
            if len(found) >= n: break
    if not found: return {"error": "none of the nearest licensed loops is clean enough to lay on your bars"}
    c0, y0, p0 = found[0]
    out = {"part": part, **_loop_entry(part, c0, p0, info), "skipped": skipped, "wav": base64.b64encode(wav_bytes(y0)).decode(),
           "others": [_loop_entry(part, c, prep, info) for c, _y, prep in found[1:]]}
    if info.get("artist"):
        own = info["own"]
        out["artist"] = {"name": info["artist"], "records": info["records"], "w": info.get("w", 1.0),
                         "closer_than": out.get("closer_than"), "own_median": round(float(np.median(own)), 3)}
    return out


def loop_audio(part, lid, f, shift, tempo):
    """One listed loop's prepared cycle (WAV, base64) with its preparation: what loop_for gave for the first loop, for
    any of the others when the page picks it."""
    if part not in PARTS: return {"error": "unknown part"}
    m = loop_by_id(part, lid)
    if not m: return {"error": "no such loop"}
    try:
        y, prep = prepare_cached(m, float(f or 1.0), int(shift or 0) if part != "drums" else 0, float(tempo))
    except Exception as e:
        return {"error": "that loop could not be prepared: " + str(e)[:120]}
    return {"part": part, "loop": {"id": m.get("id"), "name": m.get("name"), "user": m.get("user"), "licence": _licence(m.get("license")),
                                   "licence_url": m.get("license"), "page": f"https://freesound.org/s/{m.get('id')}/", "tempo": m.get("tempo"), "key": m.get("key")},
            "prep": prep, "wav": base64.b64encode(wav_bytes(y)).decode()}




# ---------------------------------------------------------------- following your notes (your line or pattern, the loop's sound)
NOTE_RANGES = {"bass": (35, 330), "other": (80, 1500), "vocals": (80, 1100)}

def _rise_peaks(r, fr, wait_s=0.06, rel=0.2):
    """Local maxima of a rise envelope at least rel of its 95th-percentile peak, at least wait_s apart (the stronger kept)."""
    thr = rel * (np.percentile(r[r > 0], 95) if (r > 0).any() else 1.0)
    cand = np.where((r[1:-1] >= r[:-2]) & (r[1:-1] > r[2:]) & (r[1:-1] >= thr))[0] + 1
    out = []; wait = int(wait_s * fr)
    for i in cand:
        if out and i - out[-1] < wait:
            if r[i] > r[out[-1]]: out[-1] = i
            continue
        out.append(i)
    return np.array(out, int)

def notes_of(x, part, g0, bar, nb, sr_in=SR, delay=0.0):
    import librosa
    """Per sixteenth of each bar: the MIDI note sounding there (int) or None. pyin on the mono part at 22050 Hz gives a
    pitch per frame; a step is voiced when 40% of its middle frames are; a note runs from an onset (a peak in the rise
    of the part's level) or the first voiced step to the next onset or silence, and every step of it takes the pitch
    of the run's loudest frames (the median over the frames above the run's median level), so a bent or wobbling note
    reads as one note. Returns (notes, onsets)."""
    sr = 22050; y = librosa.resample(x.mean(1).astype(np.float32), orig_sr=sr_in, target_sr=sr) if sr_in != sr else x.mean(1).astype(np.float32)
    fmin, fmax = NOTE_RANGES[part]; hop = 256; fr = sr / hop; step = bar / 16
    f0, voiced, prob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=hop)
    midi = librosa.hz_to_midi(np.where(voiced, f0, np.nan))
    e = np.abs(y); eh = e[: len(e) // hop * hop].reshape(-1, hop).max(1); rise = np.maximum(0, np.diff(eh, prepend=eh[:1]))
    pk = _rise_peaks(rise, fr, 0.06, 0.2); pk_steps = set(int(round((i / fr - delay - g0) / step)) for i in pk)
    def frames(k):
        t0 = g0 + delay + (k // 16) * bar + (k % 16) * step; return int((t0 + 0.15 * step) * fr), int((t0 + 0.85 * step) * fr)
    raw = []
    for k in range(nb * 16):
        i0, i1 = frames(k); seg = midi[i0:i1]
        raw.append(np.isfinite(seg).mean() >= 0.4 if len(seg) else False)
    notes = [None] * len(raw); onsets = [False] * len(raw); i = 0
    while i < len(raw):
        if not raw[i]: i += 1; continue
        j = i + 1
        while j < len(raw) and raw[j] and j not in pk_steps: j += 1
        i0, _ = frames(i); _, i1 = frames(j - 1); m = midi[i0:i1]; lv = eh[i0:i1][: len(m)]
        ok = np.isfinite(m)
        if ok.any():
            loud = lv >= np.median(lv[ok]) if ok.sum() > 2 else np.ones(len(m), bool)
            sel = m[ok & loud] if (ok & loud).any() else m[ok]
            n = int(round(float(np.median(sel))))
            for k in range(i, j): notes[k] = n
            onsets[i] = True
        i = j
    return notes, onsets

def hits_of(x, g0, bar, nb, sr_in=SR, delay=0.0):
    from scipy.signal import butter, sosfiltfilt
    """Per sixteenth: a kick and a snare. Hits are peaks in the rise of the drums' level (full band, 1.5 ms frames);
    each is sorted by the sound in its first 40 ms: a kick when the 30-120 Hz band holds at least 40% of the loudest
    nearby hit's (a hit within 100 ms after a kick is that kick's swell); a snare (or clap, with or without a kick under it) when 150-800 Hz holds a quarter and 2-8 kHz
    three tenths of the loudest nearby hit's. Each lands on its nearest step. Returns (kick, snare) as lists of 0/1."""
    m = x.mean(1).astype(np.float32); hop = 64; fr = sr_in / hop; step = bar / 16; n = nb * 16
    e = np.abs(m); eh = e[: len(e) // hop * hop].reshape(-1, hop).max(1); rise = np.maximum(0, np.diff(eh, prepend=eh[:1]))
    pk = _rise_peaks(rise, fr, 0.04, 0.08)
    def band(lo, hi): return np.abs(sosfiltfilt(butter(4, [lo, hi], "band", fs=sr_in, output="sos"), m))
    L, M, H = band(30, 120), band(150, 800), band(2000, 8000); W = int(0.04 * sr_in)
    rows = []
    for i in pk:
        s0 = int(i * hop); s1 = s0 + W
        rows.append((i / fr - delay, float(L[s0:s1].mean()), float(M[s0:s1].mean()), float(H[s0:s1].mean()), float(rise[i])))
    kick = np.zeros(n, int); snare = np.zeros(n, int)
    if not rows: return kick.tolist(), snare.tolist()
    T = np.array([r[0] for r in rows]); Lv = np.array([r[1] for r in rows]); Mv = np.array([r[2] for r in rows]); Hv = np.array([r[3] for r in rows])
    # kicks first: low band at least half the loudest nearby and outweighing the mids; a second kick within 100 ms is the
    # same kick's swell
    Rv = np.array([r[4] for r in rows])
    isk = np.zeros(len(T), bool); last_k = -1.0; last_r = 0.0
    for q, t in enumerate(T):
        near = np.abs(T - t) <= 4 * bar; lref = Lv[near].max() + 1e-9
        # within 150 ms of a kick, only a rise as sharp as that kick's (60%) is a new kick; a softer one is its swell
        if Lv[q] >= 0.4 * lref and (t - last_k > 0.15 or Rv[q] >= 0.6 * last_r): isk[q] = True; last_k = t; last_r = Rv[q]
    tail = np.zeros(len(T), bool)   # a hit within 100 ms after a kick is that kick's tail
    kt = T[isk]
    for q, t in enumerate(T):
        if not isk[q] and kt.size and ((t - kt > 0) & (t - kt <= 0.15)).any(): tail[q] = True
    for q, t in enumerate(T):
        k = int(round((t - g0) / step))
        if not (0 <= k < n): continue
        if isk[q]: kick[k] = 1
        if tail[q]: continue
        # a snare or clap, with or without a kick under it: mids and highs both strong against the loudest nearby hit
        # (a kick alone has little above 2 kHz; a hat has little between 150 and 800 Hz)
        near = (np.abs(T - t) <= 4 * bar) & ~tail
        mref = Mv[near].max() + 1e-9; href = Hv[near].max() + 1e-9
        if Mv[q] >= 0.25 * mref and Hv[q] >= 0.3 * href: snare[k] = 1
    return kick.tolist(), snare.tolist()


def loop_notes(cycle, part, tempo, cycle_bars):
    """The prepared cycle's own notes per sixteenth (its grid starts at its first sample)."""
    steps = int(round(cycle_bars * 16)); bar = 240 / tempo
    n, _on = notes_of(cycle, part, 0.0, bar, max(1, int(math.ceil(cycle_bars))))
    return n[:steps] if len(n) >= steps else n + [None] * (steps - len(n))


def loop_kit(cycle, tempo):
    """A drum loop's kit: its kick (the loudest kick hit, to the next hit or 350 ms), its snare (likewise, the loudest
    hit that is not a kick), each faded at its end, and its top (the cycle above 1.8 kHz: hats and the rest)."""
    from scipy.signal import butter, sosfiltfilt
    m = cycle.mean(1).astype(np.float32); hop = 64; fr = SR / hop
    e = np.abs(m); eh = e[: len(e) // hop * hop].reshape(-1, hop).max(1); rise = np.maximum(0, np.diff(eh, prepend=eh[:1]))
    pk = _rise_peaks(rise, fr, 0.04, 0.08)
    def band(lo, hi): return np.abs(sosfiltfilt(butter(4, [lo, hi], "band", fs=SR, output="sos"), m))
    L, M, H = band(30, 120), band(150, 800), band(2000, 8000); W = int(0.04 * SR)
    hits = [(int(i * hop), float(L[int(i * hop): int(i * hop) + W].mean()), float(M[int(i * hop): int(i * hop) + W].mean()), float(H[int(i * hop): int(i * hop) + W].mean()), float(rise[i])) for i in pk]
    def take(q):
        """From hit q to the next hit whose rise is as sharp as its own (60%: a new attack, not a swell), or 350 ms."""
        s0 = hits[q][0]; e_ = min(len(cycle), s0 + int(0.35 * SR))
        for h in hits[q + 1:]:
            if h[4] >= 0.6 * hits[q][4]: e_ = min(e_, h[0]); break
        seg = cycle[s0:e_].copy(); F = min(len(seg), int(0.01 * SR))
        if F > 1: seg[-F:] *= np.linspace(1, 0, F)[:, None]
        return seg
    kit = {"kick": None, "snare": None, "top": sosfiltfilt(butter(4, 1800, "high", fs=SR, output="sos"), cycle, axis=0).astype(np.float32)}
    if hits:
        lref = max(h[1] for h in hits) + 1e-9; isk = [h[1] >= 0.4 * lref for h in hits]
        tail = [False] * len(hits)   # a softer hit within 150 ms after a kick is its swell
        for q, h in enumerate(hits):
            if isk[q]: continue
            for q2 in range(q - 1, -1, -1):
                if h[0] - hits[q2][0] > 0.15 * SR: break
                if isk[q2] and h[4] < 0.6 * hits[q2][4]: tail[q] = True; break
        ks = [q for q in range(len(hits)) if isk[q]]
        if ks: kit["kick"] = take(max(ks, key=lambda q: hits[q][1]))
        sn = [q for q, h in enumerate(hits) if not isk[q] and not tail[q] and h[2] > 0 and h[3] > 0]
        if sn:
            mref = max(hits[q][2] for q in sn) + 1e-9; sn = [q for q in sn if hits[q][2] >= 0.33 * mref]
            kit["snare"] = take(max(sn, key=lambda q: hits[q][2] + hits[q][3]))
    return kit


def _tile(cycle, n):
    out = np.zeros((n, 2), np.float32); C = len(cycle); k = 0
    while k < n:
        m = min(C, n - k); out[k:k + m] = cycle[:m]; k += C
    return out


def follow_tonal(cycle, cycle_bars, lnotes, your, tempo, bars, part):
    """The cycle, tiled over bars whole bars and re-pitched sixteenth by sixteenth to your notes with Rubber Band's
    frequency map: at each step the loop's own note there is shifted to your note, your line moved whole octaves to
    the loop's register (the octave that puts the median of your notes nearest the median of the loop's). Where you
    hold no note, your last note in the bar holds; before your first note, the loop plays its own. Returns (audio,
    info): the steps followed, the shift range in semitones."""
    bar = 240 / tempo; n = int(round(bars * bar * SR)); x = _tile(cycle, n)
    steps = int(round(cycle_bars * 16)); ln = [v for v in lnotes if v is not None]; yn = [v for v in your if v is not None]
    if not ln or not yn: return x, {"followed": 0, "why": "no notes read in the " + ("loop" if not ln else "part")}
    lmed = float(np.median(ln)); off = 12 * int(round((lmed - float(np.median(yn))) / 12))
    ratios = []; last = None; followed = 0; shifts = []
    for k in range(bars * 16):
        if k % 16 == 0: last = None
        y = your[k] if k < len(your) else None
        if y is not None: last = y
        l = lnotes[k % steps] if steps else None
        if last is None: r = 1.0
        else:
            src = l if l is not None else lmed; st = (last + off) - src; shifts.append(st); r = float(2 ** (st / 12)); followed += 1
        ratios.append(min(4.0, max(0.25, r)))
    step_n = n / (bars * 16); lines = []; prev = None
    for k, r in enumerate(ratios):
        if prev is None or abs(r - prev) > 1e-6: lines.append(f"{int(round(k * step_n))} {r:.6f}"); prev = r
    if len(lines) <= 1 and (prev is None or abs(prev - 1) < 1e-6): return x, {"followed": followed, "shift_semitones": [0, 0]}
    import soundfile as sf
    w = tempfile.mkdtemp()
    try:
        a, b, mp = os.path.join(w, "in.wav"), os.path.join(w, "out.wav"), os.path.join(w, "map.txt")
        sf.write(a, x, SR, subtype="FLOAT"); open(mp, "w").write("\n".join(lines) + "\n")
        subprocess.run(["rubberband", "-q", "-3", "--freqmap", mp, a, b], check=True, timeout=600, capture_output=True)
        y, _sr = sf.read(b, always_2d=True, dtype="float32")
    finally:
        shutil.rmtree(w, ignore_errors=True)
    out = np.zeros((n, 2), np.float32); m = min(n, len(y)); out[:m] = y[:m, :2]
    pk = float(np.abs(out).max()) or 1.0
    if pk > 0.99: out *= 0.99 / pk
    return out, {"followed": followed, "of": bars * 16, "shift_semitones": [round(min(shifts), 1), round(max(shifts), 1)] if shifts else [0, 0], "octave": off}


def follow_drums(cycle, kit, kick, snare, tempo, bars):
    """The loop's top tiled over bars whole bars, with its kick at your kicks and its snare at your snares (per
    sixteenth). Returns (audio, info)."""
    bar = 240 / tempo; n = int(round(bars * bar * SR)); out = _tile(kit["top"], n) if kit.get("top") is not None else np.zeros((n, 2), np.float32)
    step = bar / 16; placed = {"kick": 0, "snare": 0}
    for name, flags in (("kick", kick), ("snare", snare)):
        smp = kit.get(name)
        if smp is None or not len(smp): continue
        for k, f in enumerate(flags[: bars * 16]):
            if not f: continue
            s0 = int(round(k * step * SR)); m = min(len(smp), n - s0)
            if m <= 0: break
            out[s0:s0 + m] += smp[:m]; placed[name] += 1
    pk = float(np.abs(out).max()) or 1.0
    if pk > 0.99: out *= 0.99 / pk
    return out, {"placed": placed, "has": {"kick": kit.get("kick") is not None, "snare": kit.get("snare") is not None}}


def follow_part(part, meta, f, shift, tempo, bars, your):
    """A listed loop following your notes (bass, melody, voice: notes per sixteenth) or your pattern (drums: kick and
    snare flags per sixteenth) over bars whole bars from a bar line. Returns (audio, prep, info)."""
    y, prep = prepare_cached(meta, float(f or 1.0), int(shift or 0) if part != "drums" else 0, float(tempo))
    if part == "drums":
        kit = loop_kit(y, tempo); out, info = follow_drums(y, kit, your.get("kick") or [], your.get("snare") or [], tempo, bars)
    else:
        ln = loop_notes(y, part, tempo, prep["cycle_bars"]); out, info = follow_tonal(y, prep["cycle_bars"], ln, your.get("notes") or [], tempo, bars, part)
        info["loop_notes"] = sum(1 for v in ln if v is not None); info["loop_steps"] = len(ln)
    m = out.mean(1); rms = float(np.sqrt(np.mean(m ** 2)))
    info["rms_db"] = round(20 * math.log10(rms), 2) if rms > 1e-6 else -120.0
    return out, prep, info


def follow_audio(part, lid, f, shift, tempo, bars, your):
    """The /follow endpoint: a listed loop following your notes or pattern over bars whole bars, as MP3 (base64) with
    the loop's preparation and what was followed."""
    if part not in PARTS: return {"error": "unknown part"}
    m = loop_by_id(part, lid)
    if not m: return {"error": "no such loop"}
    try:
        out, prep, info = follow_part(part, m, f, shift, float(tempo), int(bars), your or {})
    except Exception as e:
        return {"error": "could not follow: " + type(e).__name__ + ": " + str(e)[:120]}
    return {"part": part, "loop": {"id": m.get("id"), "name": m.get("name"), "user": m.get("user")}, "prep": prep, "info": info,
            "bars": int(bars), "mp3": base64.b64encode(mp3_bytes(out)).decode()}


# ---------------------------------------------------------------- the whole track, by the plan the page played
def clean_plan(plan, nb_max=10000):
    """The page's plan, checked. Per part: nothing (keep yours), or one or more layers, each a loop id with the bars it
    plays in and a gain for each bar (a layer per section where the page chose differently by section). "keep": the
    share of your own part heard under a swap, 0 to 1. "moves": cuts and rolls at phrase ends (phrase length, which
    bar the phrases count from, and the parts each applies to). "master": loudness matched to the track and limited."""
    g = plan.get("grid") or {}
    grid = {"g0": float(g.get("g0") or 0), "bar": float(g.get("bar") or 0), "nb": int(g.get("nb") or 0)}
    if not (0.5 < grid["bar"] < 6 and 0 <= grid["g0"] < grid["bar"] and 0 < grid["nb"] <= nb_max): raise ValueError("grid")
    parts = {}
    for p in PARTS:
        c = (plan.get("parts") or {}).get(p)
        layers = c if isinstance(c, list) else ([c] if c else [])
        L = []
        for lay_ in layers[:8]:
            if not isinstance(lay_, dict) or not lay_.get("loop"): continue
            gains = [min(8.0, max(0.0, float(x))) for x in (lay_.get("gains") or [])][:grid["nb"]]
            gains += [0.0] * (grid["nb"] - len(gains))
            runs = []
            for r in (lay_.get("runs") or [])[:2000]:
                r0, r1 = int(r[0]), int(r[1])
                if 0 <= r0 < r1 <= grid["nb"]: runs.append((r0, r1))
            if runs and any(gains): L.append({"loop": int(lay_["loop"]), "f": float(lay_.get("f") or 1), "shift": int(lay_.get("shift") or 0), "gains": gains, "runs": runs, "follow": bool(lay_.get("follow"))})
        if L: parts[p] = L
    keep = {}
    for p in PARTS:
        k = (plan.get("keep") or {}).get(p)
        keep[p] = min(1.0, max(0.0, float(k))) if k is not None else 0.0
    mv = plan.get("moves") or {}
    phrase = int(mv.get("phrase") or 8); phrase = phrase if phrase in (4, 8, 16) else 8
    moves = {"phrase": phrase, "offset": int(mv.get("offset") or 0) % phrase,
             "cut": [p for p in PARTS if p in (mv.get("cut") or [])], "roll": [p for p in PARTS if p in (mv.get("roll") or [])]}
    return {"grid": grid, "parts": parts, "keep": keep, "moves": moves, "master": bool(plan.get("master", True)),
            "tempo": float(plan.get("tempo") or 0), "name": str(plan.get("name") or "track")[:100]}


def loop_by_id(part, lid):
    L = loop_index()[FAM[part]]
    for m in L["meta"]:
        if int(m.get("id") or -1) == int(lid): return m
    return None


def phrase_ends(nb, phrase, offset):
    """The last bar of each phrase: bars b with (b - offset) % phrase == phrase - 1."""
    return [b for b in range(nb) if (b - offset) % phrase == phrase - 1]


def apply_moves(out, g0, bar, gains, runs, nb, moves, part, ramp_s=0.005):
    """Cuts and rolls at phrase ends on a laid part, before its gain envelope. A cut: the last beat of the phrase's last
    bar is silent (five-millisecond ramp). A roll: that beat is the loop's own first sixteenth of it, played as two
    sixteenths then four thirty-seconds, each with a two-millisecond fade (a fill from the loop's own material). Only
    in bars where the layer plays."""
    if not moves or (part not in moves.get("cut", ()) and part not in moves.get("roll", ())): return
    active = np.zeros(nb, bool)
    for r0, r1 in runs: active[r0:r1] = True
    beat = bar / 4; six = beat / 4; n = len(out); R = max(1, int(ramp_s * SR)); F = max(1, int(0.002 * SR))
    for b in phrase_ends(nb, moves["phrase"], moves["offset"]):
        if not active[b] or gains[b] <= 0: continue
        s4, e = int(round((g0 + b * bar + 3 * beat) * SR)), min(n, int(round((g0 + (b + 1) * bar) * SR)))
        if s4 >= n or e <= s4: continue
        if part in moves.get("roll", ()):
            sl = out[s4: s4 + int(round(six * SR))].copy()
            if len(sl) > 2 * F:
                sl[:F] *= np.linspace(0, 1, F)[:, None]
                seg = np.zeros((e - s4, 2), np.float32); pos = 0
                for ln in (six, six, six / 2, six / 2, six / 2, six / 2):
                    m = min(int(round(ln * SR)), len(sl), e - s4 - pos)
                    if m <= 0: break
                    piece = sl[:m].copy(); piece[-F:] *= np.linspace(1, 0, min(F, m))[:, None]
                    seg[pos:pos + m] = piece; pos += m
                out[s4:e] = seg
        if part in moves.get("cut", ()):
            r0 = max(0, s4 - R); out[r0:s4] *= np.linspace(1, 0, s4 - r0)[:, None]; out[s4:e] = 0


def lay(cycle, n, g0, bar, gains, runs, ramp_s=0.005, moves=None, part=None, timed=None):
    """A swapped part for the whole track: the cycle from the start of each run of bars, repeated to its end (or, with
    timed, audio already laid out from the first bar line: a part following your notes), at each bar's gain, with
    five-millisecond ramps where the gain changes (as the page schedules it), and the moves applied."""
    out = np.zeros((n, 2), np.float32)
    if timed is not None:
        s0 = int(round(g0 * SR)); m = min(len(timed), n - s0)
        if m > 0: out[s0:s0 + m] = timed[:m]
    else:
        C = len(cycle)
        for r0, r1 in runs:
            s0, s1 = int(round((g0 + r0 * bar) * SR)), min(n, int(round((g0 + r1 * bar) * SR)))
            k = s0
            while k < s1:
                m = min(C, s1 - k); out[k:k + m] += cycle[:m]; k += C
    if moves: apply_moves(out, g0, bar, gains, runs, len(gains), moves, part, ramp_s)
    return out * bar_envelope(n, g0, bar, gains, ramp_s)[:, None]


def bar_envelope(n, g0, bar, gains, ramp_s=0.005):
    """A gain for every sample from a gain per bar, each change ramped in over the last five milliseconds of the bar
    before, so a bar's first hit has its own gain."""
    env = np.zeros(n, np.float32); R = max(1, int(ramp_s * SR)); prev = 0.0
    for b, gv in enumerate(gains):
        s0, s1 = int(round((g0 + b * bar) * SR)), min(n, int(round((g0 + (b + 1) * bar) * SR)))
        if s0 >= n: break
        env[s0:s1] = gv
        if gv != prev:
            r0 = max(0, s0 - R); env[r0:s0] = np.linspace(prev, gv, s0 - r0, endpoint=False)
        prev = gv
    if gains and int(round((g0 + len(gains) * bar) * SR)) < n: env[int(round((g0 + len(gains) * bar) * SR)):] = prev
    return env


def render(path, plan):
    """The whole track with the plan's loops in place of those parts, as MP3 (320 kbps) with the loops' credits."""
    import soundfile as sf
    from features import stems as S
    P = clean_plan(plan); t0 = time.time()
    tempo = P["tempo"] or 240 / P["grid"]["bar"]
    g0, bar, nb = P["grid"]["g0"], P["grid"]["bar"], P["grid"]["nb"]
    w = tempfile.mkdtemp()
    try:
        wav = os.path.join(w, "track.wav"); decode_to_wav(path, wav)
        st = S.separate(wav, os.path.join(w, "sep"))
        X = {k: read_wav(st[k])[0] for k in PARTS if k in st}
        n = min(len(v) for v in X.values()); X = {k: v[:n] for k, v in X.items()}
        original = sum(X.values())
        out = np.zeros((n, 2), np.float32); credits = []; laid = {}; lines = {}
        def your_line(p):   # what your part plays, read once from the fresh separation on the plan's bar lines
            if p not in lines:
                if p == "drums": kk, ss = hits_of(X[p], g0, bar, nb); lines[p] = {"kick": kk, "snare": ss}
                else: nts, _on = notes_of(X[p], p, g0, bar, nb); lines[p] = {"notes": nts}
            return lines[p]
        for p in PARTS:
            layers = P["parts"].get(p) or []
            swapped = np.zeros(nb, bool)
            for c in layers:
                m = loop_by_id(p, c["loop"])
                if not m: continue
                try:
                    if c.get("follow"):
                        y, prep, finfo = follow_part(p, m, c["f"], c["shift"], tempo, nb, your_line(p))
                        out += lay(None, n, g0, bar, c["gains"], c["runs"], moves=P["moves"], part=p, timed=y)
                        laid.setdefault(p + "_follow", []).append(finfo)
                    else:
                        y, prep = prepare_cached(m, c["f"], c["shift"] if p != "drums" else 0, tempo)
                        out += lay(y, n, g0, bar, c["gains"], c["runs"], moves=P["moves"], part=p)
                except Exception as e:
                    laid.setdefault("errors", []).append(f"{p} {c['loop']}: {type(e).__name__}: {str(e)[:80]}"); continue
                swapped |= np.asarray(c["gains"]) > 0
                cr = f"{m.get('name')} by {m.get('user')} (freesound.org/s/{m.get('id')}), {_licence(m.get('license'))}"
                if cr not in credits: credits.append(cr)
            # your own part: whole where nothing is swapped in the bar, at the kept share under a swap
            if swapped.any():
                yours = np.where(swapped, P["keep"][p], 1.0).tolist()
                out += X[p] * bar_envelope(n, g0, bar, yours)[:, None]
                laid[p] = {"layers": len(layers), "bars_swapped": int(swapped.sum()), "keep": P["keep"][p]}
            else:
                out += X[p]
        loud = None; af = None
        if P["master"] and laid:
            ref, _pk = loudness_lufs(original); cur, _pk2 = loudness_lufs(out)
            gain_db = 0.0
            if ref is not None and cur is not None and ref > -70 and cur > -70: gain_db = max(-12.0, min(12.0, ref - cur))
            out *= 10 ** (gain_db / 20)
            # a lookahead limiter at -2 dBFS: MP3 coding adds up to a decibel of overshoot, so the file stays under full scale
            af = "alimiter=limit=0.8:attack=5:release=60:level=false"
            loud = {"track_lufs": ref, "swapped_lufs_before": cur, "gain_db": round(gain_db, 2), "ceiling_dbfs": -1.9}
        else:
            pk = float(np.abs(out).max()) or 1.0
            if pk > 0.98: out *= 0.98 / pk
        meta = {"title": f"{P['name']} (swapped)", "comment": "Loops: " + "; ".join(credits) if credits else "No parts swapped"}
        data = mp3_bytes(out, kbps=320, meta=meta, af=af)
        return {"name": P["name"], "seconds": round(n / SR, 1), "credits": credits, "mp3": data, "took": round(time.time() - t0, 1),
                "laid": laid, "moves": P["moves"], "loudness": loud}
    finally:
        shutil.rmtree(w, ignore_errors=True)
