"""Swap: a track of yours split into its parts, any part swapped for the openly licensed loop nearest your own part or
nearest an artist's version of it, laid where your part plays, and the whole track rendered to download.

Where each step runs (worker/modal_app.py wires them to the swap endpoints):
  read_track(path)    the parts (htdemucs, as every record was split), a steady beat grid and where bars start, each
                      part's level bar by bar, its measures on a listening window (the corpus measured two-minute
                      previews, so the window is measured the same way), the licensed loops and the artists nearest
                      each part, and the window of each part as MP3 for the page to play.
  loop_for(...)       one licensed loop for one part and one target (your part, or the average of an artist's separated
                      records' same part), stretched to your tempo and moved to your key with Rubber Band, cut to whole
                      bars. No audio of yours is involved.
  render(path, plan)  separates again (nothing is kept between calls) and lays the chosen loops by the plan the page
                      played: the same bars, the same gains, the same loop processing.
The sounds come only from the loop library (Freesound, CC0 and CC BY), never from another record.
"""
import os, io, json, math, time, hmac, hashlib, base64, subprocess, tempfile, shutil, urllib.request
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


def mp3_bytes(x, sr=SR, kbps=160, meta=None):
    """float32 stereo -> MP3 bytes (libmp3lame, constant bit rate)."""
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    cmd = ["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", str(sr), "-ac", "2", "-i", "pipe:0", "-c:a", "libmp3lame", "-b:a", f"{kbps}k"]
    for k, v in (meta or {}).items(): cmd += ["-metadata", f"{k}={v}"]
    p = subprocess.run(cmd + ["-f", "mp3", "pipe:1"], input=pcm, capture_output=True, check=True, timeout=600)
    return p.stdout


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
    drums, its grid refitted to the kicks), read on forty seconds from the middle; None when it cannot say."""
    import soundfile as sf
    from features import stems as S
    n = len(drums); m = int(40 * SR)
    seg = drums[max(0, n // 2 - m // 2): max(0, n // 2 - m // 2) + m]
    w = tempfile.mkdtemp()
    try:
        fp = os.path.join(w, "mid_drums.wav"); sf.write(fp, seg, SR, subtype="PCM_16")
        r = S.rhythm_of_stem(fp) or {}
        T = float(r.get("beats_per_minute") or 0)
        return T if 50 <= T <= 250 else None
    except Exception:
        return None
    finally:
        shutil.rmtree(w, ignore_errors=True)


def beat_grid(drums, mix, dur, T0=None):
    """A steady beat grid for the whole track. Kicks are not on every beat in a broken beat, so the grid is read from
    every hit of the drum part (the mix when the drums are near silent), which sit on eighth notes in nearly every style:
    the rough tempo as the corpus reads it (T0; librosa's tracker when there is none), folded into 86-186 bpm; the exact
    period and the eighth-note phase from the hits' Fourier component at the eighth-note rate, searched within 4% in
    steps of 0.01%; which eighths are beats from where the kicks and snares fall (30-500 Hz), on the beats more than
    between them in house, techno, garage, breaks and drum and bass alike; then a straight line through the hits nearest
    each eighth, timed in the waveform. Returns (first beat at or after 0 s, period in seconds)."""
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
    p = pb; a = (-ang / (2 * np.pi) * (p / 2)) % (p / 2)
    # the hits: peaks of the envelope, placed between frames by a parabola through the peak and its neighbours, then
    # timed in the waveform; a line through those nearest each eighth note
    pk = _peaks(full, wait=int(0.4 * (p / 2) * fr))
    if len(pk) >= 16:
        strong = pk[full[pk] >= 0.25 * np.percentile(full[pk], 90)]
        strong = strong[(strong > 0) & (strong < len(full) - 1)]
        y0, y1, y2 = full[strong - 1], full[strong], full[strong + 1]
        den = y0 - 2 * y1 + y2; off = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
        tp = _attack_times(y22, (strong + np.clip(off, -0.5, 0.5)) / fr)
        a, h = _fit_line(tp, a, p / 2); p = 2 * h
    # beats or the eighths between them: where the kicks and snares land
    def at_beats(a_):
        k = np.arange(max(0, int(math.ceil(-a_ / p))), int((dur - a_) / p))
        idx = ((a_ + k * p) * fr).astype(int); idx = idx[(idx >= 1) & (idx < len(lowmid) - 3)]
        return float(sum(lowmid[i - 1:i + 3].max() for i in idx))
    if at_beats(a + p / 2) > at_beats(a): a = a + p / 2
    while 60 / p < 86: p /= 2
    while 60 / p >= 186: p *= 2
    a = a - math.floor(a / p) * p      # the first beat at or after zero
    return float(a), float(p)


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
        T0 = rough_tempo(X["drums"]); timing["rough_tempo"] = round(time.time() - t_start, 1)
        a, p = beat_grid(X["drums"], mix, dur, T0); timing["grid"] = round(time.time() - t_start, 1)
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
                         "nbeats": nbeats, "rough_tempo": round(T0, 2) if T0 else None},
                "beat_levels": blevels,
                "window": {"b0": b0, "bars": W, "start_s": round(ws / SR, 5), "end_s": round(we / SR, 5)},
                "levels": levels, "vectors": vec, "plain": plain, "nearest": nearest, "like": like,
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
    return c, {"artist": nm, "records": len(rows), "own": own}


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


def loop_bars(d, lt):
    """How many bars a loop of d seconds holds at its stated tempo lt (or half or double it, for loops tagged at the
    other octave), and the tempo its length says exactly: (bars, tempo), or None when no whole 1, 2, 4, 8 or 16 bars
    fits within 4%. A loop that does not divide into whole bars at its tempo drifts off the beat, so it is not used."""
    best = None
    for q in (1.0, 2.0, 0.5):
        bars_f = d * lt * q / 240
        for N in (1, 2, 4, 8, 16):
            e = abs(bars_f / N - 1)
            if e <= 0.04 and (best is None or e < best[0]): best = (e, N, N * 240 / d)
    return (best[1], best[2]) if best else None


def loop_shape(x, meta, tempo):
    """A decoded loop's bars, the tempo its length says, whether it plays at half or double time against the track, and
    how squarely its hits sit on a straight sixteenth grid at its own tempo: (bars, tempo, f, fit). Raises NotALoop when
    it does not divide into whole bars or is too far from the track's tempo to stretch."""
    d = len(x) / SR
    lb = loop_bars(d, float(meta.get("tempo") or tempo))
    if not lb: raise NotALoop("its length is not a whole number of bars at its tempo")
    N, lt_eff = lb                                # a clean loop: its length says its exact tempo
    f = min((1.0, 2.0, 0.5), key=lambda q: abs(lt_eff * q - tempo))   # half or double time where that is nearer
    if abs(lt_eff * f / tempo - 1) > 0.15: raise NotALoop("too far from the track's tempo to stretch")
    return N, lt_eff, f, grid_fit(x, lt_eff)


def prepare_audio(x, meta, f, shift, tempo, min_fit=None):
    N, lt_eff, f, fit = loop_shape(x, meta, tempo)
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
                                    "samples": C, "rms_db": round(20 * math.log10(rms), 2) if rms > 1e-6 else -120.0, "grid_fit": round(fit, 3)}


GRID_FIT_MIN = 0.45   # a swung loop scores about 0.48 and passes; one that starts late or drifts does not


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


def loop_for(part, target, tempo, key=None, exclude=()):
    """One licensed loop for one part: target {"kind": "yours", "vector": [53]} or {"kind": "artist", "name": ...}.
    Returns its credit, why it was chosen in one measured number, and the prepared cycle as WAV (base64)."""
    if part not in PARTS: return {"error": "unknown part"}
    tempo = float(tempo or 0)
    if not (60 <= tempo <= 200): return {"error": "no tempo"}
    info = {}
    if (target or {}).get("kind") == "artist":
        v, info = artist_target(part, target.get("name"))
        if v is None: return info
    else:
        v = (target or {}).get("vector")
        if not (isinstance(v, list) and len(v) == 53): return {"error": "no part measures"}
    cands = choose(part, v, tempo, key, exclude)
    if not cands: return {"error": "no licensed loop at a tempo that stretches to this one"}
    y = None; skipped = 0
    for c in cands[:10]:   # the nearest that is a clean loop: whole bars at its own tempo, its hits on a straight grid
        m = c["meta"]
        try:
            y, prep = prepare(m, c["f"], c["shift"] if part != "drums" else 0, tempo, GRID_FIT_MIN); break
        except Exception:
            skipped += 1; y = None; continue
    if y is None: return {"error": "none of the nearest licensed loops is clean enough to lay on your bars"}
    out = {"part": part, "loop": {"id": m.get("id"), "name": m.get("name"), "user": m.get("user"), "licence": _licence(m.get("license")),
                                  "licence_url": m.get("license"), "page": f"https://freesound.org/s/{m.get('id')}/", "tempo": m.get("tempo"), "key": m.get("key")},
           "prep": prep, "closeness": round(c["sim"], 3), "rank": c["rank"], "of": c["of"], "skipped": skipped, "wav": base64.b64encode(wav_bytes(y)).decode()}
    if info.get("artist"):
        own = info["own"]
        out["artist"] = {"name": info["artist"], "records": info["records"],
                         "closer_than": round(float(np.mean(own < c["sim"])), 3),   # share of the artist's own records further from their average
                         "own_median": round(float(np.median(own)), 3)}
    return out


# ---------------------------------------------------------------- the whole track, by the plan the page played
def clean_plan(plan, nb_max=10000):
    """The page's plan, checked: per part either nothing (keep yours) or a loop id with the bars it plays in and a gain
    for each bar."""
    g = plan.get("grid") or {}
    grid = {"g0": float(g.get("g0") or 0), "bar": float(g.get("bar") or 0), "nb": int(g.get("nb") or 0)}
    if not (0.5 < grid["bar"] < 6 and 0 <= grid["g0"] < grid["bar"] and 0 < grid["nb"] <= nb_max): raise ValueError("grid")
    parts = {}
    for p in PARTS:
        c = (plan.get("parts") or {}).get(p)
        if not c or not c.get("loop"): continue
        gains = [min(8.0, max(0.0, float(x))) for x in (c.get("gains") or [])][:grid["nb"]]
        gains += [0.0] * (grid["nb"] - len(gains))
        runs = []
        for r in (c.get("runs") or [])[:2000]:
            r0, r1 = int(r[0]), int(r[1])
            if 0 <= r0 < r1 <= grid["nb"]: runs.append((r0, r1))
        parts[p] = {"loop": int(c["loop"]), "f": float(c.get("f") or 1), "shift": int(c.get("shift") or 0), "gains": gains, "runs": runs}
    return {"grid": grid, "parts": parts, "tempo": float(plan.get("tempo") or 0), "name": str(plan.get("name") or "track")[:100]}


def loop_by_id(part, lid):
    L = loop_index()[FAM[part]]
    for m in L["meta"]:
        if int(m.get("id") or -1) == int(lid): return m
    return None


def lay(cycle, n, g0, bar, gains, runs, ramp_s=0.005):
    """A swapped part for the whole track: the cycle from the start of each run of bars, repeated to its end, at each
    bar's gain, with five-millisecond ramps where the gain changes (as the page schedules it)."""
    out = np.zeros((n, 2), np.float32); C = len(cycle)
    for r0, r1 in runs:
        s0, s1 = int(round((g0 + r0 * bar) * SR)), min(n, int(round((g0 + r1 * bar) * SR)))
        k = s0
        while k < s1:
            m = min(C, s1 - k); out[k:k + m] += cycle[:m]; k += C
    env = np.zeros(n, np.float32); R = max(1, int(ramp_s * SR)); prev = 0.0
    for b, gv in enumerate(gains):
        s0, s1 = int(round((g0 + b * bar) * SR)), min(n, int(round((g0 + (b + 1) * bar) * SR)))
        if s0 >= n: break
        env[s0:s1] = gv
        if gv != prev:   # the change ramps in over the last five milliseconds of the bar before, so a bar's first hit has its own gain
            r0 = max(0, s0 - R); env[r0:s0] = np.linspace(prev, gv, s0 - r0, endpoint=False)
        prev = gv
    return out * env[:, None]


def render(path, plan):
    """The whole track with the plan's loops in place of those parts, as MP3 (320 kbps) with the loops' credits."""
    import soundfile as sf
    from features import stems as S
    P = clean_plan(plan); t0 = time.time()
    tempo = P["tempo"] or 240 / P["grid"]["bar"]
    w = tempfile.mkdtemp()
    try:
        wav = os.path.join(w, "track.wav"); decode_to_wav(path, wav)
        st = S.separate(wav, os.path.join(w, "sep"))
        X = {k: read_wav(st[k])[0] for k in PARTS if k in st}
        n = min(len(v) for v in X.values()); X = {k: v[:n] for k, v in X.items()}
        out = np.zeros((n, 2), np.float32); credits = []
        for p in PARTS:
            c = P["parts"].get(p)
            if not c: out += X[p]; continue
            m = loop_by_id(p, c["loop"])
            if not m: out += X[p]; continue
            y, prep = prepare(m, c["f"], c["shift"] if p != "drums" else 0, tempo)
            out += lay(y, n, P["grid"]["g0"], P["grid"]["bar"], c["gains"], c["runs"])
            credits.append(f"{m.get('name')} by {m.get('user')} (freesound.org/s/{m.get('id')}), {_licence(m.get('license'))}")
        pk = float(np.abs(out).max()) or 1.0
        if pk > 0.98: out *= 0.98 / pk
        meta = {"title": f"{P['name']} (swapped)", "comment": "Loops: " + "; ".join(credits) if credits else "No parts swapped"}
        data = mp3_bytes(out, kbps=320, meta=meta)
        return {"name": P["name"], "seconds": round(n / SR, 1), "credits": credits, "mp3": data, "took": round(time.time() - t0, 1)}
    finally:
        shutil.rmtree(w, ignore_errors=True)
