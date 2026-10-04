"""Do the stored record skeletons measure real records correctly? Each of ~25 real records (spread across scenes) is measured
two independent ways on the same sixty seconds: (A) the record method, through the deployed reader (htdemucs separation,
the beat tracker, kicks told apart by their 30-120 Hz energy: how the stored skeletons were made), and (B) the full-mix
fold (bands, a kick-led tempo and phase fit, the high band folded onto one bar), which read a known-timing loop almost
exactly. Two synthetic loops with known answers run through both as references: the first loop (bright noise hats over a
soft sine kick) and a realistic one built from the CC0 kit with the kick loudest. Pre-registered: the methods agree if, on at
least 70% of records, the hats land on mostly the same steps (Jaccard >= 0.6) and lateness on shared steps differs by
under 0.05 of a sixteenth (median)."""
import io, json, math, wave, random, urllib.request, numpy as np, modal
SITE = "https://www.earlysignal.live"
def get(u, t=60): return urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "sonic-agree"}), timeout=t).read()
def wav_bytes(y, sr):
    b = io.BytesIO()
    with wave.open(b, "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    return b.getvalue()
# ---- the full-mix fold, ported from the groovebox
def band_flux(y, sr, N, H, lo, hi):
    win = np.hanning(N + 1)[:-1]; nf = (len(y) - N) // H; hz = sr / N; a = max(1, round(lo / hz)); b = min(N // 2 - 1, round(hi / hz))
    fr = np.lib.stride_tricks.sliding_window_view(y, N)[::H][:nf] * win; e = (np.abs(np.fft.rfft(fr, axis=1)[:, a:b + 1]) ** 2).sum(1)
    le = np.log(e + 1e-3 * np.percentile(e, 95) + 1e-12); return np.maximum(0, np.diff(le, prepend=le[0])), N / 2 / sr, H / sr
def onsets(fl, t0, dt):
    med = np.median(fl); mad = np.median(np.abs(fl - med)) or 1e-6; th = med + 2.5 * mad; out = []; last = -99; gap = round(0.035 / dt)
    for f in range(2, len(fl) - 2):
        if fl[f] > th and fl[f] >= fl[f - 1] and fl[f] >= fl[f + 1] and fl[f] >= fl[f - 2] and fl[f] >= fl[f + 2] and f - last >= gap:
            a0, a1, a2 = fl[f - 1], fl[f], fl[f + 1]; dn = a0 - 2 * a1 + a2; dl = 0.5 * (a0 - a2) / dn if dn else 0; out.append(t0 + (f + max(-.5, min(.5, dl))) * dt); last = f
    return np.array(out)
def fold_read(y, sr):
    lo, t0l, dtl = band_flux(y, sr, 1024, 128, 40, 150); hiF, t0h, dth = band_flux(y, sr, 256, 128, 5000, 11000); md, _, _ = band_flux(y, sr, 256, 128, 1000, 4000)
    n = min(len(lo), len(hiF)); env = lo[:n] + hiF[:n] + 0.5 * md[:n]; fps = 1 / dtl; best = -1; bpm0 = 124
    for bpm in np.arange(80, 180.5, 0.5):
        lag = 60 / bpm * fps; f = np.arange(0, int(n - lag - 1), 2); j = f + lag; j0 = np.floor(j).astype(int); fr = j - j0
        s = np.mean(env[f] * (env[j0] * (1 - fr) + env[j0 + 1] * fr)) * math.exp(-math.log2(bpm / 125) ** 2 / (2 * 0.45 ** 2))
        if s > best: best, bpm0 = s, bpm
    lon = onsets(lo, t0l, dtl); T = 60 / bpm0; ph0 = 0; bs = -1e9
    for bp in np.arange(bpm0 - 1.5, bpm0 + 1.5001, 0.02):
        TT = 60 / bp; phs = np.arange(0, TT, 0.004); r = (lon[None, :] - phs[:, None]) / TT; sc = np.cos(2 * np.pi * (r - np.round(r))).sum(1); k = int(np.argmax(sc))
        if sc[k] > bs: bs, T, ph0 = sc[k], TT, phs[k]
    q = T / 4; SUB = 24
    def prof(fl, t0, dt):
        P = np.zeros(16 * SUB); C = np.zeros(16 * SUB); t = t0 + np.arange(len(fl)) * dt; m = t >= ph0; k = (np.floor((((t[m] - ph0) / q) % 16) * SUB).astype(int)) % (16 * SUB)
        np.add.at(P, k, fl[m]); np.add.at(C, k, 1); return np.where(C > 0, P / np.maximum(C, 1), 0)
    def steps16(P):
        base = np.median(P); out = []
        for s in range(16):
            idx = (s * SUB + np.arange(-SUB // 2, SUB // 2)) % (16 * SUB); seg = P[idx]; k = int(np.argmax(seg)); km, kp = seg[max(0, k - 1)], seg[min(len(seg) - 1, k + 1)]
            dn = km - 2 * seg[k] + kp; dl = 0.5 * (km - kp) / dn if dn else 0; out.append([max(0, seg[k] - base), (k - SUB // 2 + max(-.5, min(.5, dl))) / SUB])
        mx = max(x[0] for x in out) or 1
        for x in out: x[0] /= mx
        return out
    hf, t0f, dtf = band_flux(y, sr, 256, 32, 5000, 11000); mf, _, _ = band_flux(y, sr, 256, 32, 1000, 4000)
    H16 = steps16(prof(hf, t0f, dtf)); M16 = steps16(prof(mf, t0f, dtf)); L16 = steps16(prof(lo, t0l, dtl))
    rot = max((0, 4, 8, 12), key=lambda r: M16[(4 + r) % 16][0] + M16[(12 + r) % 16][0]); idx = [(s + rot) % 16 for s in range(16)]
    h = [H16[k][0] for k in idx]; o = [H16[k][1] for k in idx]; ref = sorted(o[s] for s in (0, 4, 8, 12) if h[s] >= 0.2); base = ref[len(ref) // 2] if ref else 0
    return {"tempo": round(60 / T, 1), "kick": "".join("K" if L16[k][0] >= 0.5 else "." for k in idx), "strength": [round(x, 3) for x in h], "rel": [round(o[s] - base, 3) if h[s] >= 0.2 else 0.0 for s in range(16)]}
# ---- the two reference loops
def loop(kind, sr=32000, bpm=124.0, dur=40):
    beat = 60 / bpm; six = beat / 4; n = int(sr * dur); y = np.zeros(n); rng = np.random.default_rng(3)
    def add(t, sig, g=1.0):
        i = int(t * sr); j = min(n, i + len(sig))
        if i < n: y[i:j] += g * sig[:j - i]
    if kind == "first":
        tt = np.arange(int(0.25 * sr)) / sr; kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-tt * 40)) * tt) * np.exp(-tt * 9) * 0.9
        nh = int(0.05 * sr); hat = np.diff(np.concatenate([[0], rng.standard_normal(nh) * np.exp(-np.arange(nh) / sr * 90) * 0.35]))
        ns = int(0.18 * sr); sn = (rng.standard_normal(ns) * 0.5 + np.sin(2 * np.pi * 200 * np.arange(ns) / sr) * 0.5) * np.exp(-np.arange(ns) / sr * 18) * 0.6; g = (1, 1, 1)
    else:
        import librosa
        G = json.loads(get(SITE + "/data/groove-scenes.json")); kit = [s for s in G["scenes"] if s["id"] == "house"][0]["kits"]["match"]
        def smp(r): x, _ = librosa.load(io.BytesIO(get(SITE + "/data/groove-kit/" + kit[r] + ".mp3")), sr=sr, mono=True); return x / (np.max(np.abs(x)) or 1)
        kick, sn, hat = smp("kick"), smp("snare"), smp("hat"); g = (1.0, 0.55, 0.3)
    tb = np.arange(int(0.2 * sr)) / sr; bass = sum(np.sin(2 * np.pi * 55 * k * tb) / k for k in range(1, 6)) * np.minimum(1, tb * 200) * np.exp(-tb * 6) * 0.35
    for b in range(int((dur - 1) / (4 * beat))):
        for s in range(16):
            t = 0.5 + (b * 16 + s) * six
            if s % 4 == 0: add(t, kick, g[0])
            if s in (4, 12): add(t, sn, g[1])
            if s in (2, 6, 10, 14): add(t + 0.1 * six, hat, g[2])
            if s % 4 == 3: add(t, bass, 1.0)
    return y / np.max(np.abs(y)) * 0.8
app = modal.App("sonic-agree-test")
@app.local_entrypoint()
def main():
    import librosa
    REC = json.loads(get(SITE + "/data/groove-records.json")); PV = json.loads(get(SITE + "/data/previews.json"))["previews"]
    import base64
    ids = np.frombuffer(base64.b64decode(REC["id"]), np.uint32); sc = np.frombuffer(base64.b64decode(REC["sc"]), np.uint8); kp = np.frombuffer(base64.b64decode(REC["kp"]), np.uint16)
    want = ["house", "tech-house", "deep-house", "techno-peak-time", "techno-raw-deep-hypnotic", "melodic-house-techno", "uk-garage-speed-garage", "drum-and-bass", "amapiano", "afro-house", "progressive-house", "breaks-breakbeat-uk-bass", "trance-main-floor"]
    rng = random.Random(7); picks = []
    for w in want:
        si = REC["scenes"].index(w); cand = [i for i in range(REC["n"]) if sc[i] == si and kp[i] != 65535 and f"bp:{int(ids[i])}" in PV]; rng.shuffle(cand); picks += [(w, i) for i in cand[:2]]
    clips, meta = [], []
    for w, i in picks:
        try:
            y, sr = librosa.load(io.BytesIO(get(PV[f"bp:{int(ids[i])}"])), sr=32000, mono=True)
            if len(y) < sr * 30: continue
            mid = len(y) // 2; seg = y[max(0, mid - 30 * sr): mid + 30 * sr]; clips.append(seg); meta.append({"scene": w, "id": f"bp:{int(ids[i])}", "name": REC["names"].split("\u0002")[i].split("\u0001")[0]})
        except Exception as e:
            print("skip", w, i, type(e).__name__)
    for kind in ("first", "kit"):
        clips.append(loop(kind)); meta.append({"scene": "reference", "id": "loop-" + kind, "name": "known-timing loop (" + kind + ")", "truth": {"tempo": 124, "kick": "K...K...K...K...", "hats": [2, 6, 10, 14], "late": 0.1}})
    read = modal.Function.from_name("sonic-parts", "read_parts")
    halves = []
    for c, m in zip(clips, meta):
        if m["scene"] != "reference": h = len(c) // 2; halves += [c[:h], c[h:]]
    allA = list(read.map([wav_bytes(c, 32000) for c in clips + halves], return_exceptions=True)); A = allA[:len(clips)]; H = allA[len(clips):]
    rows = []
    for c, m, a in zip(clips, meta, A):
        sk = a.get("skeleton") if isinstance(a, dict) else None
        y22 = librosa.resample(c, orig_sr=32000, target_sr=22050); B = fold_read(y22, 22050)
        row = {**m, "A": sk, "B": B}
        if sk and sk.get("occ"):
            ha = {s for s in range(16) if sk["occ"][s] >= 0.34}; hb = {s for s in range(16) if B["strength"][s] >= 0.3}
            J = len(ha & hb) / max(1, len(ha | hb)); both = sorted(ha & hb); dl = float(np.median([abs(sk["rel"][s] - B["rel"][s]) for s in both])) if both else None
            ta, tb_ = sk.get("tempo") or 0, B["tempo"]; tr = min(abs(ta - tb_), abs(ta * 2 - tb_), abs(ta - tb_ * 2))
            row.update(jaccard=round(J, 2), late_diff=round(dl, 3) if dl is not None else None, tempo_diff=round(tr, 1), kick_same=(sk.get("kick") == B["kick"]), agree=bool(J >= 0.6 and dl is not None and dl < 0.05))
        rows.append(row)
    # consistency: the record method on the first and the second half of the same minute
    hi = 0
    for r in rows:
        if r["scene"] == "reference": continue
        a1 = H[hi] if hi < len(H) else None; a2 = H[hi + 1] if hi + 1 < len(H) else None; hi += 2
        s1 = a1.get("skeleton") if isinstance(a1, dict) else None; s2 = a2.get("skeleton") if isinstance(a2, dict) else None
        if s1 and s2 and s1.get("occ") and s2.get("occ"):
            h1 = {s for s in range(16) if s1["occ"][s] >= 0.34}; h2 = {s for s in range(16) if s2["occ"][s] >= 0.34}; both = sorted(h1 & h2)
            r["halves_jaccard"] = round(len(h1 & h2) / max(1, len(h1 | h2)), 2); r["halves_late_diff"] = round(float(np.median([abs(s1["rel"][s] - s2["rel"][s]) for s in both])), 3) if both else None
            r["halves_kick_same"] = s1.get("kick") == s2.get("kick")
    real = [r for r in rows if r["scene"] != "reference" and "agree" in r]; n_ag = sum(r["agree"] for r in real)
    summary = {"records": len(real), "agree": n_ag, "share": round(n_ag / max(1, len(real)), 2), "verdict": "agree" if real and n_ag / len(real) >= 0.7 else "disagree",
               "median_jaccard": round(float(np.median([r["jaccard"] for r in real])), 2) if real else None, "median_late_diff": round(float(np.median([r["late_diff"] for r in real if r["late_diff"] is not None])), 3) if real else None,
               "median_tempo_diff": round(float(np.median([r["tempo_diff"] for r in real])), 1) if real else None, "kick_same": sum(r["kick_same"] for r in real),
               "halves_median_jaccard": round(float(np.median([r["halves_jaccard"] for r in real if "halves_jaccard" in r])), 2) if any("halves_jaccard" in r for r in real) else None,
               "halves_median_late_diff": round(float(np.median([r["halves_late_diff"] for r in real if r.get("halves_late_diff") is not None])), 3) if any(r.get("halves_late_diff") is not None for r in real) else None,
               "halves_kick_same": sum(1 for r in real if r.get("halves_kick_same"))}
    json.dump({"summary": summary, "rows": rows}, open("agree-test.json", "w"), indent=1)
    print("SUMMARY", json.dumps(summary))
    for r in rows:
        if r["scene"] == "reference":
            a = r["A"] or {}; print("REF", r["id"], "| A tempo", a.get("tempo"), "kick", a.get("kick"), "hats", [s for s in range(16) if (a.get("occ") or [0] * 16)[s] >= 0.34], "late", [round((a.get("rel") or [0]*16)[s], 2) for s in (2, 6, 10, 14)], "| B tempo", r["B"]["tempo"], "kick", r["B"]["kick"], "hats", [s for s in range(16) if r["B"]["strength"][s] >= 0.3])
        else:
            print("REC", r["scene"], "|", r.get("name", "")[:30], "| J", r.get("jaccard"), "late diff", r.get("late_diff"), "tempo diff", r.get("tempo_diff"), "kick same", r.get("kick_same"), "agree", r.get("agree"))
