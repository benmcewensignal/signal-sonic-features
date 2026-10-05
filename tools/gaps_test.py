"""Two open questions, tested on the deployed reader.
A  Can the separated 'Your track' reading be switched on? ~26 real records: a sixty-second clip through the reader against
   the record's stored skeleton (the whole preview, the same corrected method). Pre-registered pass: kick patterns match up to
   where the bar starts on at least 80% of records; hat steps (share of bars >= 0.34) overlap with a median Jaccard of 0.7 or
   more; lateness on shared steps differs by a median under 0.05 of a sixteenth.
B  Is the bleed filter discarding real bass? Rig tracks with an 808-style sub kick: (1) bass only off the kick: does the
   separated bass carry the kick? (2) bass on every kick: how many of those real notes does the filter drop?"""
import io, json, wave, base64, random, urllib.request, numpy as np, modal
SITE = "https://www.earlysignal.live"; SR = 32000
def get(u, t=60): return urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "sonic-gaps"}), timeout=t).read()
def wav_bytes(y):
    b = io.BytesIO()
    with wave.open(b, "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    return b.getvalue()
canon = lambda p: min(p[k:] + p[:k] for k in range(16)) if p else None
def nib(b, signed=False):
    hi = (b >> 4).astype(int); lo = (b & 15).astype(int); o = np.empty(len(b) * 2, int); o[0::2] = hi; o[1::2] = lo
    return np.where(o > 7, o - 16, o) if signed else o
def rig(kind, bpm=126, bars=16):
    import librosa
    G = json.loads(get(SITE + "/data/groove-scenes.json")); kit = [s for s in G["scenes"] if s["id"] == "house"][0]["kits"]["match"]
    def smp(r): x, _ = librosa.load(io.BytesIO(get(SITE + "/data/groove-kit/" + kit[r] + ".mp3")), sr=SR, mono=True); return x / (np.max(np.abs(x)) or 1)
    clap, hat = smp("snare"), smp("hat"); six = 60 / bpm / 4; n = int((bars * 16 * six + 1.5) * SR); D, Bs = np.zeros(n), np.zeros(n)
    tk = np.arange(int(0.45 * SR)) / SR; kick = np.sin(2 * np.pi * (45 + 90 * np.exp(-tk * 30)) * tk) * np.exp(-tk * 4.5)   # an 808-style sub kick, long tail
    L = int(0.22 * SR); tb = np.arange(L) / SR; note = sum(np.sin(2 * np.pi * 55 * k * tb) / k for k in range(1, 6)) * np.minimum(1, tb * 300) * np.exp(-tb * 5) * np.minimum(1, (L - np.arange(L)) / (0.03 * SR))
    steps = (2, 6, 10, 14) if kind == "off" else (0, 4, 8, 12)
    kt, bt = [], []
    def add(buf, t, s, g):
        i = int(t * SR); j = min(n, i + len(s))
        if i < n: buf[i:j] += g * s[:j - i]
    for b in range(bars):
        for s in range(16):
            t = 0.5 + (b * 16 + s) * six
            if s % 4 == 0: add(D, t, kick, 1.0); kt.append(t)
            if s in (4, 12): add(D, t, clap, 0.5)
            if s in (2, 6, 10, 14): add(D, t, hat, 0.3)
            if s in steps: add(Bs, t, note, 0.45); bt.append(t)
    mix = D + Bs; g = 0.8 / np.max(np.abs(mix)); return mix * g, D * g, Bs * g, kt, bt, steps
def energy_at(y, times, a=0.0, b=0.06):
    e = [float(np.mean(y[int((t + a) * SR):int((t + b) * SR)] ** 2)) for t in times if int((t + b) * SR) <= len(y)]
    return float(np.mean(e)) if e else 0.0
app = modal.App("sonic-gaps-test")
@app.local_entrypoint()
def main():
    import librosa
    REC = json.loads(get(SITE + "/data/groove-records.json")); PV = json.loads(get(SITE + "/data/previews.json"))["previews"]; n = REC["n"]
    ids = np.frombuffer(base64.b64decode(REC["id"]), np.uint32); sc = np.frombuffer(base64.b64decode(REC["sc"]), np.uint8); kp = np.frombuffer(base64.b64decode(REC["kp"]), np.uint16)
    SK = np.frombuffer(get(SITE + "/data/groove-skeleton.bin", 120), np.uint8).reshape(n, 33)
    want = ["house", "tech-house", "deep-house", "techno-peak-time", "techno-raw-deep-hypnotic", "melodic-house-techno", "uk-garage-speed-garage", "drum-and-bass", "amapiano", "afro-house", "progressive-house", "breaks-breakbeat-uk-bass", "trance-main-floor"]
    rng = random.Random(11); picks = []
    for w in want:
        si = REC["scenes"].index(w); cand = [i for i in range(n) if sc[i] == si and kp[i] != 65535 and (SK[i, 0] & 1) and f"bp:{int(ids[i])}" in PV]; rng.shuffle(cand); picks += cand[:2]
    clips, meta = [], []
    for i in picks:
        try:
            y, _ = librosa.load(io.BytesIO(get(PV[f"bp:{int(ids[i])}"])), sr=SR, mono=True)
            if len(y) < SR * 40: continue
            mid = len(y) // 2; clips.append(y[mid - 30 * SR: mid + 30 * SR]); meta.append(i)
        except Exception: pass
    rigs = [rig("off"), rig("on")]
    read = modal.Function.from_name("sonic-parts", "read_parts")
    outs = list(read.map([wav_bytes(c) for c in clips] + [wav_bytes(r[0]) for r in rigs], [False] * len(clips) + [True] * len(rigs), return_exceptions=True))
    A = outs[:len(clips)]; R = outs[len(clips):]; rows = []
    for i, o in zip(meta, A):
        sk = o.get("skeleton") if isinstance(o, dict) else None
        if not sk or not sk.get("occ"): rows.append({"i": int(i), "error": "no skeleton"}); continue
        so = nib(SK[i, 1:9]) / 15; sr_ = nib(SK[i, 9:17], True) / 20; P = REC["patterns"][kp[i]]
        ha = {s for s in range(16) if sk["occ"][s] >= 0.34}; hb = {s for s in range(16) if so[s] >= 0.34}; both = sorted(ha & hb)
        rows.append({"i": int(i), "scene": REC["scenes"][sc[i]], "kick_same": canon(sk.get("kick")) == canon(P), "jaccard": round(len(ha & hb) / max(1, len(ha | hb)), 2),
                     "late": round(float(np.median([abs(sk["rel"][s] - sr_[s]) for s in both])), 3) if both else None, "kick_clip": sk.get("kick"), "kick_stored": P})
    ok = [r for r in rows if "jaccard" in r]
    A_sum = {"records": len(ok), "kick_same": sum(r["kick_same"] for r in ok), "median_jaccard": round(float(np.median([r["jaccard"] for r in ok])), 2) if ok else None,
             "median_late": round(float(np.median([r["late"] for r in ok if r["late"] is not None])), 3) if ok else None}
    A_sum["pass"] = bool(ok) and A_sum["kick_same"] >= 0.8 * len(ok) and A_sum["median_jaccard"] >= 0.7 and A_sum["median_late"] < 0.05
    B_rows = []
    for (mix, D, Bs, kt, bt, steps), o in zip(rigs, R):
        if not isinstance(o, dict): B_rows.append({"error": str(o)[:120]}); continue
        sk = o.get("skeleton") or {}; pa = o.get("part_audio") or {}; row = {"bass_steps": steps, "bleed_share": sk.get("bleed_share"), "bass_occ": [round(x, 2) for x in (sk.get("bass_occ") or [])]}
        if "bass" in pa:
            yb, _ = librosa.load(io.BytesIO(base64.b64decode(pa["bass"])), sr=SR, mono=True); nb = min(len(yb), len(Bs))
            kick_only = [t for t in kt if all(abs(t - u) > 0.05 for u in bt)]
            row["sep_bass_at_kicks"] = round(energy_at(yb[:nb], kick_only) / (energy_at(yb[:nb], bt) + 1e-12), 3); row["true_bass_at_kicks"] = round(energy_at(Bs[:nb], kick_only) / (energy_at(Bs[:nb], bt) + 1e-12), 3)
        B_rows.append(row)
    out = {"A": {"summary": A_sum, "rows": rows}, "B": B_rows}
    json.dump(out, open("gaps-test.json", "w"), indent=1)
    print("A", json.dumps(A_sum)); [print("A_ROW", json.dumps(r)) for r in rows]; [print("B", json.dumps(r)) for r in B_rows]
