"""The groovebox as a test rig: tracks rendered with every answer known, run through the deployed reader once each.
1 Separation against the true parts: how much of the kick the separated bass carries (the bass never plays on a kick,
  so bass-part energy at a kick is bleed), how much bass the separated drums carry, and the error of the separated parts.
2 The reader's measures against exact answers: tempo, swing, kick pattern, which steps carry hats and how late, the bass
  notes, the key.
3 Whether the part measures behind brightness, punch, busyness and bass weight respond to their own quality: one change
  at a time on a house track, each in both directions, with every other measure watched for spill."""
import io, json, wave, base64, math, urllib.request, numpy as np, modal
SITE = "https://www.earlysignal.live"; SR = 32000
def get(u, t=60): return urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "sonic-rig"}), timeout=t).read()
def wav_bytes(y):
    b = io.BytesIO()
    with wave.open(b, "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    return b.getvalue()
PC = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}
SCENES = [("house", 126, 0.50, "K...K...K...K...", [2, 6, 10, 14], [2, 6, 10, 14], "Am"), ("tech-house", 127, 0.50, "K...K...K...K...", [2, 3, 6, 7, 10, 11, 14, 15], [2, 6, 10, 14], "Fm"),
          ("techno-peak-time", 132, 0.50, "K...K...K...K...", [2, 6, 10, 14], [3, 7, 11, 15], "Gm"), ("uk-garage-speed-garage", 134, 0.58, "K...K...K...K...", [2, 3, 6, 7, 10, 11, 14, 15], [3, 6, 11, 14], "Cm"),
          ("drum-and-bass", 172, 0.50, "K....K..K....K..", [2, 6, 10, 14], [3, 7, 11], "Dm"), ("amapiano", 113, 0.50, "K...K...K...K...", [2, 3, 6, 7, 10, 11, 14, 15], [3, 6, 10, 15], "A#m")]
def kit_for(scene):
    import librosa
    G = json.loads(get(SITE + "/data/groove-scenes.json")); kit = [s for s in G["scenes"] if s["id"] == scene][0]["kits"]["match"]
    out = {}
    for r in ("kick", "snare", "hat"):
        x, _ = librosa.load(io.BytesIO(get(SITE + "/data/groove-kit/" + kit[r] + ".mp3")), sr=SR, mono=True); out[r] = x / (np.max(np.abs(x)) or 1)
    return out
def saw(f, dur, harm=8):
    t = np.arange(int(dur * SR)) / SR; return sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, harm + 1) if f * k < SR / 2)
def render(scene, bpm, swing, kpat, hats, bsteps, key, var=None, seed=1):
    rng = np.random.default_rng(seed); kit = kit_for(scene); six = 60 / bpm / 4; bars = 16; n = int((bars * 16 * six + 1.0) * SR)
    D, B, C = np.zeros(n), np.zeros(n), np.zeros(n)
    def add(buf, t, sig, g):
        i = int(t * SR); j = min(n, i + len(sig))
        if 0 <= i < n: buf[i:j] += g * sig[:j - i]
    v = var or {}; hat_steps = hats if v.get("busy") is None else ([2, 6, 10, 14] if v["busy"] < 0 else list(range(1, 16, 1)))
    hat_steps = [s for s in hat_steps if s not in (4, 12) and kpat[s] != "K"]
    root = PC[key.rstrip("m")]; minor = key.endswith("m"); prog = [0, 8, 3, 10] if minor else [0, 7, 9, 5]
    late = {s: ((2 * swing - 1) if s % 2 == 1 else 0.0) for s in range(16)}
    for b in range(bars):
        for s in range(16):
            t = 0.5 + (b * 16 + s) * six
            if kpat[s] == "K": add(D, t, kit["kick"], 1.0)
            if s in (4, 12): add(D, t, kit["snare"], 0.6)
            if s in hat_steps: add(D, t + late[s] * six, kit["hat"], 0.35 if s % 4 == 2 else 0.22)
            if s in bsteps:
                f = 55 * 2 ** (((root + (7 if s % 8 == 7 else 0)) % 12 - 9) / 12); tone = saw(f, min(0.9, 3.5 * six)); env = np.minimum(1, np.arange(len(tone)) / (0.004 * SR)) * np.exp(-np.arange(len(tone)) / SR * 5)
                add(B, t + late[s] * six * 0.5, tone * env, 0.35)
        cr = (root + prog[(b // 2) % 4]) % 12; iv = [0, 3, 7] if (minor and prog[(b // 2) % 4] == 0) else [0, 4, 7]
        pad = sum(saw(220 * 2 ** (((cr + x) % 12 - 9) / 12), 16 * six, 4) for x in iv); penv = np.minimum(1, np.arange(len(pad)) / (0.2 * SR)) * np.minimum(1, (len(pad) - np.arange(len(pad))) / (0.2 * SR))
        add(C, 0.5 + b * 16 * six, pad * penv, 0.06)
    if v.get("bright"):
        if v["bright"] > 0: D = D + 1.5 * np.concatenate([[0], np.diff(D)])
        else: D = np.convolve(D, np.ones(9) / 9, mode="same")
    if v.get("punch"):
        if v["punch"] < 0:
            tail = rng.standard_normal(int(0.25 * SR)) * np.exp(-np.arange(int(0.25 * SR)) / SR * 12) * 0.08; D = D * 0.6 + np.convolve(D, tail, mode="full")[:n]
    if v.get("bass"): B = B * (2.0 if v["bass"] > 0 else 0.5)
    mix = D + B + C; g = 0.8 / (np.max(np.abs(mix)) or 1)
    truth = {"scene": scene, "tempo": bpm, "swing": swing, "kick": kpat, "hats": sorted(hat_steps), "late": {str(s): round(late[s], 3) for s in hat_steps}, "bass_steps": sorted(bsteps), "key": key, "var": v,
             "kick_times": [0.5 + (b * 16 + s) * six for b in range(bars) for s in range(16) if kpat[s] == "K"], "bass_times": [0.5 + (b * 16 + s) * six for b in range(bars) for s in bsteps]}
    return mix * g, {"drums": D * g, "bass": B * g, "other": C * g}, truth
def win_energy(y, times, a=0.0, b=0.06):
    e = [];
    for t in times:
        i, j = int((t + a) * SR), int((t + b) * SR)
        if 0 <= i < j <= len(y): e.append(float(np.mean(y[i:j] ** 2)))
    return float(np.mean(e)) if e else 0.0
app = modal.App("sonic-rig-test")
@app.local_entrypoint()
def main():
    import librosa
    jobs = []
    for k, (sc, bpm, sw, kp, ht, bs, key) in enumerate(SCENES):
        for rep in range(2):
            kk = list(PC)[(PC[key.rstrip("m")] + 5 * rep) % 12] + ("m" if key.endswith("m") else ""); jobs.append(render(sc, bpm + rep * 2, sw, kp, ht, bs, kk, None, seed=k * 10 + rep))
    base = SCENES[0]
    for f in ("bright", "punch", "busy", "bass"):
        for d in (-1, 1): jobs.append(render(*base, var={f: d}, seed=99))
    read = modal.Function.from_name("sonic-parts", "read_parts")
    outs = list(read.map([wav_bytes(m) for m, _, _ in jobs], [True] * len(jobs), return_exceptions=True))
    rows = []
    for (mix, stems, tr), o in zip(jobs, outs):
        row = {"truth": {k: v for k, v in tr.items() if k not in ("kick_times", "bass_times")}}
        if not isinstance(o, dict): row["error"] = str(o)[:200]; rows.append(row); continue
        parts = o.get("parts") or {}; sk = o.get("skeleton") or {}; dr = parts.get("drums") or {}; ba = parts.get("bass") or {}
        row["read"] = {"key": o.get("key"), "drums": {k: dr.get(k) for k in ("beats_per_minute", "swing16", "kick_pattern", "centroid_hz", "crest", "onsets_per_s", "share_of_energy", "level", "grid_refit")},
                       "bass": {k: ba.get(k) for k in ("share_of_energy", "level", "centroid_hz")}, "skeleton": {k: sk.get(k) for k in ("grid", "tempo", "kick", "occ", "rel", "bass_occ", "bass_per_bar", "bleed_share", "bass_notes")}}
        pa = o.get("part_audio") or o.get("audio") or {}
        sep = {}
        for k in ("drums", "bass", "other"):
            if k in pa:
                try:
                    y, _ = librosa.load(io.BytesIO(base64.b64decode(pa[k])), sr=SR, mono=True); ref = stems[k][:len(y)]; y = y[:len(ref)]
                    sep[k] = round(float(10 * np.log10((np.sum(ref ** 2) + 1e-9) / (np.sum((ref - y) ** 2) + 1e-9))), 2)
                    if k == "bass": sep["bass_at_kicks_vs_notes"] = round(win_energy(y, tr["kick_times"]) / (win_energy(y, tr["bass_times"]) + 1e-12), 3); sep["true_bass_at_kicks_vs_notes"] = round(win_energy(ref, tr["kick_times"]) / (win_energy(ref, tr["bass_times"]) + 1e-12), 3)
                    if k == "drums": sep["drums_low_at_bass_notes"] = round(win_energy(librosa.effects.preemphasis(y, coef=-0.97), tr["bass_times"], 0.02, 0.2) / (win_energy(librosa.effects.preemphasis(y, coef=-0.97), tr["kick_times"]) + 1e-12), 3)
                except Exception as e: sep[k + "_error"] = type(e).__name__
        row["separation"] = sep; row["keys_returned"] = sorted(o.keys())[:20]; rows.append(row)
    json.dump(rows, open("rig-test.json", "w"), indent=1)
    print("RIG", len(rows), "renders")
    for r in rows:
        t = r["truth"]; rd = r.get("read") or {}; sk = rd.get("skeleton") or {}; d = rd.get("drums") or {}
        hats = [s for s in range(16) if (sk.get("occ") or [0] * 16)[s] >= 0.34]; late = [round((sk.get("rel") or [0] * 16)[int(s)], 2) for s in t["late"]]
        print("ROW", t["scene"], t["key"], t["tempo"], "var", t["var"] or "-", "| tempo", sk.get("tempo"), d.get("beats_per_minute"), "| key", rd.get("key"), "| swing", d.get("swing16"), "true", t["swing"],
              "| kick", sk.get("kick"), "true", t["kick"], "| hats", hats, "true", t["hats"], "| late", late, "true", list(t["late"].values()),
              "| bass occ peaks", [s for s in range(16) if (sk.get("bass_occ") or [0] * 16)[s] >= 0.5], "true", t["bass_steps"], "| bleed", sk.get("bleed_share"),
              "| sep", r.get("separation"), "| drums cent", d.get("centroid_hz"), "crest", d.get("crest"), "onsets", d.get("onsets_per_s"), "bass share", (rd.get("bass") or {}).get("share_of_energy"), "| keys", r.get("keys_returned") if r is rows[0] else "")
