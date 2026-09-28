"""The worker's loop library: every measured loop, standardised with its family's corpus part statistics, plus what the
page shows (name, creator, licence, preview, tempo, key) and four plain measures for saying why a loop matched."""
import json, pickle, numpy as np
L = json.load(open("data/loops-measured.json")); S = json.load(open("data/part-stats.json")); keep = S["keep"]
SC = ("level", "crest", "dynamic_span", "centroid_hz", "rolloff_hz", "flatness", "onsets_per_s", "share_of_energy")
out = {}
for fam, st in S["families"].items():
    rows = [l for l in L if (l.get("cat") or "drums") == fam and isinstance(l.get("v"), list) and len(l["v"]) == 53]
    if not rows: continue
    mu, sd = np.array(st["mu"]), np.array(st["sd"])
    V = np.array([np.array(r["v"])[keep] for r in rows]); Z = (V - mu) / sd; Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    meta = [{k: r.get(k) for k in ("id", "name", "user", "license", "preview", "tempo", "key")} | {"plain": {n: round(float(r["v"][45 + SC.index(n)]), 4) for n in ("crest", "centroid_hz", "flatness", "onsets_per_s")}} for r in rows]
    out[fam] = {"keep": keep, "mu": mu, "sd": sd, "Z": Z.astype(np.float16), "meta": meta}
pickle.dump(out, open("worker/loop_index.pkl", "wb"))
print("loop library:", {f: len(v["meta"]) for f, v in out.items()})
