"""The worker's loop library: every measured loop, standardised with its family's corpus part statistics, plus what the
page shows (name, creator, licence, preview, tempo, key) and four plain measures for saying why a loop matched."""
import json, pickle, numpy as np
import os
L = json.load(open(os.environ.get("LOOPS", "data/loops-measured.json"))); S = json.load(open("data/part-stats.json")); keep = S["keep"]
# an older run kept only the numbers: fill names, creators, licences and previews from the published drum matches,
# and leave out any loop still missing them, rather than show a nameless loop with no player
try:
    known = {x["id"]: x for L_ in json.load(open("data/sample-matches.json"))["scenes"].values() for x in L_}
except Exception:
    known = {}
for l in L:
    k = known.get(l.get("id"))
    if k:
        for f in ("name", "user", "license", "preview"):
            if not l.get(f): l[f] = k.get(f)
L = [l for l in L if l.get("name") and l.get("preview") and l.get("license")]
SC = ("level", "crest", "dynamic_span", "centroid_hz", "rolloff_hz", "flatness", "onsets_per_s", "share_of_energy")
out = {}
for fam, st in S["families"].items():
    rows = [l for l in L if (l.get("cat") or "drums") == fam and isinstance(l.get("v"), list) and len(l["v"]) == 53]
    if not rows: continue
    mu, sd = np.array(st["mu"]), np.array(st["sd"])
    V = np.array([np.array(r["v"])[keep] for r in rows]); Z = (V - mu) / sd; Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    meta = [{k: r.get(k) for k in ("id", "name", "user", "license", "preview", "tempo", "key", "page", "src")} | {"plain": {n: round(float(r["v"][45 + SC.index(n)]), 4) for n in ("crest", "centroid_hz", "flatness", "onsets_per_s")}} for r in rows]
    out[fam] = {"keep": keep, "mu": mu, "sd": sd, "Z": Z.astype(np.float16), "meta": meta}
try:   # the smudged library (worker/gap_tests.py::cal), so separated parts search loops that went through the same separation
    S_ = np.load("data/smudged-library.npz")
    for f_, L_ in out.items():
        Zs_ = S_[f_ + "_Zs"].astype(np.float32) if f_ + "_Zs" in S_ else None
        if Zs_ is not None and Zs_.shape == L_["Z"].shape:
            h_ = np.linalg.norm(Zs_, axis=1) > 0.5; L_["Zs"] = np.where(h_[:, None], Zs_, L_["Z"].astype(np.float32)).astype(np.float16)
except FileNotFoundError:
    pass
pickle.dump(out, open("worker/loop_index.pkl", "wb"))
print("loop library:", {f: len(v["meta"]) for f, v in out.items()})
