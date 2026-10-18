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
# the smudged library (worker/gap_tests.py::cal) and the smudge-robust measure (worker/gap_tests.py::robust) were made on
# the library of 13 October 2026; their rows are matched to loops by id (data/loop-index-ids.json). A loop added since has
# no smudged reading, so it is searched clean: Zs is its clean Z, and Zp is its clean Z through the family's W.
try:
    IDS = json.load(open("data/loop-index-ids.json"))["ids"]
except FileNotFoundError:
    IDS = {}
def aligned(fam, rows_, M, fill):
    """Rows of M (made for the loops in IDS[fam], in order) laid against this family's loops; fill(i) for a loop not there."""
    at = {i: k for k, i in enumerate(IDS.get(fam, []))}
    outm = np.zeros((len(rows_), M.shape[1]), np.float32); n_old = 0
    for i, r in enumerate(rows_):
        k = at.get(r["id"])
        if k is not None and k < len(M): outm[i] = M[k]; n_old += 1
        else: outm[i] = fill(i)
    return outm, n_old
try:
    S_ = np.load("data/smudged-library.npz")
    for f_, L_ in out.items():
        if f_ + "_Zs" not in S_: continue
        Zc = L_["Z"].astype(np.float32); rows_ = [r for r in L if (r.get("cat") or "drums") == f_ and isinstance(r.get("v"), list) and len(r["v"]) == 53]
        Zs_, n_old = aligned(f_, rows_, S_[f_ + "_Zs"].astype(np.float32), lambda i: Zc[i])
        h_ = np.linalg.norm(Zs_, axis=1) > 0.5; L_["Zs"] = np.where(h_[:, None], Zs_, Zc).astype(np.float16)
        print(f"  {f_}: smudged readings for {n_old} of {len(rows_)} loops")
except FileNotFoundError:
    pass
try:
    R_ = np.load("data/robust-measure.npz")
    for f_, L_ in out.items():
        if f_ + "_W" not in R_: continue
        W = R_[f_ + "_W"].astype(np.float32); Zc = L_["Z"].astype(np.float32); rows_ = [r for r in L if (r.get("cat") or "drums") == f_ and isinstance(r.get("v"), list) and len(r["v"]) == 53]
        def clean_p(i, W=W, Zc=Zc): p = Zc[i] @ W; return p / (np.linalg.norm(p) + 1e-9)
        Zp_, n_old = aligned(f_, rows_, R_[f_ + "_Zp"].astype(np.float32), clean_p)
        L_["W"], L_["Zp"] = W, Zp_.astype(np.float16)
        print(f"  {f_}: robust space from the learned rows for {n_old} loops, from the clean measure for {len(rows_) - n_old}")
except FileNotFoundError:
    pass
pickle.dump(out, open("worker/loop_index.pkl", "wb"))
print("loop library:", {f: len(v["meta"]) for f, v in out.items()})
