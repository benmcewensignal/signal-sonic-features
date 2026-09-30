"""Build a set's ranker (site: data/dj-ranker.json): a logistic model trained on what working DJs played next, from
published tracklists (true neighbours only, inside a ±16% tempo window, both records in the learned ear), scored on
features the page computes for any pair: closeness in the ear, tempo change, key relation, same scene, same label,
shared artist, and the difference on each measure. Record identities are not features, so it carries over to records
nobody has played. Evaluated on DJs it never saw (the real next record hidden among 200 at a mixable tempo).
  python tools/dj_ranker.py --tracklists ../signal-sonic/data/tracklists --djindex ../signalgood/data/dj-index.json \
      --djnames ../signalgood/data/dj-names.json --out ../signalgood/data/dj-ranker.json
"""
import argparse, base64, glob, json, os, numpy as np
from sklearn.linear_model import LogisticRegression
MS = ("bright", "busy", "punch", "vocal", "bass", "drums")
NAMES = ["ear", "abs_dt", "dt", "key_exact", "key_relative", "key_step", "key_other", "same_scene", "same_label", "shared_artist"] + [p + m for m in MS for p in ("absd_", "d_")]

def main():
    ap = argparse.ArgumentParser()
    for k in ("tracklists", "djindex", "djnames", "out"): ap.add_argument("--" + k, required=True)
    a = ap.parse_args(); rng = np.random.default_rng(7)
    X = json.load(open(a.djindex)); n = X["n"]; at = {t: i for i, t in enumerate(X["ids"])}; d = X["eardims"]
    E = np.frombuffer(base64.b64decode(X["ear"]), np.int8).astype(np.float32).reshape(n, d) * np.array(X["earscale"], np.float32) / 127
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9); has = np.frombuffer(base64.b64decode(X["earhas"]), np.uint8)[:n].astype(bool)
    tm = np.frombuffer(base64.b64decode(X["tempo"]), np.uint16)[:n].astype(np.float32) / 10; key = np.frombuffer(base64.b64decode(X["key"]), np.uint8)[:n]
    sc = np.frombuffer(base64.b64decode(X["scene"]), np.uint8)[:n]; M = {m: np.frombuffer(base64.b64decode(X[m]), np.uint8)[:n].astype(np.float32) for m in MS}
    N = json.load(open(a.djnames)); LB = N.get("l") or [""] * n; AR = [set(x.split("|")) - {""} if x else set() for x in N["a"]]
    def mix(x, y, w=0.16): return bool(tm[x]) and any(abs(tm[y] * f - tm[x]) / tm[x] <= w for f in (1, 2, 0.5))
    def keyrel(x, y):
        if key[x] == 255 or key[y] == 255: return 0
        u, v = int(key[x]), int(key[y])
        if u == v: return 1
        if u % 12 == v % 12: return 2
        if (u < 12) == (v < 12) and (u - v) % 12 in (1, 11): return 3
        return 4
    def feats(x, B):
        B = np.asarray(B); dt = (tm[B] - tm[x]) / max(tm[x], 1); kr = np.array([keyrel(x, y) for y in B])
        F = [E[B] @ E[x], np.abs(dt), dt] + [(kr == k).astype(float) for k in (1, 2, 3, 4)] + [(sc[B] == sc[x]).astype(float),
             np.array([float(bool(LB[x]) and LB[y] == LB[x]) for y in B]), np.array([float(bool(AR[x] & AR[y])) for y in B])]
        for m in MS: F += [np.abs(M[m][B] - M[m][x]) / 100, (M[m][B] - M[m][x]) / 100]
        return np.stack(F, 1)
    seqs, owner = [], []
    for f in sorted(glob.glob(os.path.join(a.tracklists, "*.json"))):
        if f.endswith("curves.json"): continue
        for st in json.load(open(f)).get("sets", []):
            seqs.append([at.get(r.get("bp"), -1) for r in st.get("records", [])]); owner.append(f)
    pairs = lambda Q: [(x, y) for s in Q for x, y in zip(s, s[1:]) if x >= 0 and y >= 0 and x != y and has[x] and has[y] and mix(x, y)]
    def cands(x): 
        m = np.zeros(n, bool)
        for fct in (1, 2, 0.5): m |= np.abs(tm * fct - tm[x]) / max(tm[x], 1) <= 0.16
        m &= has; m[x] = False; return np.flatnonzero(m)
    def train(P):
        Xs, ys = [], []
        for x, y in P:
            c = cands(x); c = c[c != y]
            if len(c) < 10: continue
            Xs.append(feats(x, [y] + list(rng.choice(c, 10, replace=False)))); ys += [1] + [0] * 10
        Z = np.vstack(Xs); mu, sd = Z.mean(0), Z.std(0) + 1e-9
        return LogisticRegression(max_iter=3000).fit((Z - mu) / sd, np.array(ys)), mu, sd
    # evaluation on DJs it never saw
    DJS = sorted(set(owner)); held = {DJS[i] for i in rng.permutation(len(DJS))[:len(DJS) // 5]}
    tr = [q for q, o in zip(seqs, owner) if o not in held]; te = [q for q, o in zip(seqs, owner) if o in held]
    clf, mu, sd = train(pairs(tr)); r_ear, r_mod = [], []
    for x, y in pairs(te):
        c = cands(x); c = c[c != y]
        if len(c) < 200: continue
        B = [y] + list(rng.choice(c, 200, replace=False)); F = feats(x, B)
        for r, s in ((r_ear, F[:, 0]), (r_mod, clf.decision_function((F - mu) / sd))): r.append(int((s > s[0]).sum()) + 1)
    ev = {"held_out_djs": len(held), "test_pairs": len(r_mod), "ear_top10": round(float(np.mean(np.array(r_ear) <= 10)) * 100, 1), "model_top10": round(float(np.mean(np.array(r_mod) <= 10)) * 100, 1),
          "ear_first": round(float(np.mean(np.array(r_ear) == 1)) * 100, 1), "model_first": round(float(np.mean(np.array(r_mod) == 1)) * 100, 1)}
    P = pairs(seqs); clf, mu, sd = train(P)
    json.dump({"note": __doc__.split("\n  python")[0].strip(), "features": NAMES, "mu": [round(float(v), 6) for v in mu], "sd": [round(float(v), 6) for v in sd],
               "coef": [round(float(v), 6) for v in clf.coef_[0]], "intercept": round(float(clf.intercept_[0]), 6), "trained_pairs": len(P), "test": ev}, open(a.out, "w"), indent=0)
    print(json.dumps({"trained_pairs": len(P), **ev}))

if __name__ == "__main__":
    main()
