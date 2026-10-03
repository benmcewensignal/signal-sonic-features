"""Does MERT help find what DJs actually played next? The played-next test from tools/dj_ranker.py, repeated on records
that have MERT fingerprints (the mert-records release): the real next record hidden among 200 at a tempo within 16%,
ranked on DJs held out of training, averaged over five splits. Contenders on the same pairs and the same draws: the
learned ear (today's sound measure), MERT alone (mean over parts of the centred cosine), ear plus MERT, Build a set's
ranker as it stands, and the ranker with MERT closeness added as one more feature.
  python tools/dj_mert_check.py --tracklists ../signal-sonic/data/tracklists --djindex dj-index.json --djnames dj-names.json \
      --mert mert-dir [--charts charts.jsonl] --out result.json"""
import argparse, base64, glob, gzip, json, os, re, unicodedata, collections, numpy as np
from sklearn.linear_model import LogisticRegression
MS = ("bright", "busy", "punch", "vocal", "bass", "drums")

def main():
    ap = argparse.ArgumentParser()
    for k in ("tracklists", "djindex", "djnames", "mert", "out"): ap.add_argument("--" + k, required=True)
    ap.add_argument("--charts", default=None); a = ap.parse_args(); rng = np.random.default_rng(7)
    X = json.load(open(a.djindex)); n = X["n"]; at = {t: i for i, t in enumerate(X["ids"])}; d = X["eardims"]
    E = np.frombuffer(base64.b64decode(X["ear"]), np.int8).astype(np.float32).reshape(n, d) * np.array(X["earscale"], np.float32) / 127
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9); has = np.frombuffer(base64.b64decode(X["earhas"]), np.uint8)[:n].astype(bool)
    tm = np.frombuffer(base64.b64decode(X["tempo"]), np.uint16)[:n].astype(np.float32) / 10; key = np.frombuffer(base64.b64decode(X["key"]), np.uint8)[:n]
    sc = np.frombuffer(base64.b64decode(X["scene"]), np.uint8)[:n]; M = {m: np.frombuffer(base64.b64decode(X[m]), np.uint8)[:n].astype(np.float32) for m in MS}
    N = json.load(open(a.djnames)); LB = N.get("l") or [""] * n; AR = [set(x.split("|")) - {""} if x else set() for x in N["a"]]
    # MERT: four parts per record, each centred on the mean over records and normalised; closeness is the mean over parts
    parts = ("drums", "bass", "other", "vocals"); F = {p: np.zeros((n, 768), np.float32) for p in parts}; hm = np.zeros(n, bool)
    for f in glob.glob(os.path.join(a.mert, "*.jsonl.gz")):
        for line in gzip.open(f, "rt"):
            r = json.loads(line); i = at.get(r["track_id"])
            if i is None or not all(r.get(p) for p in parts): continue
            for p in parts: F[p][i] = np.frombuffer(base64.b64decode(r[p]), np.float16).astype(np.float32)
            hm[i] = True
    if os.environ.get("FAKE_MERT"):   # a check of the check: random fingerprints for every record should leave MERT at chance
        g = np.random.default_rng(1)
        for p in parts: F[p] = g.normal(size=(n, 768)).astype(np.float32)
        hm[:] = True
    for p in parts:
        F[p][hm] -= F[p][hm].mean(0); F[p][hm] /= np.maximum(np.linalg.norm(F[p][hm], axis=1, keepdims=True), 1e-9)
    def mert_sim(x, B): return np.mean([F[p][B] @ F[p][x] for p in parts], axis=0)
    ok = has & hm
    nmz = lambda z: " ".join(re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKD", str(z or "")).encode("ascii", "ignore").decode().lower()).split())
    CH = []
    if a.charts and os.path.exists(a.charts):
        for line in open(a.charts):
            try: r = json.loads(line)
            except Exception: continue
            ii = [at[t] for t in (r.get("tracks") or []) if t in at]
            if ii: CH.append((str(r.get("dj_id") or r.get("dj")), nmz(r.get("dj")), ii))
    def chart_table(skip=frozenset()):
        ids_ = {}; per = collections.defaultdict(set)
        for k, nk, ii in CH:
            if nk in skip: continue
            j = ids_.setdefault(k, len(ids_))
            for i in ii: per[i].add(j)
        return per
    CP = [chart_table()]
    def mix(x, y, w=0.16): return bool(tm[x]) and any(abs(tm[y] * f - tm[x]) / tm[x] <= w for f in (1, 2, 0.5))
    def keyrel(x, y):
        if key[x] == 255 or key[y] == 255: return 0
        u, v = int(key[x]), int(key[y])
        if u == v: return 1
        if u % 12 == v % 12: return 2
        if (u < 12) == (v < 12) and (u - v) % 12 in (1, 11): return 3
        return 4
    def feats(x, B, with_mert):
        B = np.asarray(B); dt = (tm[B] - tm[x]) / max(tm[x], 1); kr = np.array([keyrel(x, y) for y in B])
        Fz = [E[B] @ E[x], np.abs(dt), dt] + [(kr == k).astype(float) for k in (1, 2, 3, 4)] + [(sc[B] == sc[x]).astype(float),
              np.array([float(bool(LB[x]) and LB[y] == LB[x]) for y in B]), np.array([float(bool(AR[x] & AR[y])) for y in B])]
        for m in MS: Fz += [np.abs(M[m][B] - M[m][x]) / 100, (M[m][B] - M[m][x]) / 100]
        P_ = CP[0]; A_ = P_.get(x, set())
        Fz += [np.array([np.log1p(len(A_ & P_.get(y, set()))) for y in B]), np.array([float(bool(A_) and bool(P_.get(y))) for y in B])]
        if with_mert: Fz.append(mert_sim(x, B))
        return np.stack(Fz, 1)
    seqs, owner = [], []
    for f in sorted(glob.glob(os.path.join(a.tracklists, "*.json"))):
        if f.endswith("curves.json"): continue
        for st in json.load(open(f)).get("sets", []):
            seqs.append([at.get(r.get("bp"), -1) for r in st.get("records", [])]); owner.append(f)
    pairs = lambda Q: [(x, y) for s in Q for x, y in zip(s, s[1:]) if x >= 0 and y >= 0 and x != y and ok[x] and ok[y] and mix(x, y)]
    def cands(x):
        m = np.zeros(n, bool)
        for fct in (1, 2, 0.5): m |= np.abs(tm * fct - tm[x]) / max(tm[x], 1) <= 0.16
        m &= ok; m[x] = False; return np.flatnonzero(m)
    def train(P, with_mert):
        Xs, ys = [], []
        for x, y in P:
            c = cands(x); c = c[c != y]
            if len(c) < 10: continue
            Xs.append(feats(x, [y] + list(rng.choice(c, 10, replace=False)), with_mert)); ys += [1] + [0] * 10
        Z = np.vstack(Xs); mu, sd = Z.mean(0), Z.std(0) + 1e-9
        return LogisticRegression(max_iter=3000).fit((Z - mu) / sd, np.array(ys)), mu, sd
    if len(pairs(seqs)) < 200:
        out = {"records_with_ear_and_mert": int(ok.sum()), "pairs_available": len(pairs(seqs)), "note": "too few played-next pairs with both records fingerprinted yet"}
        json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out)); return
    DJS = sorted(set(owner)); R = collections.defaultdict(list); FST = collections.defaultdict(list); NT = 0
    for split in range(5):
        rr = np.random.default_rng(100 + split); held = {DJS[i] for i in rr.permutation(len(DJS))[:len(DJS) // 5]}
        tr = [q for q, o in zip(seqs, owner) if o not in held]; te = [q for q, o in zip(seqs, owner) if o in held]
        CP[0] = chart_table(frozenset(nmz(json.load(open(o)).get("dj")) for o in held))
        m0, mu0, sd0 = train(pairs(tr), False); m1, mu1, sd1 = train(pairs(tr), True); rk = collections.defaultdict(list)
        for x, y in pairs(te):
            c = cands(x); c = c[c != y]
            if len(c) < 200: continue
            B = [y] + list(rng.choice(c, 200, replace=False)); F1 = feats(x, B, True); ear = F1[:, 0]; mt = F1[:, -1]
            z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
            scores = {"ear": ear, "mert": mt, "ear_plus_mert": z(ear) + z(mt), "ranker": m0.decision_function((F1[:, :-1] - mu0) / sd0), "ranker_plus_mert": m1.decision_function((F1 - mu1) / sd1)}
            for k, s in scores.items(): rk[k].append(int((s > s[0]).sum()) + 1)
        for k, v in rk.items(): R[k].append(float(np.mean(np.array(v) <= 10))); FST[k].append(float(np.mean(np.array(v) == 1)))
        NT += len(rk["ear"])
    out = {"records_with_ear_and_mert": int(ok.sum()), "test_pairs": NT, "top10": {k: round(float(np.mean(v)) * 100, 1) for k, v in R.items()},
           "first": {k: round(float(np.mean(v)) * 100, 1) for k, v in FST.items()}, "top10_by_split": {k: [round(x * 100, 1) for x in v] for k, v in R.items()}}
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out))

if __name__ == "__main__":
    main()
