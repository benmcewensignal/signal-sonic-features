"""Two tests of Catalogue's demand signals, on data Sonic already holds.
persistence: does a scene whose share of Beatport releases moved over one half-year keep moving the same way over
             the next? (four half-years of Beatport's own monthly counts; correlation across scenes, and how often
             the direction repeats, against shuffled scenes)
lead-lag:    do DJ plays move first? (the share of recognised plays per scene per half-year, from dated published
             sets, against the next half-year's change in release share, and the reverse)
  python tools/demand_tests.py --supply ../signal-sonic/data/supply.json --sets ../signalgood/data/dj-sets.json --djindex ../signalgood/data/dj-index.json
"""
import argparse, base64, collections, json, math, random, re
import numpy as np

def rank(a): return np.argsort(np.argsort(a)).astype(float)
def spearman(x, y): x, y = np.asarray(x, float), np.asarray(y, float); return float(np.corrcoef(rank(x), rank(y))[0, 1]) if len(x) > 2 else float("nan")
def shuffled_p(x, y, groups, n=5000, seed=1):
    """two-sided p for Spearman against shuffles of y within each period (group), so period effects cannot fake it"""
    r0 = spearman(x, y); rnd = random.Random(seed); y = list(y); idx = collections.defaultdict(list)
    for i, g in enumerate(groups): idx[g].append(i)
    hits = 0
    for _ in range(n):
        yy = y[:]
        for g, ii in idx.items():
            vals = [y[i] for i in ii]; rnd.shuffle(vals)
            for i, v in zip(ii, vals): yy[i] = v
        if abs(spearman(x, yy)) >= abs(r0) - 1e-12: hits += 1
    return r0, (hits + 1) / (n + 1)

def main():
    ap = argparse.ArgumentParser()
    for k in ("supply", "sets", "djindex"): ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    S = json.load(open(a.supply)); M = S["months"]; sc = S["scenes"]
    blocks = [range(0, 6), range(6, 12), range(12, 18), range(18, 24)]
    res = {"months": [M[0], M[-1]]}
    # persistence of release share
    X, Y, G, same, n_same = [], [], [], 0, 0
    for s, v in sc.items():
        sh = v.get("share") or []
        if len(sh) < 24: continue
        b = [np.mean([sh[m] for m in bl]) for bl in blocks]
        if min(b) <= 0: continue
        d = [math.log(b[i + 1] / b[i]) for i in range(3)]
        for i in range(2):
            X.append(d[i]); Y.append(d[i + 1]); G.append(i)
            if abs(d[i]) >= math.log(1.1): n_same += 1; same += (d[i] > 0) == (d[i + 1] > 0)
    r, p = shuffled_p(X, Y, G)
    res["persistence"] = {"scenes": len(X) // 2, "pairs": len(X), "spearman_next_change_on_last": round(r, 2), "p_shuffled": round(p, 3),
                          "direction_repeats_after_a_10pc_move": f"{same} of {n_same}" if n_same else "none moved 10%"}
    # DJ plays by half-year
    X_ = json.load(open(a.djindex)); scn = X_["scenes"]; sci = np.frombuffer(base64.b64decode(X_["scene"]), dtype=np.uint8)[:X_["n"]]
    scene_of = {t: scn[int(sci[i])] for i, t in enumerate(X_["ids"]) if int(sci[i]) < len(scn)}
    plays = collections.defaultdict(collections.Counter); tot = collections.Counter()
    for st in json.load(open(a.sets))["sets"]:
        m = re.match(r"(\d{4})-(\d{2})", st[1] or "")
        if not m: continue
        h = f"{m[1]}H{1 if int(m[2]) <= 6 else 2}"
        for t in st[2]:
            if t in scene_of: plays[h][scene_of[t]] += 1; tot[h] += 1
    halves = ["2024H2", "2025H1", "2025H2", "2026H1"]
    res["plays_per_half"] = {h: tot[h] for h in halves}
    def rel_half(s, h):
        sh = sc[s]["share"]; idx = [i for i, mm in enumerate(M) if mm[:4] == h[:4] and ((int(mm[-2:]) <= 6) == (h[-1] == "1"))]
        return float(np.mean([sh[i] for i in idx])) if len(idx) >= 4 else None
    lead, rev, same_t = ([], [], []), ([], [], []), ([], [], [])
    for s in sc:
        if len(sc[s].get("share") or []) < 24: continue
        P = {h: (plays[h][s] / tot[h] if tot[h] else 0) for h in halves}; ok = {h: plays[h][s] >= 15 for h in halves}
        R = {h: rel_half(s, h) for h in halves}
        for i in range(len(halves) - 2):
            h0, h1, h2 = halves[i], halves[i + 1], halves[i + 2]
            if ok[h0] and ok[h1] and P[h0] > 0 and P[h1] > 0 and R[h1] and R[h2]:
                lead[0].append(math.log(P[h1] / P[h0])); lead[1].append(math.log(R[h2] / R[h1])); lead[2].append(i)
            if ok[h1] and ok[h2] and P[h1] > 0 and P[h2] > 0 and R[h0] and R[h1]:
                rev[0].append(math.log(R[h1] / R[h0])); rev[1].append(math.log(P[h2] / P[h1])); rev[2].append(i)
        for i in range(len(halves) - 1):
            h0, h1 = halves[i], halves[i + 1]
            if ok[h0] and ok[h1] and P[h0] > 0 and P[h1] > 0 and R[h0] and R[h1]:
                same_t[0].append(math.log(P[h1] / P[h0])); same_t[1].append(math.log(R[h1] / R[h0])); same_t[2].append(i)
    for name, (x, y, g) in (("plays_lead_releases", lead), ("releases_lead_plays", rev), ("same_half", same_t)):
        if len(x) >= 6: r, p = shuffled_p(x, y, g); res[name] = {"pairs": len(x), "spearman": round(r, 2), "p_shuffled": round(p, 3)}
        else: res[name] = {"pairs": len(x), "note": "too few scene-halves with 15 or more plays on both sides"}
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
