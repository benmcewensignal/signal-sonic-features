"""The DJ tool's published-set file (site: data/dj-sets.json), rebuilt from the tracklist files.
next: for each record the DJ index places, the records played straight after it, how often and by whom (up to three
      DJs). Only true neighbours count: a tracklist line with no matched record is a gap, so the records either side
      of it are not joined (the version before this dropped unmatched lines and joined across them).
sets: the published sets with six or more walkable records, in order (for the set check's closest sets).
norms: how tight real sets are, for the set check; carried from the previous file with the counts updated.
  python tools/dj_sets.py --tracklists ../signal-sonic/data/tracklists --djindex ../signalgood/data/dj-index.json \
      --prev ../signalgood/data/dj-sets.json --out ../signalgood/data/dj-sets.json
"""
import argparse, base64, collections, glob, json, os, numpy as np

def main():
    ap = argparse.ArgumentParser()
    for k in ("tracklists", "djindex", "out"): ap.add_argument("--" + k, required=True)
    ap.add_argument("--prev", default=None); a = ap.parse_args()
    X = json.load(open(a.djindex)); ids = X["ids"]; place = set(ids); at = {t: i for i, t in enumerate(ids)}; n = X["n"]
    P6 = ("bright", "vocal", "drums", "bass", "busy", "punch")   # the order the page's set check reads
    MV = {m: np.frombuffer(base64.b64decode(X[m]), np.uint8)[:n].astype(np.float32) for m in P6}
    D = X["eardims"]; EV = np.frombuffer(base64.b64decode(X["ear"]), np.int8).astype(np.float32).reshape(n, D) * np.array(X["earscale"], np.float32) / 127
    EV /= np.maximum(np.linalg.norm(EV, axis=1, keepdims=True), 1e-9)
    has = np.frombuffer(base64.b64decode(X["earhas"]), np.uint8)[:X["n"]].astype(bool); walk = {t for t, h in zip(ids, has) if h}
    nxt = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, []])); sets = []; pairs = 0
    for f in sorted(glob.glob(os.path.join(a.tracklists, "*.json"))):
        if f.endswith("curves.json") or f.endswith("measured.jsonl"): continue
        d = json.load(open(f)); dj = d.get("dj") or os.path.basename(f)[:-5]
        for st in d.get("sets", []):
            seq = [r.get("bp") if r.get("bp") in place else None for r in st.get("records", [])]
            for x, y in zip(seq, seq[1:]):
                if x and y and x != y:
                    e = nxt[x][y]; e[0] += 1; pairs += 1
                    if dj not in e[1] and len(e[1]) < 3: e[1].append(dj)
            w = [t for t in seq if t in walk]
            if len(w) >= 6:
                ix = [at[t] for t in w]; cen = [int(round(float(MV[m][ix].mean()))) for m in P6]
                ec = EV[ix].mean(0); ec = ec / (np.linalg.norm(ec) or 1)
                sets.append([dj, st.get("title", ""), w, cen, [round(float(v), 4) for v in ec]])   # the set check needs each set's measure averages and its place in the ear
    prev = json.load(open(a.prev)) if a.prev and os.path.exists(a.prev) else {}
    norms = dict(prev.get("norms") or {}); norms["sets"] = len(sets); norms["transitions"] = pairs
    out = {"note": "what working DJs played next, from published tracklists: for each record, the records played straight after it (true neighbours only: an unmatched line is a gap), how often, and by whom (up to three DJs); and the published sets with six or more walkable records, in order. norms carried from the previous build with the counts updated",
           "next": {x: sorted([[y, v[0], v[1]] for y, v in m.items()], key=lambda z: -z[1]) for x, m in nxt.items()}, "sets": sets, "norms": norms}
    json.dump(out, open(a.out, "w"), separators=(",", ":"))
    print(json.dumps({"sets": len(sets), "records_with_next": len(nxt), "adjacent_pairs": pairs}))

if __name__ == "__main__":
    main()
