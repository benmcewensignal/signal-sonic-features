"""How DJs mix in each scene, from real transitions in published tracklists (site: data/scene-practice.json), and the
defaults Build a set takes from it: key ('mix' where real neighbours are key-compatible at least 1.4 times as often
as random records at a mixable tempo in the same scene, otherwise 'any'), tempo (±8% where it covers 80 in 100 real
transitions, otherwise ±16%: wider would offer records at any tempo, and the ranker already weighs tempo), and scene ('in' where DJs stay in the scene more than half the time).
  python tools/scene_practice.py --tracklists ../signal-sonic/data/tracklists --djindex ../signalgood/data/dj-index.json --out ../signalgood/data/scene-practice.json
"""
import argparse, base64, collections, glob, json, os, numpy as np
def main():
    ap = argparse.ArgumentParser()
    for k in ("tracklists", "djindex", "out"): ap.add_argument("--" + k, required=True)
    a = ap.parse_args(); rng = np.random.default_rng(0)
    X = json.load(open(a.djindex)); n = X["n"]; at = {t: i for i, t in enumerate(X["ids"])}; SCN = X["scenes"]
    tm = np.frombuffer(base64.b64decode(X["tempo"]), np.uint16)[:n].astype(np.float32) / 10; key = np.frombuffer(base64.b64decode(X["key"]), np.uint8)[:n]
    sc = np.frombuffer(base64.b64decode(X["scene"]), np.uint8)[:n]
    def mixk(u, v):
        if u == 255 or v == 255: return None
        u, v = int(u), int(v); return u == v or u % 12 == v % 12 or ((u < 12) == (v < 12) and (u - v) % 12 in (1, 11))
    P = collections.defaultdict(list)
    for f in glob.glob(os.path.join(a.tracklists, "*.json")):
        if f.endswith("curves.json"): continue
        for st in json.load(open(f)).get("sets", []):
            s = [at.get(r.get("bp"), -1) for r in st.get("records", [])]
            for x, y in zip(s, s[1:]):
                if x >= 0 and y >= 0 and x != y and tm[x] and tm[y] and sc[x] < len(SCN): P[SCN[sc[x]]].append((x, y))
    out = {}
    for scn, pr in P.items():
        if len(pr) < 150 or scn == "unknown": continue
        pool = np.flatnonzero(sc == SCN.index(scn))
        kr = [z for z in (mixk(key[x], key[y]) for x, y in pr) if z is not None]; kc = []
        for x, _ in pr:
            c = pool[np.abs(tm[pool] - tm[x]) / tm[x] <= 0.08]
            for y in (rng.choice(c, min(10, len(c))) if len(c) else []):
                z = mixk(key[x], key[y])
                if z is not None: kc.append(z)
        d = np.array([min(abs(tm[y] * f - tm[x]) / tm[x] for f in (1, 2, 0.5)) for x, y in pr])
        lift = float(np.mean(kr) / max(np.mean(kc), 1e-9)); w8, w16 = float(np.mean(d <= 0.08)), float(np.mean(d <= 0.16)); stay = float(np.mean([sc[x] == sc[y] for x, y in pr]))
        out[scn] = {"pairs": len(pr), "key_lift": round(lift, 2), "within_3": round(float(np.mean(d <= 0.03)), 3), "within_8": round(w8, 3), "within_16": round(w16, 3), "stay": round(stay, 3),
                    "defaults": {"key": "mix" if lift >= 1.4 else "any", "tempo": 8 if w8 >= 0.8 else 16, "scene": "in" if stay > 0.5 else "any"}}
    json.dump({"note": __doc__.split("\n  python")[0].strip(), "scenes": out}, open(a.out, "w"), indent=0)
    print(json.dumps({s: v["defaults"] for s, v in sorted(out.items())}))
if __name__ == "__main__":
    main()
