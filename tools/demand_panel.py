"""Catalogue's demand panel (site: data/demand.json): is each scene growing, by two measures that do not depend on
what Sonic happened to collect.
  releases: Beatport's own count of records released per genre-month (signal-sonic data/supply.json), as a share of
            all Beatport releases; the last six months against the six before. Called up or down at 10% relative.
  DJ plays: the share of recognised tracks in published DJ sets (site data/dj-sets.json, dated in their titles);
            the last twelve months of sets against the twelve before. Called up or down beyond two standard errors.
Growing or shrinking only when both agree. Descriptive: whether growth persists has not been tested.
  python tools/demand_panel.py --supply ../signal-sonic/data/supply.json --sets ../signalgood/data/dj-sets.json \
      --djindex ../signalgood/data/dj-index.json --out ../signalgood/data/demand.json
"""
import argparse, base64, collections, datetime, json, math, re
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    for k in ("supply", "sets", "djindex", "out"): ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    S = json.load(open(a.supply)); months = S["months"]
    X = json.load(open(a.djindex)); scn = X["scenes"]; sc_idx = np.frombuffer(base64.b64decode(X["scene"]), dtype=np.uint8)[:X["n"]]
    scene_of = {t: scn[int(sc_idx[i])] for i, t in enumerate(X["ids"]) if int(sc_idx[i]) < len(scn)}
    sets = json.load(open(a.sets))["sets"]
    dated = []
    for s in sets:
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s[1] or "")
        if m: dated.append((datetime.date(int(m[1]), int(m[2]), int(m[3])), s[2]))
    last = max(d for d, _ in dated); cut1 = last - datetime.timedelta(days=365); cut0 = cut1 - datetime.timedelta(days=365)
    win = {"recent": collections.Counter(), "before": collections.Counter()}
    for d, ids in dated:
        w = "recent" if d > cut1 else ("before" if d > cut0 else None)
        if not w: continue
        for t in ids:
            if t in scene_of: win[w][scene_of[t]] += 1
    n1, n0 = sum(win["recent"].values()), sum(win["before"].values())
    out = {"note": __doc__.split("\n  python")[0].strip(), "release_months": [months[-12], months[-7], months[-6], months[-1]],
           "set_windows": [str(cut0), str(cut1), str(last)], "set_plays": {"recent": n1, "before": n0},
           "plays_gap": "most records played in recent sets arrived through DJ tracklists with no genre recorded, so their scene is unknown and DJ-play share cannot be compared yet",
           "scenes": {}}
    for sc in set(S["scenes"]) | set(win["recent"]) | set(win["before"]):
        row = {}
        v = S["scenes"].get(sc)
        if v and len(v.get("share") or []) >= 12:
            r_new, r_old = float(np.mean(v["share"][-6:])), float(np.mean(v["share"][-12:-6]))
            rel = (r_new - r_old) / r_old if r_old else 0.0
            row["releases"] = {"share": round(r_new * 100, 2), "before": round(r_old * 100, 2), "per_month": int(round(np.mean(v["total"][-6:]))),
                               "call": "up" if rel >= 0.10 else ("down" if rel <= -0.10 else "flat")}
        k1, k0 = win["recent"][sc], win["before"][sc]
        if n1 and n0 and (k1 + k0) >= 30:
            p1, p0 = k1 / n1, k0 / n0; se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0) or 1e-9
            row["plays"] = {"share": round(p1 * 100, 2), "before": round(p0 * 100, 2), "n": k1, "n_before": k0,
                            "call": "up" if (p1 - p0) > 2 * se else ("down" if (p0 - p1) > 2 * se else "flat")}
        calls = [row[k]["call"] for k in ("releases", "plays") if k in row]
        if len(calls) == 2:
            row["verdict"] = ("growing" if calls == ["up", "up"] else "shrinking" if calls == ["down", "down"]
                              else "mixed" if "up" in calls and "down" in calls else "no clear change")
        elif "releases" in row:   # DJ plays cannot be scened yet for most recent records (see the note)
            row["verdict"] = {"up": "releases rising", "down": "releases falling", "flat": "releases flat"}[row["releases"]["call"]]
        else:
            row["verdict"] = "not measured"
        out["scenes"][sc] = row
    json.dump(out, open(a.out, "w"), separators=(",", ":"))
    V = collections.Counter(r["verdict"] for r in out["scenes"].values())
    print(json.dumps({"scenes": len(out["scenes"]), "verdicts": V, "set_plays": out["set_plays"], "set_windows": out["set_windows"],
                      "growing": sorted(k for k, r in out["scenes"].items() if r["verdict"] == "growing"),
                      "shrinking": sorted(k for k, r in out["scenes"].items() if r["verdict"] == "shrinking")}))

if __name__ == "__main__":
    main()
