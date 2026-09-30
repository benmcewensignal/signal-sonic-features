"""One entry per DJ across Sonic's sources (site: data/dj-identity.json), linked by normalised name: their Beatport DJ
chart profile (how many charts, which scenes), their tracklisted sets, and Signal's artist page (bookings, cities, tier).
Only 29 DJs were in both the charts and the tracklists when this was written; the link is what lets charts (taste) and
sets (running order) describe the same people, and what "who would play it" will be built on.
  python tools/dj_identity.py --sonic ../signal-sonic --out ../signalgood/data/dj-identity.json
"""
import argparse, collections, glob, json, os, re, unicodedata
def nm(s): return " ".join(re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()).split())
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sonic", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    P = collections.defaultdict(lambda: {"names": set(), "chart_ids": set(), "charts": 0, "scenes": collections.Counter(), "sets": 0})
    cp = os.path.join(a.sonic, "data", "djcharts", "charts.jsonl")
    if os.path.exists(cp):
        for line in open(cp):
            try: r = json.loads(line)
            except Exception: continue
            k = nm(r.get("dj"))
            if not k: continue
            e = P[k]; e["names"].add(r.get("dj")); e["charts"] += 1
            if r.get("dj_id"): e["chart_ids"].add(str(r["dj_id"]))
            for g in r.get("genres") or []: e["scenes"][g] += 1
    for f in glob.glob(os.path.join(a.sonic, "data", "tracklists", "*.json")):
        if f.endswith("curves.json"): continue
        d = json.load(open(f)); k = nm(d.get("dj"))
        if k: P[k]["names"].add(d.get("dj")); P[k]["sets"] += len(d.get("sets", []))
    A = json.load(open(os.path.join(a.sonic, "data", "artist-lookup.json")))["artists"]; AN = {}
    for key, v in A.items(): AN[nm(v.get("n") or key)] = v
    out = {}
    for k, e in P.items():
        v = AN.get(k) or {}
        out[k] = {"name": sorted(e["names"], key=len)[0] if e["names"] else k, "beatport_dj": sorted(e["chart_ids"]), "charts": e["charts"],
                  "chart_scenes": [g for g, _ in e["scenes"].most_common(3)], "tracklisted_sets": e["sets"], "artist": bool(v),
                  "bookings": int(v.get("bk") or 0) if str(v.get("bk") or "0").isdigit() else 0, "tier": v.get("tier") if v else None}
    both = sum(1 for x in out.values() if x["charts"] and x["tracklisted_sets"])
    json.dump({"note": __doc__.split("\n  python")[0].strip(), "djs": out}, open(a.out, "w"), separators=(",", ":"))
    print(json.dumps({"djs": len(out), "with_charts": sum(1 for x in out.values() if x["charts"]), "with_sets": sum(1 for x in out.values() if x["tracklisted_sets"]),
                      "charts_and_sets": both, "known_artists": sum(1 for x in out.values() if x["artist"])}))
if __name__ == "__main__":
    main()
