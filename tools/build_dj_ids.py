"""Each DJ's played records by id, from the published tracklists the pipeline matches to Beatport (signal-sonic's
data/tracklists, from MixesDB and others), for Swap's "point at a DJ": the worker averages the separated records a DJ
has played, part by part, to find the licensed loop nearest what they play. DJs with ten or more matched records; names
as the tracklists write them, looked up without case.
  python tools/build_dj_ids.py --tracklists ../signal-sonic/data/tracklists --out worker/dj_ids.json"""
import argparse, glob, json, os, collections
ap = argparse.ArgumentParser(); ap.add_argument("--tracklists", required=True); ap.add_argument("--out", default="worker/dj_ids.json")
ap.add_argument("--min", type=int, default=10); a = ap.parse_args()
by = collections.defaultdict(set); shown = {}; sets = collections.Counter()
for f in glob.glob(os.path.join(a.tracklists, "*.json")):
    if f.endswith("curves.json"): continue
    try: d = json.load(open(f))
    except Exception: continue
    dj = str(d.get("dj") or "").strip() if isinstance(d, dict) else ""
    if not dj: continue
    k = dj.lower(); shown.setdefault(k, dj)
    for s in d.get("sets") or []:
        got = [r.get("bp") for r in s.get("records") or [] if isinstance(r, dict) and r.get("bp")]
        if got: sets[k] += 1
        by[k].update(got)
out = {shown[k]: sorted(v) for k, v in sorted(by.items()) if len(v) >= a.min}
json.dump({"note": "DJ name -> the records they have played in published tracklists (ten or more matched), with how many sets; built by tools/build_dj_ids.py",
           "djs": out, "sets": {shown[k]: sets[k] for k in by if shown[k] in out}}, open(a.out, "w"), separators=(",", ":"), ensure_ascii=False)
print(f"{len(out):,} DJs with {a.min}+ matched records, {sum(len(v) for v in out.values()):,} ids")
