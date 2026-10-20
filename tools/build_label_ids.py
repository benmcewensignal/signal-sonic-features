"""Each label's records by id, from the pipeline database's catalogue entries, for Swap's "point at a label": the worker
averages a label's separated records part by part (record-parts.npz on the embed volume) to find the licensed loop
nearest that label's drums, bass, melody or voice. Labels with five or more records; names as the catalogue writes them,
looked up without case.
  python tools/build_label_ids.py --db sonic.db --out worker/label_ids.json"""
import argparse, json, sqlite3, collections
ap = argparse.ArgumentParser(); ap.add_argument("--db", required=True); ap.add_argument("--out", default="worker/label_ids.json")
ap.add_argument("--min", type=int, default=5); a = ap.parse_args()
c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
by = collections.defaultdict(set); shown = {}
for tid, lab in c.execute("select track_id, label from track_meta where label is not null and label != ''"):
    n = str(lab).strip()
    if not n: continue
    k = n.lower(); by[k].add(tid); shown.setdefault(k, n)
out = {shown[k]: sorted(v) for k, v in sorted(by.items()) if len(v) >= a.min}
json.dump({"note": "label name -> record ids (five or more), from the catalogue entries; built by tools/build_label_ids.py", "labels": out},
          open(a.out, "w"), separators=(",", ":"), ensure_ascii=False)
print(f"{len(out):,} labels with {a.min}+ records, {sum(len(v) for v in out.values()):,} ids")
