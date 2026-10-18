"""Each artist's records by id, from the pipeline database's catalogue entries, for Swap's "like an artist": the worker
averages an artist's separated records part by part (record-parts.npz on the embed volume) to find the licensed loop
nearest that artist's drums, bass, melody or voice. Artists with three or more records; names as the catalogue writes
them, looked up without case.
  python tools/build_artist_ids.py --db sonic.db --out worker/artist_ids.json"""
import argparse, json, sqlite3, collections
ap = argparse.ArgumentParser(); ap.add_argument("--db", required=True); ap.add_argument("--out", default="worker/artist_ids.json")
ap.add_argument("--min", type=int, default=3); a = ap.parse_args()
c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
by = collections.defaultdict(set); shown = {}
for tid, arts in c.execute("select track_id, artists from track_meta where artists is not null"):
    try: names = json.loads(arts)
    except Exception: continue
    for n in names if isinstance(names, list) else []:
        n = str(n or "").strip()
        if not n: continue
        k = n.lower(); by[k].add(tid); shown.setdefault(k, n)
out = {shown[k]: sorted(v) for k, v in sorted(by.items()) if len(v) >= a.min}
json.dump({"note": "artist name -> record ids (three or more), from the catalogue entries; built by tools/build_artist_ids.py",
           "artists": out}, open(a.out, "w"), separators=(",", ":"))
print(f"{len(out):,} artists with {a.min}+ records, {sum(len(v) for v in out.values()):,} ids")
