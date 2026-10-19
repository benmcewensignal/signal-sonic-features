"""The scenes' records as the Listen page's map has them (signalgood/data/walkmap.json: 7,068 records in 20 scenes), for
Swap's scene targets: worker/scene_ids.json, scene slug -> name and record ids. The worker averages the separated parts
of those ids. Run from a checkout that has the site beside it.
  python tools/build_scene_ids.py ../signalgood/data/walkmap.json"""
import json, sys, time
src = sys.argv[1] if len(sys.argv) > 1 else "../signalgood/data/walkmap.json"
d = json.load(open(src))
scenes = {s: {"name": d["names"].get(s, s), "ids": []} for s in d["scenes"]}
for tid, si in zip(d["ids"], d["scene"]): scenes[d["scenes"][si]]["ids"].append(tid)
out = {"built": time.strftime("%Y-%m-%d"), "source": "the Listen page's walk map", "scenes": scenes}
json.dump(out, open("worker/scene_ids.json", "w"), separators=(",", ":"))
print({s: len(v["ids"]) for s, v in scenes.items()})
