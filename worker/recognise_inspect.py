"""Which records are in the live catalogue index, and under what names: reads data/inspect-ids.json, prints one line per id.
  modal run worker/recognise_inspect.py"""
import json, modal
app = modal.App("sonic-recognise-inspect")
image = modal.Image.debian_slim(python_version="3.12")
vol = modal.Volume.from_name("sonic-recognise", create_if_missing=True)

@app.function(image=image, memory=4096, timeout=600, volumes={"/idx": vol})
def look(ids):
    import os
    meta = json.load(open("/idx/all/meta.json")); by = {m[0]: m for m in meta["tracks"]}
    return {"info": meta["info"], "active": os.path.exists("/idx/all/ACTIVE"),
            "rows": [[i, i in by, (by.get(i) or [None, None])[1], ((by.get(i) or [None, None, []])[2] or [])[:2], bool((by.get(i) or [0]*5)[4])] for i in ids]}

@app.local_entrypoint()
def main():
    r = look.remote(json.load(open("data/inspect-ids.json"))); json.dump(r, open("data/inspect-out.json", "w"))
    print("INSPECT " + json.dumps(r))
