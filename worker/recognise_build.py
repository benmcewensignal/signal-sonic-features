"""The dense classics index for recognition in a room.

The site's index keeps about the first nine seconds of each record's fingerprints, which a room's reverb mostly destroys
(recognised 15 times in 100 in the window test, against 92 with every fingerprint). This builds an index of every
fingerprint of the classics: the canon files (whole previews, every version found), deduplicated by record, sorted by
hash into three arrays on the sonic-recognise volume, which the recognise endpoint memory-maps when it wakes.
  modal run worker/recognise_build.py
"""
import json, modal
app = modal.App("sonic-recognise-build")
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy")
vol = modal.Volume.from_name("sonic-recognise", create_if_missing=True)
REPO = "benmcewensignal/signal-sonic-audio"

@app.function(image=image, cpu=2.0, memory=12288, timeout=3600, volumes={"/idx": vol})
def build():
    import base64, os, time, urllib.request
    import numpy as np
    t0 = time.time()
    ls = json.load(urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/repos/{REPO}/contents/out", headers={"User-Agent": "sonic-recognise"})))
    files = sorted([f for f in ls if f["name"].startswith("fp-canon") and f["name"].endswith(".jsonl")], key=lambda f: f["name"])
    best = {}   # track_id -> (pass v2?, n, record) : prefer a whole-preview (v2) fingerprint, then the longer one
    for f in files:
        raw = urllib.request.urlopen(urllib.request.Request(f["download_url"], headers={"User-Agent": "sonic-recognise"}), timeout=300).read().decode("utf-8", "ignore")
        for line in raw.splitlines():
            try: d = json.loads(line)
            except Exception: continue
            if not d.get("found") or "hashes" not in d or not d.get("track_id"): continue
            key = (d.get("pass") == "v2", int(d.get("n") or 0))
            cur = best.get(d["track_id"])
            if cur is None or key > cur[0]: best[d["track_id"]] = (key, {k: d.get(k) for k in ("track_id", "name", "artists", "scene", "hashes", "frames", "query")})
    meta, Hs, Ts, Fs = [], [], [], []
    for ti, (_, d) in enumerate(best.values()):
        h = np.frombuffer(base64.b64decode(d["hashes"]), dtype="<u4"); fr = np.frombuffer(base64.b64decode(d["frames"]), dtype="<u2")
        n = min(len(h), len(fr))
        if n < 50: continue
        Hs.append(h[:n]); Fs.append(fr[:n]); Ts.append(np.full(n, len(meta), dtype=np.uint32))
        meta.append([d["track_id"], d.get("name"), d.get("artists") or [], d.get("scene"), d.get("query")])
    H = np.concatenate(Hs); T = np.concatenate(Ts); F = np.concatenate(Fs)
    o = np.argsort(H, kind="stable"); H, T, F = H[o], T[o], F[o]
    if len(meta) < 65536: T = T.astype(np.uint16)
    os.makedirs("/idx/classics", exist_ok=True)
    for nm, arr in (("H", H), ("T", T), ("F", F)): np.save(f"/idx/classics/{nm}.npy", arr)
    info = {"built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "records": len(meta), "fingerprints": int(len(H)), "files": len(files), "seconds": round(time.time() - t0)}
    json.dump({"info": info, "tracks": meta}, open("/idx/classics/meta.json", "w"))
    vol.commit()
    return info

@app.local_entrypoint()
def main():
    info = build.remote(); print("::notice title=dense classics index::" + json.dumps(info))
