"""The catalogue-wide dense index: every record Sonic has fingerprinted, room-grade.

The site's own index keeps about the first nine seconds of each record's fingerprints, which a room destroys (about 15 in
100 recognised); 8,000 fingerprints spread across the whole preview were recognised 69 times in 100 in the density test.
This reads the main fingerprint store (every fingerprint of each whole preview, in the fp-store release of signal-sonic),
spreads each record's to 8,000, keeps the classics complete (the canon files), and writes one index, sorted by hash, to
the sonic-recognise volume under all/. The recognise endpoint prefers it when it exists.
  modal run worker/recognise_build_all.py
"""
import json, modal
app = modal.App("sonic-recognise-build-all")
image = modal.Image.debian_slim(python_version="3.12").apt_install("curl").pip_install("numpy")
vol = modal.Volume.from_name("sonic-recognise", create_if_missing=True)
PER = 8000

@app.function(image=image, cpu=4.0, memory=32768, ephemeral_disk=80_000, timeout=7200, volumes={"/idx": vol})
def build():
    import base64, os, sqlite3, subprocess, time, urllib.request
    import numpy as np
    t0 = time.time(); UA = {"User-Agent": "sonic-recognise"}
    rel = json.load(urllib.request.urlopen(urllib.request.Request("https://api.github.com/repos/benmcewensignal/signal-sonic/releases/tags/fp-store", headers=UA)))
    A = {a["name"]: a["browser_download_url"] for a in rel["assets"]}
    SET, N = urllib.request.urlopen(urllib.request.Request(A["fpstore.current"], headers=UA)).read().decode().split()[:2]
    parts = [A[f"fpstore-{SET}.gz.{i:02d}"] for i in range(int(N))]
    subprocess.run("curl -sL " + " ".join(f"'{u}'" for u in parts) + " | gunzip > /tmp/fp.db", shell=True, check=True)
    print("store:", round(os.path.getsize("/tmp/fp.db") / 1e9, 2), "GB", flush=True)
    # the classics, complete (as in the classics build)
    ls = json.load(urllib.request.urlopen(urllib.request.Request("https://api.github.com/repos/benmcewensignal/signal-sonic-audio/contents/out", headers=UA)))
    best = {}
    for f in sorted([f for f in ls if f["name"].startswith("fp-canon") and f["name"].endswith(".jsonl")], key=lambda f: f["name"]):
        raw = urllib.request.urlopen(urllib.request.Request(f["download_url"], headers=UA), timeout=300).read().decode("utf-8", "ignore")
        for line in raw.splitlines():
            try: d = json.loads(line)
            except Exception: continue
            if not d.get("found") or "hashes" not in d or not d.get("track_id"): continue
            key = (d.get("pass") == "v2", int(d.get("n") or 0)); cur = best.get(d["track_id"])
            if cur is None or key > cur[0]: best[d["track_id"]] = (key, {k: d.get(k) for k in ("track_id", "name", "artists", "scene", "hashes", "frames", "query")})
    # names for the catalogue, from the recognition index
    IX = json.load(urllib.request.urlopen(urllib.request.Request("https://raw.githubusercontent.com/benmcewensignal/signal-sonic-audio/main/out/index.json", headers=UA), timeout=300))
    nm = {t.get("track_id"): t for t in IX.get("tracks", [])}
    meta, Hs, Ts, Fs = [], [], [], []
    def add(h, fr, m):
        n = min(len(h), len(fr))
        if n < 50: return
        Hs.append(h[:n]); Fs.append(fr[:n]); Ts.append(np.full(n, len(meta), dtype=np.uint32)); meta.append(m)
    for _, d in best.values():
        add(np.frombuffer(base64.b64decode(d["hashes"]), "<u4"), np.frombuffer(base64.b64decode(d["frames"]), "<u2"),
            [d["track_id"], d.get("name"), d.get("artists") or [], d.get("scene"), d.get("query")])
    classics = len(meta); con = sqlite3.connect("/tmp/fp.db")
    for tid, hb, fb in con.execute("SELECT track_id, hashes, frames FROM fp_tracks"):
        if tid in best: continue
        h = np.frombuffer(hb, "<u4"); fr = np.frombuffer(fb, "<u2"); n = min(len(h), len(fr))
        if n > PER:
            o = np.argsort(fr[:n], kind="stable"); sel = o[np.unique(np.linspace(0, n - 1, PER).astype(np.int64))]; h, fr = h[sel], fr[sel]
        t = nm.get(tid) or {}
        add(np.ascontiguousarray(h), np.ascontiguousarray(fr), [tid, t.get("name"), t.get("artists") or [], t.get("scene"), None])
    con.close(); os.remove("/tmp/fp.db")
    H = np.concatenate(Hs); del Hs; T = np.concatenate(Ts); del Ts; F = np.concatenate(Fs); del Fs
    o = np.argsort(H, kind="stable"); H = H[o]; T = T[o]; F = F[o]; del o
    T = T.astype(np.uint16) if len(meta) < 65536 else T
    os.makedirs("/idx/all", exist_ok=True)
    for nm_, arr in (("H", H), ("T", T), ("F", F)): np.save(f"/idx/all/{nm_}.npy", arr)
    info = {"built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "records": len(meta), "classics": classics, "fingerprints": int(len(H)), "per_record": PER, "seconds": round(time.time() - t0)}
    json.dump({"info": info, "tracks": meta}, open("/idx/all/meta.json", "w")); vol.commit()
    return info

@app.local_entrypoint()
def main():
    info = build.remote(); print("::notice title=catalogue dense index::" + json.dumps(info))
