"""Calibrating the catalogue-wide index before it goes live: room-degraded 25-second clips of catalogue records and of
classics (both in the index), and of charted records Sonic has never fingerprinted (outside it), fingerprinted with the
store's own code and matched with the endpoint's own rule, in-process against all/. Writes data/recognise-test-all.json.
  modal run worker/recognise_test_all.py          (add --activate to mark the index live if no outside record is named)"""
import json, modal
app = modal.App("sonic-recognise-test-all")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg", "libsndfile1").pip_install("numpy", "scipy", "librosa==0.10.2", "soundfile")
         .add_local_file("worker/modal_app.py", "/root/modal_app_src.py"))
vol = modal.Volume.from_name("sonic-recognise", create_if_missing=True)

@app.function(image=image, cpu=4.0, memory=16384, timeout=5400, volumes={"/idx": vol})
def run(activate: bool = False):
    import os, random, sys, tempfile, urllib.request, json as J
    import numpy as np
    from scipy.signal import butter, sosfilt, fftconvolve
    UA = {"User-Agent": "sonic-recognise"}
    open("/tmp/fpmod.py", "wb").write(urllib.request.urlopen(urllib.request.Request("https://raw.githubusercontent.com/benmcewensignal/signal-sonic/main/sonic/fingerprints.py", headers=UA)).read())
    sys.path.insert(0, "/tmp"); import fpmod as FP
    src = open("/root/modal_app_src.py").read(); i = src.find("def _rx_group(m):"); j = src.find("@app.function(image=light", i); ns = {}; exec(src[i:j], ns); rx_match = ns["rx_match"]; grp = ns["_rx_group"]
    meta = J.load(open("/idx/all/meta.json")); R = {"H": np.load("/idx/all/H.npy"), "T": np.load("/idx/all/T.npy", mmap_mode="r"), "F": np.load("/idx/all/F.npy", mmap_mode="r"), "meta": meta}
    inidx = {m[0] for m in meta["tracks"]}; classics = [m for m in meta["tracks"] if m[4]]; G = {m[0]: grp(m) for m in meta["tracks"]}
    get = lambda u: J.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=180))
    PV = get("https://raw.githubusercontent.com/benmcewensignal/signalgood/main/data/previews.json").get("u", {})
    CT = get("https://raw.githubusercontent.com/benmcewensignal/signal-sonic/main/data/chart-tracks.json")
    for t, v in CT.items():
        if v and len(v) > 4 and v[4]: PV.setdefault(t, v[4])
    pick = random.Random(11); rng = np.random.default_rng(11)
    cat = [m for m in meta["tracks"] if not m[4] and m[0] in PV]; pick.shuffle(cat)
    cls = [m for m in classics if m[0] in PV]; pick.shuffle(cls)
    out_ = [t for t in CT if t not in inidx and t in PV]; pick.shuffle(out_)
    def room(y):
        sr = FP.SR; n = int(0.6 * sr); ir = rng.normal(0, 1, n) * np.exp(-np.arange(n) / (0.12 * sr)); ir[0] = 1.0
        w = fftconvolve(y, ir)[: len(y)]; w = sosfilt(butter(4, [200, 6000], btype="band", fs=sr, output="sos"), w)
        w = w / (np.max(np.abs(w)) + 1e-9); return (w + rng.normal(0, 10 ** (-15 / 20) * np.sqrt(np.mean(w ** 2)), len(w))).astype(np.float32)
    def ask(tid):
        fd, p = tempfile.mkstemp(suffix=".mp3"); os.close(fd)
        try: urllib.request.urlretrieve(PV[tid], p); y = FP.load_audio(p, max_seconds=None)
        finally: os.remove(p)
        L = int(FP.SR * 25); st = pick.randint(0, max(0, len(y) - L - 1))
        return rx_match([[int(h), int(f)] for h, f in FP.hashes(room(y[st:st + L]))][:20000], R)
    res = {}
    for kind, L_ in (("catalogue", [m[0] for m in cat[:60]]), ("classics", [m[0] for m in cls[:40]]), ("outside", out_[:80])):
        rows = []
        for tid in L_:
            try: r = ask(tid)
            except Exception as e: rows.append({"id": tid, "error": type(e).__name__}); continue
            same = r.get("found") and (r["track_id"] == tid or (tid in G and G.get(r["track_id"]) == G[tid]))
            rows.append({"id": tid, "found": bool(r.get("found")), "right": bool(same), "wrong": bool(r.get("found") and not same), "path": r.get("path"), "votes": r.get("votes"), "next": r.get("next_best")})
        res[kind] = rows
    summ = {k: {"n": len(v), "right": sum(x.get("right", False) for x in v), "wrong": sum(x.get("wrong", False) for x in v), "errors": sum(1 for x in v if x.get("error"))} for k, v in res.items()}
    activated = False
    if activate and summ["outside"]["wrong"] == 0 and summ["catalogue"]["wrong"] == 0 and summ["classics"]["wrong"] == 0 and summ["outside"]["n"] >= 60:
        open("/idx/all/ACTIVE", "w").write(J.dumps(summ)); vol.commit(); activated = True
    return {"summary": summ, "activated": activated, "results": res, "index": meta["info"]}

@app.local_entrypoint()
def main(activate: bool = False):
    r = run.remote(activate); json.dump(r, open("data/recognise-test-all.json", "w"))
    print("::notice title=catalogue index test::" + json.dumps({"summary": r["summary"], "activated": r["activated"], "index": r["index"]}))
