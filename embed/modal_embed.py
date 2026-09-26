"""A learned embedding for Sonic, pilot. Spectrogram patches from corpus previews, a compact CNN trained
on scene with whole artists held out, scored at record level against the hand-built model's 48%.
  modal run embed/modal_embed.py --manifest-path manifest.json [--stage extract|train|all]
"""
import json, modal
app = modal.App("sonic-embed")
vol = modal.Volume.from_name("sonic-embed", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg", "libsndfile1")
         .pip_install("numpy<2", "librosa==0.10.2", "soundfile", "requests", "torch==2.4.1"))
W = 188   # a patch: three seconds at 16 kHz, hop 256


@app.function(image=image, volumes={"/data": vol}, timeout=1800, cpu=2, retries=1, max_containers=16)
def extract(batch):
    import os, tempfile, numpy as np, librosa, requests
    import time, collections
    os.makedirs("/data/patches", exist_ok=True); done = 0; why = collections.Counter()
    for tid, url in batch:
        p = f"/data/patches/{tid.replace(':', '_')}.npy"
        if os.path.exists(p): done += 1; why["already"] += 1; continue
        try:
            for attempt in range(4):   # throttling and server errors: wait and try again
                r = requests.get(url, timeout=40, headers={"User-Agent": "signal-sonic"})
                if r.status_code in (429, 500, 502, 503, 504): time.sleep(3 * (attempt + 1) ** 2); continue
                break
            if r.status_code != 200: why[f"http {r.status_code}"] += 1; continue
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f: f.write(r.content); fn = f.name
            y, _ = librosa.load(fn, sr=16000, mono=True, duration=120); os.remove(fn)
            M = np.log1p(1000 * librosa.feature.melspectrogram(y=y, sr=16000, n_fft=512, hop_length=256, n_mels=96)).astype(np.float16)
            if M.shape[1] < 2 * W: continue
            np.save(p, np.stack([M[:, s:s + W] for s in np.linspace(0, M.shape[1] - W, 8).astype(int)])); done += 1; why["new"] += 1
        except Exception as e:
            why[type(e).__name__] += 1
    vol.commit(); return dict(why)


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=14400, memory=65536)
def train(manifest, epochs: int = 20):
    import os, random, numpy as np, torch, torch.nn as nn
    vol.reload()
    items = [m for m in manifest if m.get("scene") and os.path.exists(f"/data/patches/{m['id'].replace(':', '_')}.npy")]
    scenes = sorted({m["scene"] for m in items}); si = {s: i for i, s in enumerate(scenes)}
    arts = sorted({m["artist"] for m in items}); random.Random(0).shuffle(arts); hold = set(arts[:len(arts) // 5])
    tr = [m for m in items if m["artist"] not in hold]; te = [m for m in items if m["artist"] in hold]
    load = lambda m: np.load(f"/data/patches/{m['id'].replace(':', '_')}.npy")   # kept at half precision; converted per batch
    print('loading', len(tr), 'training and', len(te), 'test records across', len(scenes), 'scenes', flush=True)
    from concurrent.futures import ThreadPoolExecutor
    def fill(ms):   # one pre-sized array, filled in place: stacking a list would hold everything twice
        X = np.empty((len(ms), 8, 96, W), np.float16)
        def put(i): X[i] = load(ms[i])
        with ThreadPoolExecutor(48) as ex: list(ex.map(put, range(len(ms))))
        return X
    Xtr = fill(tr); Xte = fill(te)
    ytr = np.array([si[m["scene"]] for m in tr]); yte = np.array([si[m["scene"]] for m in te])
    smp = Xtr[:2000].astype(np.float32); mu, sd = float(smp.mean()), float(smp.std() + 1e-6)
    dev = "cuda"
    def block(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
    class Net(nn.Module):
        def __init__(s, n):
            super().__init__(); s.f = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 256)); s.emb = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3)); s.out = nn.Linear(128, n)
        def embed(s, x):
            h = s.f(x.unsqueeze(1)); return s.emb(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))
        def forward(s, x): return s.out(s.embed(x))
    net = Net(len(scenes)).to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * (len(tr) * 8 // 256 + 1)); lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    P = torch.from_numpy(Xtr.reshape(-1, 96, W)); Y = torch.tensor(np.repeat(ytr, 8))
    def evaluate():
        net.eval(); outs = []
        with torch.no_grad():
            for i in range(0, len(Xte), 64):
                x = ((torch.from_numpy(Xte[i:i + 64].astype(np.float32)) - mu) / sd).to(dev); b = x.shape[0]
                outs.append(torch.softmax(net(x.reshape(-1, 96, W)), 1).reshape(b, 8, -1).mean(1).cpu().numpy())
        p = np.concatenate(outs); net.train()
        return float(np.mean(p.argmax(1) == yte)), float(np.mean([yte[k] in np.argsort(-p[k])[:3] for k in range(len(yte))]))
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(len(P))
        for i in range(0, len(P), 256):
            idx = perm[i:i + 256]; x = ((P[idx].float() - mu) / sd).to(dev); y = Y[idx].to(dev)
            f = torch.randint(0, 80, (1,)).item(); x[:, f:f + 12, :] = 0   # a little frequency masking
            opt.zero_grad(); l = lossf(net(x), y); l.backward(); opt.step(); sched.step()
        if ep % 5 == 4 or ep == epochs - 1: a1, a3 = evaluate(); hist.append([ep + 1, round(a1, 3), round(a3, 3)]); print("epoch", ep + 1, a1, a3, flush=True)
    torch.save({"state": net.state_dict(), "scenes": scenes, "mu": mu, "sd": sd}, f"/data/embed_{len(items)}.pt"); vol.commit()
    a1, a3 = evaluate()
    return {"records": len(items), "train": len(tr), "test": len(te), "artists_held_out": len(hold), "first": round(a1, 3), "top3": round(a3, 3), "history": hist}


@app.function(image=image, volumes={"/data": vol}, timeout=900)
def split(manifest):
    """The exact split train() uses: the records whose patches exist, with a fifth of artists held out."""
    import os, random
    vol.reload()
    items = [m for m in manifest if m.get("scene") and os.path.exists(f"/data/patches/{m['id'].replace(':', '_')}.npy")]
    arts = sorted({m["artist"] for m in items}); random.Random(0).shuffle(arts); hold = set(arts[:len(arts) // 5])
    return {"train": [[m["id"], m["scene"]] for m in items if m["artist"] not in hold], "test": [[m["id"], m["scene"]] for m in items if m["artist"] in hold], "held_artists": sorted(hold)}



def make_net(n):
    import torch.nn as nn, torch
    def block(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
    class Net(nn.Module):
        def __init__(s, n):
            super().__init__(); s.f = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 256)); s.emb = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3)); s.out = nn.Linear(128, n)
        def embed(s, x):
            h = s.f(x.unsqueeze(1)); return s.emb(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))
        def forward(s, x): return s.out(s.embed(x))
    return Net(n)


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600, memory=16384)
def calibrate(manifest):
    """Temperature and reliability for the latest trained model, on its held-out records; saved as embed_live.pt."""
    import os, glob, random, numpy as np, torch
    vol.reload()
    ck = sorted(glob.glob("/data/embed_*.pt"), key=lambda f: int(f.split("_")[-1].split(".")[0]) if f.split("_")[-1].split(".")[0].isdigit() else 0)[-1]
    C = torch.load(ck, map_location="cuda"); scenes = C["scenes"]; si = {s: i for i, s in enumerate(scenes)}
    net = make_net(len(scenes)).cuda(); net.load_state_dict(C["state"]); net.eval()
    items = [m for m in manifest if m.get("scene") in si and os.path.exists(f"/data/patches/{m['id'].replace(':', '_')}.npy")]
    arts = sorted({m["artist"] for m in items}); random.Random(0).shuffle(arts); hold = set(arts[:len(arts) // 5])
    te = [m for m in items if m["artist"] in hold]; y = np.array([si[m["scene"]] for m in te]); P = []
    from concurrent.futures import ThreadPoolExecutor
    ld = lambda m: np.load(f"/data/patches/{m['id'].replace(':', '_')}.npy").astype(np.float32)
    with torch.no_grad():
        for i in range(0, len(te), 64):
            with ThreadPoolExecutor(32) as ex: x = np.stack(list(ex.map(ld, te[i:i + 64])))
            b = x.shape[0]
            x = ((torch.from_numpy(x) - C["mu"]) / C["sd"]).cuda().reshape(-1, 96, W)
            P.append(torch.softmax(net(x), 1).reshape(b, 8, -1).mean(1).cpu().numpy())
    P = np.concatenate(P); rng = np.random.default_rng(0); idx = rng.permutation(len(te)); A, B = idx[:len(idx) // 2], idx[len(idx) // 2:]
    def cal(p, T): q = np.power(np.clip(p, 1e-9, 1), 1 / T); return q / q.sum(1, keepdims=True)
    Ts = np.arange(0.5, 3.01, 0.05); T = float(Ts[np.argmin([-np.mean(np.log(cal(P[A], t)[np.arange(len(A)), y[A]])) for t in Ts])])
    Q = cal(P, T); pred = Q.argmax(1); conf = Q.max(1); BINS = ((0.6, 1.01), (0.4, 0.6), (0.0, 0.4))
    tiers = [[lo, hi, round(float(np.mean(pred[B][(conf[B] >= lo) & (conf[B] < hi)] == y[B][(conf[B] >= lo) & (conf[B] < hi)])), 3) if ((conf[B] >= lo) & (conf[B] < hi)).sum() >= 30 else None] for lo, hi in BINS]
    scene_tiers = {}
    for k, sname in enumerate(scenes):
        rows = []
        for lo, hi in BINS:
            m = B[(pred[B] == k) & (conf[B] >= lo) & (conf[B] < hi)]
            rows.append([lo, hi, round(float(np.mean(pred[m] == y[m])), 3) if len(m) >= 30 else None])
        scene_tiers[sname] = rows
    first = float(np.mean(pred == y)); top3 = float(np.mean([y[i] in np.argsort(-Q[i])[:3] for i in range(len(y))]))
    C.update({"temperature": T, "tiers": tiers, "scene_tiers": scene_tiers, "held_out_accuracy": round(first, 3), "held_out_top3": round(top3, 3), "held_out_records": len(te), "trained_on": len(items) - len(te), "built": "2026-09-26", "from": os.path.basename(ck)})
    torch.save({k: (v if k != "state" else {kk: vv.cpu() for kk, vv in v.items()}) for k, v in C.items()}, "/data/embed_live.pt"); vol.commit()
    return {"temperature": T, "tiers": tiers, "first": round(first, 3), "top3": round(top3, 3), "records": len(te), "from": os.path.basename(ck)}


CONDS = ["clean", "clip 60 s", "first 30 s", "middle 30 s", "last 30 s", "phone", "64 kbps", "12 dB quieter"]


@app.function(image=image, volumes={"/data": vol}, timeout=1800, cpu=2, retries=1, max_containers=12)
def robust_batch(batch):
    """Each held-out record downloaded again, degraded eight ways, and read by the live model."""
    import os, subprocess, tempfile, numpy as np, librosa, requests, torch
    vol.reload(); C = torch.load("/data/embed_live.pt", map_location="cpu"); net = make_net(len(C["scenes"])); net.load_state_dict(C["state"]); net.eval()
    si = {s_: i for i, s_ in enumerate(C["scenes"])}; out = []; rng = np.random.default_rng(0)
    def patches(y, a=0.0, b=1.0):
        M = np.log1p(1000 * librosa.feature.melspectrogram(y=y, sr=16000, n_fft=512, hop_length=256, n_mels=96)).astype(np.float32)
        lo, hi = int(a * M.shape[1]), int(b * M.shape[1]); M = M[:, lo:hi]
        if M.shape[1] < W: return None
        return np.stack([M[:, s_:s_ + W] for s_ in np.linspace(0, M.shape[1] - W, 8).astype(int)])
    def read(x):
        with torch.no_grad(): p = torch.softmax(net((torch.from_numpy(x) - C["mu"]) / C["sd"]), 1).mean(0).numpy()
        return int(p.argmax())
    def phone(y):   # a phone in a room: band-limited, a short room tail, and background noise
        F = np.fft.rfft(y); f = np.fft.rfftfreq(len(y), 1 / 16000); F[(f < 200) | (f > 6000)] = 0; z = np.fft.irfft(F, len(y))
        ir = rng.standard_normal(2400) * np.exp(-np.arange(2400) / 500); ir[0] = 1; z = np.convolve(z, ir / np.abs(ir).sum() * 4, mode="same")
        return (z + rng.standard_normal(len(z)) * np.std(z) * 0.1).astype(np.float32)
    for tid, url, scene in batch:
        if scene not in si: continue
        try:
            r = requests.get(url, timeout=40, headers={"User-Agent": "signal-sonic"}); r.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f: f.write(r.content); fn = f.name
            y, _ = librosa.load(fn, sr=16000, mono=True, duration=120); L = len(y) / 16000
            lowfn = fn + ".64.mp3"; subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-i", fn, "-b:a", "64k", lowfn], check=True)
            ylow, _ = librosa.load(lowfn, sr=16000, mono=True, duration=120); os.remove(fn); os.remove(lowfn)
            mid = max(0, (L - 60) / 2) / L
            X = {"clean": patches(y), "clip 60 s": patches(y, mid, mid + min(1, 60 / L)),
                 "first 30 s": patches(y, 0, 30 / L), "middle 30 s": patches(y, max(0, (L - 30) / 2) / L, min(1, (L + 30) / 2 / L)), "last 30 s": patches(y, 1 - 30 / L, 1),
                 "phone": patches(phone(y)), "64 kbps": patches(ylow), "12 dB quieter": patches(y * 10 ** (-12 / 20))}
            if X["clean"] is None: continue
            out.append({"id": tid, "y": si[scene], "p": {k: (read(x) if x is not None else None) for k, x in X.items()}})
        except Exception as e:
            print("skip", tid, type(e).__name__)
    return out


@app.local_entrypoint()
def main(manifest_path: str, stage: str = "all", epochs: int = 20):
    man = json.load(open(manifest_path))
    if stage in ("all", "extract"):
        batches = [[(m["id"], m["url"]) for m in man[i:i + 40]] for i in range(0, len(man), 40)]
        import collections; tot = collections.Counter()
        for w_ in extract.map(batches): tot.update(w_ if isinstance(w_, dict) else {"new": w_})
        print("::notice title=extraction::" + json.dumps({"of": len(man), **dict(tot)}))
    if stage == "robust":
        import random
        items = [m for m in man if m.get("scene")]; arts = sorted({m["artist"] for m in items}); random.Random(0).shuffle(arts); hold = set(arts[:len(arts) // 5])
        te = [m for m in items if m["artist"] in hold]; random.Random(7).shuffle(te); te = te[:2000]
        rows = [r for b_ in robust_batch.map([[(m["id"], m["url"], m["scene"]) for m in te[i:i + 40]] for i in range(0, len(te), 40)]) for r in b_]
        res = {"records": len(rows)}
        for k in CONDS:
            ok = [r for r in rows if r["p"].get(k) is not None]
            res[k] = round(sum(r["p"][k] == r["y"] for r in ok) / max(1, len(ok)), 3)
            if k != "clean": res[k + " same call"] = round(sum(r["p"][k] == r["p"]["clean"] for r in ok) / max(1, len(ok)), 3)
        print("::notice title=robustness::" + json.dumps(res))
    if stage == "calibrate":
        res = calibrate.remote(man); print("::notice title=calibration::" + json.dumps(res))
    if stage == "split":
        res = split.remote(man); json.dump(res, open("split.json", "w")); print("split:", len(res["train"]), "train,", len(res["test"]), "test")
    if stage in ("all", "train", "traincal"):
        res = train.remote(man, epochs); print("::notice title=embedding pilot::" + json.dumps(res))
    if stage == "traincal":
        res = calibrate.remote(man); print("::notice title=calibration::" + json.dumps(res))
