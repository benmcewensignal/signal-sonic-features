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


@app.function(image=image, volumes={"/data": vol}, timeout=900, cpu=2, retries=1)
def extract(batch):
    import os, tempfile, numpy as np, librosa, requests
    os.makedirs("/data/patches", exist_ok=True); done = 0
    for tid, url in batch:
        p = f"/data/patches/{tid.replace(':', '_')}.npy"
        if os.path.exists(p): done += 1; continue
        try:
            r = requests.get(url, timeout=40, headers={"User-Agent": "signal-sonic"}); r.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f: f.write(r.content); fn = f.name
            y, _ = librosa.load(fn, sr=16000, mono=True, duration=120); os.remove(fn)
            M = np.log1p(1000 * librosa.feature.melspectrogram(y=y, sr=16000, n_fft=512, hop_length=256, n_mels=96)).astype(np.float16)
            if M.shape[1] < 2 * W: continue
            np.save(p, np.stack([M[:, s:s + W] for s in np.linspace(0, M.shape[1] - W, 8).astype(int)])); done += 1
        except Exception as e:
            print("skip", tid, type(e).__name__)
    vol.commit(); return done


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=14400, memory=40960)
def train(manifest, epochs: int = 20):
    import os, random, numpy as np, torch, torch.nn as nn
    vol.reload()
    items = [m for m in manifest if os.path.exists(f"/data/patches/{m['id'].replace(':', '_')}.npy")]
    scenes = sorted({m["scene"] for m in items}); si = {s: i for i, s in enumerate(scenes)}
    arts = sorted({m["artist"] for m in items}); random.Random(0).shuffle(arts); hold = set(arts[:len(arts) // 5])
    tr = [m for m in items if m["artist"] not in hold]; te = [m for m in items if m["artist"] in hold]
    load = lambda m: np.load(f"/data/patches/{m['id'].replace(':', '_')}.npy")   # kept at half precision; converted per batch
    print('loading', len(tr), 'training and', len(te), 'test records', flush=True)
    Xtr = np.stack([load(m) for m in tr]); ytr = np.array([si[m["scene"]] for m in tr]); Xte = np.stack([load(m) for m in te]); yte = np.array([si[m["scene"]] for m in te])
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
    items = [m for m in manifest if os.path.exists(f"/data/patches/{m['id'].replace(':', '_')}.npy")]
    arts = sorted({m["artist"] for m in items}); random.Random(0).shuffle(arts); hold = set(arts[:len(arts) // 5])
    return {"train": [[m["id"], m["scene"]] for m in items if m["artist"] not in hold], "test": [[m["id"], m["scene"]] for m in items if m["artist"] in hold], "held_artists": sorted(hold)}


@app.local_entrypoint()
def main(manifest_path: str, stage: str = "all"):
    man = json.load(open(manifest_path))
    if stage in ("all", "extract"):
        batches = [[(m["id"], m["url"]) for m in man[i:i + 40]] for i in range(0, len(man), 40)]
        print("extracted", sum(extract.map(batches)), "of", len(man))
    if stage == "split":
        res = split.remote(man); json.dump(res, open("split.json", "w")); print("split:", len(res["train"]), "train,", len(res["test"]), "test")
    if stage in ("all", "train"):
        res = train.remote(man); print("::notice title=embedding pilot::" + json.dumps(res))
