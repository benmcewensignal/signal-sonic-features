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


@app.function(image=image, gpu="H100", volumes={"/data": vol}, timeout=14400, memory=65536)
def train(manifest, epochs: int = 20, aug: bool = False, tag: str = ""):
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
    import librosa
    fq = torch.tensor(librosa.mel_frequencies(n_mels=96, fmax=8000), device=dev)
    phone_band = ((fq < 200) | (fq > 6000)).float()[None, :, None]
    def degrade(xn):
        """On half of each batch, what users' conditions do to a spectrogram: volume changes, a phone's
        frequency range, a room's echo, background noise. Applied to the linear spectrum, then re-logged."""
        x = xn * sd + mu; M = (torch.exp(x) - 1) / 1000; b = x.shape[0]; pick = torch.rand(b, device=dev) < 0.5
        g = 10 ** (torch.empty(b, 1, 1, device=dev).uniform_(-15, 15) / 10); M = torch.where(pick[:, None, None], M * g, M)
        ph = pick & (torch.rand(b, device=dev) < 0.5); M = torch.where(ph[:, None, None], M * (1 - 0.97 * phone_band), M)
        rv = pick & (torch.rand(b, device=dev) < 0.4)
        if rv.any():
            k = torch.exp(-torch.arange(12, device=dev, dtype=torch.float32) / 3.0); k = (k / k.sum()).view(1, 1, 1, -1)
            sm = torch.nn.functional.conv2d(torch.nn.functional.pad(M.unsqueeze(1), (11, 0, 0, 0)), k).squeeze(1)
            M = torch.where(rv[:, None, None], 0.6 * M + 0.4 * sm, M)
        nz = pick & (torch.rand(b, device=dev) < 0.4); floor = M.mean((1, 2), keepdim=True) * torch.empty(b, 1, 1, device=dev).uniform_(0.01, 0.1)
        M = torch.where(nz[:, None, None], M + floor * torch.rand_like(M), M)
        return (torch.log1p(1000 * M) - mu) / sd
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(len(P))
        for i in range(0, len(P), 256):
            idx = perm[i:i + 256]; x = ((P[idx].float() - mu) / sd).to(dev); y = Y[idx].to(dev)
            f = torch.randint(0, 80, (1,)).item(); x[:, f:f + 12, :] = 0   # a little frequency masking
            if aug: x = degrade(x)
            opt.zero_grad(); l = lossf(net(x), y); l.backward(); opt.step(); sched.step()
        if ep % 5 == 4 or ep == epochs - 1: a1, a3 = evaluate(); hist.append([ep + 1, round(a1, 3), round(a3, 3)]); print("epoch", ep + 1, a1, a3, flush=True)
    torch.save({"state": net.state_dict(), "scenes": scenes, "mu": mu, "sd": sd, "augmented": aug}, f"/data/embed_{tag + '_' if tag else ''}{len(items)}.pt"); vol.commit()
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


@app.function(image=image, gpu="H100", volumes={"/data": vol}, timeout=3600, memory=16384)
def calibrate(manifest, tag: str = "", ckpt: str = ""):
    """Temperature and reliability for the latest trained model, on its held-out records; saved as embed_live.pt."""
    import os, glob, random, numpy as np, torch
    vol.reload()
    if ckpt: ck = f"/data/{ckpt}"   # named explicitly: on 26 September an unnamed pick calibrated the wrong model
    else:
        pat = f"/data/embed_{tag}_*.pt" if tag else "/data/embed_[0-9]*.pt"
        ck = sorted(glob.glob(pat), key=lambda f: int(f.split("_")[-1].split(".")[0]) if f.split("_")[-1].split(".")[0].isdigit() else 0)[-1]
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
    torch.save({k: (v if k != "state" else {kk: vv.cpu() for kk, vv in v.items()}) for k, v in C.items()}, f"/data/embed_live{'_' + tag if tag else ''}.pt"); vol.commit()
    return {"temperature": T, "tiers": tiers, "first": round(first, 3), "top3": round(top3, 3), "records": len(te), "from": os.path.basename(ck)}


CONDS = ["clean", "clip 60 s", "first 30 s", "middle 30 s", "last 30 s", "phone", "64 kbps", "12 dB quieter"]


@app.function(image=image, volumes={"/data": vol}, timeout=1800, cpu=2, retries=1, max_containers=12)
def robust_batch(batch, model: str = "embed_live.pt"):
    """Each held-out record downloaded again, degraded eight ways, and read by the live model."""
    import os, subprocess, tempfile, numpy as np, librosa, requests, torch
    vol.reload(); C = torch.load(f"/data/{model}", map_location="cpu"); net = make_net(len(C["scenes"])); net.load_state_dict(C["state"]); net.eval()
    si = {s_: i for i, s_ in enumerate(C["scenes"])}; out = []; rng = np.random.default_rng(0)
    def patches(y, a=0.0, b=1.0):
        M = np.log1p(1000 * librosa.feature.melspectrogram(y=y, sr=16000, n_fft=512, hop_length=256, n_mels=96)).astype(np.float32)
        lo, hi = int(a * M.shape[1]), int(b * M.shape[1]); M = M[:, lo:hi]
        if M.shape[1] < W: return None
        return np.stack([M[:, s_:s_ + W] for s_ in np.linspace(0, M.shape[1] - W, 8).astype(int)])
    T = C.get("temperature", 1.0)
    def read(x):   # the call and how sure it is, after the model's calibration temperature
        with torch.no_grad(): p = torch.softmax(net((torch.from_numpy(x) - C["mu"]) / C["sd"]), 1).mean(0).numpy()
        q = np.power(np.clip(p, 1e-9, 1), 1 / T); q = q / q.sum(); return [int(q.argmax()), float(q.max())]
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


@app.function(image=image, volumes={"/data": vol}, timeout=600)
def restore_live():
    """Put back the live model's calibration from its first, clean run (26 September, 07:00): the split it
    was calibrated on can no longer be reproduced, so any recalibration now leaks training artists."""
    import torch
    vol.reload(); C = torch.load("/data/embed_live.pt", map_location="cpu")
    C.update({"temperature": 0.75, "tiers": [[0.6, 1.01, 0.848], [0.4, 0.6, 0.487], [0.0, 0.4, 0.295]], "held_out_accuracy": 0.607, "held_out_top3": 0.82, "held_out_records": 9813, "restored": "2026-09-26"})
    C.pop("scene_tiers", None)
    torch.save(C, "/data/embed_live.pt"); vol.commit()
    return {"scenes": len(C["scenes"]), "temperature": C["temperature"], "tiers": C["tiers"]}


@app.function(image=image, gpu="H100", volumes={"/data": vol}, timeout=3600, memory=32768)
def confusion(manifest, tag: str = "aug"):
    """Genre by genre on the held-out records: how often each is named right, and what it is mistaken for."""
    import os, random, numpy as np, torch
    from concurrent.futures import ThreadPoolExecutor
    vol.reload(); C = torch.load(f"/data/embed_live_{tag}.pt", map_location="cuda"); sc = C["scenes"]; si = {s_: i for i, s_ in enumerate(sc)}
    net = make_net(len(sc)).cuda(); net.load_state_dict(C["state"]); net.eval()
    items = [m for m in manifest if m.get("scene") in si and os.path.exists(f"/data/patches/{m['id'].replace(':', '_')}.npy")]
    arts = sorted({m["artist"] for m in items}); random.Random(0).shuffle(arts); hold = set(arts[:len(arts) // 5])
    te = [m for m in items if m["artist"] in hold]; y = np.array([si[m["scene"]] for m in te]); P = []
    ld = lambda m: np.load(f"/data/patches/{m['id'].replace(':', '_')}.npy").astype(np.float32)
    with torch.no_grad():
        for i in range(0, len(te), 128):
            with ThreadPoolExecutor(32) as ex: x = np.stack(list(ex.map(ld, te[i:i + 128])))
            b = x.shape[0]; x = ((torch.from_numpy(x).cuda() - float(C["mu"])) / float(C["sd"])).reshape(-1, 96, W)
            P.append(torch.softmax(net(x), 1).reshape(b, 8, -1).mean(1).cpu().numpy())
    pred = np.concatenate(P).argmax(1); n = len(sc); M = np.zeros((n, n))
    for a_, b_ in zip(y, pred): M[a_, b_] += 1
    R = M / np.maximum(1, M.sum(1, keepdims=True))
    out = {sc[i]: {"n": int(M[i].sum()), "right": round(float(R[i, i]), 3), "mistaken_for": [[sc[j], round(float(R[i, j]), 3)] for j in np.argsort(-R[i]) if j != i][:4]} for i in range(n)}
    return {"tag": tag, "records": len(te), "genres": out}


@app.function(image=image, volumes={"/data": vol}, timeout=600)
def promote(tag: str):
    """Make a tagged, calibrated model the live one; the previous live file is kept as embed_live_prev.pt."""
    import shutil, os
    vol.reload(); src = f"/data/embed_live_{tag}.pt"
    if not os.path.exists(src): return {"error": "no " + src}
    if os.path.exists("/data/embed_live.pt"): shutil.copyfile("/data/embed_live.pt", "/data/embed_live_prev.pt")
    shutil.copyfile(src, "/data/embed_live.pt"); vol.commit(); return {"live": src, "backup": "embed_live_prev.pt"}


@app.function(image=image, volumes={"/data": vol}, timeout=600)
def save_condition_tiers(tiers, files):
    import torch
    vol.reload(); done = []
    for f in files:
        try:
            C = torch.load(f"/data/{f}", map_location="cpu"); C["condition_tiers"] = tiers; torch.save(C, f"/data/{f}"); done.append(f)
        except Exception as e:
            done.append(f + ": " + type(e).__name__)
    vol.commit(); return done


sep_image = image.pip_install("demucs==4.0.1")


@app.function(image=sep_image, gpu="A10G", volumes={"/data": vol}, timeout=3600, retries=1, max_containers=8)
def partextract(batch):
    """Separate each record into drums, bass, melody and voice, and save eight 3-second log-mel patches per part
    (4 x 8 x 96 x 188, float16), so the learned model can hear the parts as well as the mix."""
    import os, io, tempfile, numpy as np, requests, librosa, torch, collections
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    vol.reload(); os.makedirs("/data/partpatches", exist_ok=True); why = collections.Counter()
    model = get_model("htdemucs").cuda().eval(); ORDER = ["drums", "bass", "other", "vocals"]; idx = [model.sources.index(k) for k in ORDER]
    for tid, url in batch:
        out = f"/data/partpatches/{tid.replace(':', '_')}.npy"
        if os.path.exists(out): why["already"] += 1; continue
        try:
            r = requests.get(url, timeout=40, headers={"User-Agent": "signal-sonic"})
            if r.status_code != 200: why[f"http {r.status_code}"] += 1; continue
            with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
                f.write(r.content); f.flush(); y, sr = librosa.load(f.name, sr=model.samplerate, mono=False, duration=120)
            if y.ndim == 1: y = np.stack([y, y])
            with torch.no_grad(): S = apply_model(model, torch.from_numpy(y[None]).float().cuda(), split=True, overlap=0.1)[0].cpu().numpy()
            P = []
            for k in idx:
                m = librosa.resample(S[k].mean(0), orig_sr=model.samplerate, target_sr=16000)
                M = np.log1p(1000 * librosa.feature.melspectrogram(y=m, sr=16000, n_fft=512, hop_length=256, n_mels=96)).astype(np.float16)
                if M.shape[1] < W: raise ValueError("too short")
                P.append(np.stack([M[:, s_:s_ + W] for s_ in np.linspace(0, M.shape[1] - W, 8).astype(int)]))
            np.save(out, np.stack(P)); why["new"] += 1
        except Exception as e:
            why[type(e).__name__] += 1
    vol.commit(); return dict(why)


@app.function(image=image, gpu="H100", volumes={"/data": vol}, timeout=3600)
def fairtest(manifest, ckpts, cutoff_ts: float):
    """Compare models only on records neither trained on. The split shuffles the whole artist list, so adding
    artists reshuffles it: records held out now were often training records before. Held out here means held
    out in yesterday's split (records whose patches predate the cutoff) and in today's, plus artists new today."""
    import os, random, numpy as np, torch
    vol.reload()
    pth = lambda m: f"/data/patches/{m['id'].replace(':', '_')}.npy"
    items = [m for m in manifest if m.get("scene") and os.path.exists(pth(m))]
    old = [m for m in items if os.path.getmtime(pth(m)) < cutoff_ts]
    def held(its):
        arts = sorted({m["artist"] for m in its}); random.Random(0).shuffle(arts); return set(arts[:len(arts) // 5])
    old_arts = {m["artist"] for m in old}; old_hold = held(old); new_hold = held(items)
    fair = [m for m in items if m["artist"] in new_hold and (m["artist"] in old_hold or m["artist"] not in old_arts)]
    out = {"records": len(fair), "artists": len({m["artist"] for m in fair}), "old_records": len(old)}
    from concurrent.futures import ThreadPoolExecutor
    ld = lambda m: np.load(pth(m)).astype(np.float32)
    for ck in ckpts:
        C = torch.load(f"/data/{ck}", map_location="cuda"); scenes = C["scenes"]; si = {x: i for i, x in enumerate(scenes)}
        net = make_net(len(scenes)).cuda(); net.load_state_dict(C["state"]); net.eval()
        te = [m for m in fair if m["scene"] in si]; y = np.array([si[m["scene"]] for m in te]); P = []
        with torch.no_grad():
            for i in range(0, len(te), 64):
                with ThreadPoolExecutor(32) as ex: x = np.stack(list(ex.map(ld, te[i:i + 64])))
                b = x.shape[0]; x = ((torch.from_numpy(x) - C["mu"]) / C["sd"]).cuda().reshape(-1, 96, W)
                P.append(torch.softmax(net(x), 1).reshape(b, 8, -1).mean(1).cpu().numpy())
        P = np.concatenate(P); top = np.argsort(-P, 1)
        old_only = np.array([m["artist"] in old_hold for m in te]); new_only = ~old_only
        r = lambda mask: {"n": int(mask.sum()), "first": round(float((top[mask, 0] == y[mask]).mean()), 3), "top3": round(float(np.mean([y[i] in top[i, :3] for i in np.where(mask)[0]])), 3)} if mask.sum() else None
        out[ck] = {"all": r(np.ones(len(te), bool)), "artists held out in both splits": r(old_only), "artists new today": r(new_only)}
    return out


dsep_image = image.pip_install("demucs==4.0.1", "audio-separator[gpu]", "soundfile")
TEMPO_MEDIANS = {"uk-garage-speed-garage": 134.7, "uk-funky-gqom": 139.9, "afro-house": 123.0, "amapiano": 112.8, "140-deep-dubstep-grime": 139.9, "breaks-breakbeat-uk-bass": 135.3, "tech-house": 127.0, "techno-peak-time": 135.3, "techno-raw-deep-hypnotic": 136.7, "house": 126.8, "deep-house": 123.1, "melodic-house-techno": 125.3, "drum-and-bass": 173.4, "hard-techno": 154.4, "bass-house": 128.3, "trance-main-floor": 138.2, "progressive-house": 123.7, "psy-trance": 143.9, "indie-dance": 125.2, "organic-house": 122.3, "african": 136.5, "ambient-experimental": 144.8, "brazilian-funk": 129.7, "downtempo": 135.6, "dubstep": 142.6, "electro": 129.8, "electronica": 135.0, "funky-house": 126.0, "hard-dance-hardcore": 154.2, "jackin-house": 125.2, "latin-electronic": 129.8, "mainstage": 128.4, "minimal-deep-tech": 126.8, "nu-disco-disco": 123.1, "trance-raw-deep-hypnotic": 132.8, "trap-future-bass": 143.9}


@app.function(image=dsep_image, gpu="A10G", volumes={"/data": vol}, timeout=3600, retries=0, max_containers=4)
def drumpilot(batch, tempos=None):
    """Finer separation pilot: each record's drums (htdemucs) split into kick, snare, toms, hi-hats, ride and crash
    (MDX23C DrumSep), then checked: does the kick sit on the beat where it should, do the hats sit off it, do they leak."""
    import os, tempfile, time, numpy as np, requests, librosa, torch, soundfile as sf
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from audio_separator.separator import Separator
    work = tempfile.mkdtemp(); rows = []
    dm = get_model("htdemucs").cuda().eval(); di = dm.sources.index("drums")
    sep = Separator(output_dir=work, output_format="WAV")
    try:
        sep.load_model(model_filename="MDX23C-DrumSep-aufr33-jarredou.ckpt")
    except Exception as e:
        return {"error": f"drum model: {type(e).__name__}: {e}"[:400]}
    for tid, url, scene in batch:
        try:
            t0 = time.time(); mp3 = os.path.join(work, "a.mp3"); open(mp3, "wb").write(requests.get(url, timeout=40).content)
            y, sr = librosa.load(mp3, sr=dm.samplerate, mono=False, offset=20, duration=60)
            if y.ndim == 1: y = np.stack([y, y])
            with torch.no_grad(): S = apply_model(dm, torch.from_numpy(y[None]).float().cuda(), split=True)[0].cpu().numpy()
            dpath = os.path.join(work, "drums.wav"); sf.write(dpath, S[di].T, dm.samplerate)
            files = sep.separate(dpath); t1 = time.time()
            parts = {}
            for f in files:
                fp = f if os.path.isabs(f) else os.path.join(work, f); fn = os.path.basename(fp).lower()
                for k in ("kick", "snare", "toms", "hh", "ride", "crash"):
                    if f"({k})" in fn: parts[k] = fp
            dmono = librosa.to_mono(S[di]); dmono = librosa.resample(dmono, orig_sr=dm.samplerate, target_sr=22050)
            oe = librosa.onset.onset_strength(y=dmono, sr=22050); f_ = getattr(librosa.feature, "rhythm", None)
            T = float(np.atleast_1d((f_.tempo if f_ else librosa.beat.tempo)(onset_envelope=oe, sr=22050))[0])
            med = (tempos or {}).get(scene)
            if med and T:   # fold a half- or double-speed reading into the scene's range
                while T < med * 0.75: T *= 2
                while T > med * 1.5: T /= 2
            P = 60.0 / T if T else None
            def load(k):
                if k not in parts: return None
                v, _ = librosa.load(parts[k], sr=22050, mono=True); return v
            def lock(on, period, mult=1):   # how tightly onsets fall at the same point of each beat (1 = always, 0 = anywhere)
                if len(on) < 8 or not period: return None
                ph = (np.asarray(on) % period) / period; return float(abs(np.mean(np.exp(2j * np.pi * mult * ph))))
            rec = {"id": tid, "scene": scene, "secs": round(t1 - t0, 1), "found": sorted(parts), "tempo": round(T, 1) if T else None}
            lev = {}
            for k in parts:
                v = load(k); lev[k] = float(np.sqrt(np.mean(v ** 2))) if v is not None else 0.0
            tot = sum(lev.values()) or 1; rec["share"] = {k: round(v / tot, 3) for k, v in lev.items()}
            kick = load("kick"); hh = load("hh"); dur = len(dmono) / 22050
            if kick is not None and P:
                ko = librosa.onset.onset_detect(y=kick, sr=22050, units="time", backtrack=True)
                rec["kick_lock"] = lock(ko, P); rec["kicks_per_beat"] = round(len(ko) / (dur / P), 2)
                kph = float(np.angle(np.mean(np.exp(2j * np.pi * (np.asarray(ko) % P) / P))) / (2 * np.pi) % 1) if len(ko) >= 8 else None
            else: kph = None
            if hh is not None and P and kph is not None:
                ho = librosa.onset.onset_detect(y=hh, sr=22050, units="time", backtrack=True)
                if len(ho) >= 8:
                    rel = ((np.asarray(ho) % P) / P - kph) % 1   # hat positions within the beat, measured from the kick
                    rec["hats_offbeat"] = round(float(np.mean(np.abs(rel - 0.5) < 0.1)), 3); rec["hats_with_kick"] = round(float(np.mean((rel < 0.1) | (rel > 0.9))), 3)
            if kick is not None and hh is not None:
                a_ = librosa.onset.onset_strength(y=kick, sr=22050); b_ = librosa.onset.onset_strength(y=hh, sr=22050); n_ = min(len(a_), len(b_))
                rec["kick_hat_overlap"] = round(float(np.corrcoef(a_[:n_], b_[:n_])[0, 1]), 3)
            keep = f"/data/drumpilot/{tid.replace(':', '_')}"; os.makedirs(keep, exist_ok=True)   # kept, so later tests need no separation
            for k in ("kick", "snare", "hh"):
                if k in parts: os.system(f'ffmpeg -y -loglevel quiet -i "{parts[k]}" -ac 1 -b:a 64k "{keep}/{k}.mp3"')
            for f in files:
                try: os.remove(f if os.path.isabs(f) else os.path.join(work, f))
                except Exception: pass
            rows.append(rec)
        except Exception as e:
            rows.append({"id": tid, "scene": scene, "error": type(e).__name__ + ": " + str(e)[:80]})
    vol.commit(); return rows


@app.function(image=dsep_image, volumes={"/data": vol}, timeout=1800, cpu=4)
def drumcheck(items, tempos):
    """On the kept kick and hat parts: each record's exact tempo found from its kicks (a fine search within 8% of the
    scene's tempo, never half or double), then beat-locking and hats relative to the kick; and the same search on
    random onsets, so a fine search cannot flatter the result."""
    import os, numpy as np, librosa
    rng = np.random.default_rng(0); rows = []
    def best_lock(on, med):
        best = (0.0, None)
        for T in np.linspace(med * 0.92, med * 1.08, 641):
            P = 60.0 / T; L = float(abs(np.mean(np.exp(2j * np.pi * (on % P) / P))))
            if L > best[0]: best = (L, T)
        return best
    for tid, scene in items:
        d = f"/data/drumpilot/{tid.replace(':', '_')}"
        if not os.path.exists(d + "/kick.mp3"): continue
        try:
            k, _ = librosa.load(d + "/kick.mp3", sr=22050, mono=True); ko = librosa.onset.onset_detect(y=k, sr=22050, units="time", backtrack=True)
            if len(ko) < 16: continue
            med = tempos.get(scene) or 125.0; L, T = best_lock(ko, med); P = 60.0 / T
            Lr, _ = best_lock(np.sort(rng.uniform(0, len(k) / 22050, len(ko))), med)
            kph = float(np.angle(np.mean(np.exp(2j * np.pi * (ko % P) / P))) / (2 * np.pi) % 1)
            rec = {"scene": scene, "kick_lock": round(L, 3), "random_lock": round(Lr, 3), "tempo": round(T, 2), "kicks_per_beat": round(len(ko) / ((len(k) / 22050) / P), 2)}
            if os.path.exists(d + "/hh.mp3"):
                h, _ = librosa.load(d + "/hh.mp3", sr=22050, mono=True); ho = librosa.onset.onset_detect(y=h, sr=22050, units="time", backtrack=True)
                if len(ho) >= 16:
                    rel = ((ho % P) / P - kph) % 1
                    rec["hats_offbeat"] = round(float(np.mean(np.abs(rel - 0.5) < 0.1)), 3); rec["hats_on_16ths"] = round(float(abs(np.mean(np.exp(2j * np.pi * 4 * rel)))), 3)
            rows.append(rec)
        except Exception:
            pass
    return rows


@app.local_entrypoint()
def main(manifest_path: str, stage: str = "all", epochs: int = 20, aug: int = 0, tag: str = "", ckpt: str = ""):
    man = json.load(open(manifest_path))
    if stage in ("all", "extract"):
        batches = [[(m["id"], m["url"]) for m in man[i:i + 40]] for i in range(0, len(man), 40)]
        import collections; tot = collections.Counter()
        for w_ in extract.map(batches): tot.update(w_ if isinstance(w_, dict) else {"new": w_})
        print("::notice title=extraction::" + json.dumps({"of": len(man), **dict(tot)}))
    if stage == "calcond":
        # reliability measured in the conditions readings actually run in: the full reading's 60-second clip, and a phone
        import random
        test_ids = {t for t, _ in split.remote(man)["test"]}
        te = [m for m in man if m.get("scene") and m["id"] in test_ids]; random.Random(11).shuffle(te); te = te[:3000]
        rows = [r for b_ in robust_batch.starmap([([(m["id"], m["url"], m["scene"]) for m in te[i:i + 40]], "embed_live.pt") for i in range(0, len(te), 40)]) for r in b_]
        BINS = ((0.6, 1.01), (0.4, 0.6), (0.0, 0.4)); tiers = {}
        for k in ("clip 60 s", "phone", "clean"):
            ok = [r["p"][k] + [r["y"]] for r in rows if r["p"].get(k)]
            tiers[k] = [[lo, hi, (round(sum(1 for p_, c_, y_ in ok if lo <= c_ < hi and p_ == y_) / n_, 3) if n_ >= 30 else None), n_] for lo, hi in BINS for n_ in [sum(1 for p_, c_, y_ in ok if lo <= c_ < hi)]]
        print("::notice title=condition tiers::" + json.dumps({"records": len(rows), **tiers}))
        print("::notice title=saved::" + json.dumps(save_condition_tiers.remote(tiers, ["embed_live.pt", "embed_live_aug.pt"])))
    if stage == "robust":
        import random
        test_ids = {t for t, _ in split.remote(man)["test"]}   # the exact held-out split training used
        te = [m for m in man if m.get("scene") and m["id"] in test_ids]; random.Random(7).shuffle(te); te = te[:2000]
        model = f"embed_live{'_' + tag if tag else ''}.pt"
        rows = [r for b_ in robust_batch.starmap([([(m["id"], m["url"], m["scene"]) for m in te[i:i + 40]], model) for i in range(0, len(te), 40)]) for r in b_]
        res = {"records": len(rows)}
        for k in CONDS:
            ok = [r for r in rows if r["p"].get(k) is not None]
            res[k] = round(sum(r["p"][k][0] == r["y"] for r in ok) / max(1, len(ok)), 3)
            if k != "clean": res[k + " same call"] = round(sum(r["p"][k][0] == r["p"]["clean"][0] for r in ok) / max(1, len(ok)), 3)
        print("::notice title=robustness::" + json.dumps({"model": model, **res}))
    if stage == "partextract":
        import collections; tot = collections.Counter(); todo = [(m["id"], m["url"]) for m in man]
        if epochs and epochs < len(todo): todo = todo[:epochs]   # epochs doubles as a cap for a trial run
        for w_ in partextract.map([todo[i:i + 20] for i in range(0, len(todo), 20)]): tot.update(w_)
        print("::notice title=part extraction::" + json.dumps({"of": len(todo), **dict(tot)}))
    if stage == "fairtest":
        import datetime
        cut = datetime.datetime(2026, 9, 27, 11, 0, tzinfo=datetime.timezone.utc).timestamp()   # today's extraction began at 11:45
        res = fairtest.remote(man, ["embed_live.pt", ckpt or "embed_aug122k_104842.pt"], cut)
        print("::notice title=fair test::" + json.dumps(res))
    if stage == "drumcheck":
        import random, collections, statistics as st_
        want = ["techno-peak-time", "tech-house", "house", "deep-house", "hard-techno", "drum-and-bass", "breaks-breakbeat-uk-bass", "uk-garage-speed-garage", "140-deep-dubstep-grime", "amapiano"]
        pool = [m for m in man if m.get("scene") in want]; random.Random(1).shuffle(pool); per = collections.Counter(); pick = []
        for m in pool:
            if per[m["scene"]] < 20: pick.append((m["id"], m["scene"])); per[m["scene"]] += 1
        rows = [r for res in drumcheck.map([pick[i:i + 20] for i in range(0, len(pick), 20)], kwargs={"tempos": TEMPO_MEDIANS}) for r in res]
        med = lambda xs: round(st_.median(xs), 2) if xs else None
        by = {sc: {"n": len([r for r in rows if r["scene"] == sc]), "kick lock": med([r["kick_lock"] for r in rows if r["scene"] == sc]),
                   "random": med([r["random_lock"] for r in rows if r["scene"] == sc]), "kicks per beat": med([r["kicks_per_beat"] for r in rows if r["scene"] == sc]),
                   "hats off beat": med([r["hats_offbeat"] for r in rows if r["scene"] == sc and "hats_offbeat" in r]),
                   "hats on 16ths": med([r["hats_on_16ths"] for r in rows if r["scene"] == sc and "hats_on_16ths" in r])} for sc in want}
        print("::notice title=drum check::" + json.dumps({"records": len(rows), "by scene": by}))
    if stage == "drumpilot":
        import random, collections, statistics as st_
        want = ["techno-peak-time", "tech-house", "house", "deep-house", "hard-techno", "drum-and-bass", "breaks-breakbeat-uk-bass", "uk-garage-speed-garage", "140-deep-dubstep-grime", "amapiano"]
        pool = [m for m in man if m.get("scene") in want]; random.Random(1).shuffle(pool); per = collections.Counter(); pick = []
        for m in pool:
            if per[m["scene"]] < 20: pick.append((m["id"], m["url"], m["scene"])); per[m["scene"]] += 1
        rows = []
        TMP = TEMPO_MEDIANS
        for res in drumpilot.map([pick[i:i + 25] for i in range(0, len(pick), 25)], kwargs={"tempos": TMP}):
            if isinstance(res, dict): print("::notice title=drum pilot::" + json.dumps(res)); break
            rows += res
        ok = [r for r in rows if "error" not in r]; errs = collections.Counter(r["error"][:50] for r in rows if "error" in r)
        med = lambda xs: round(st_.median(xs), 2) if xs else None
        by = {}
        for sc in want:
            R = [r for r in ok if r["scene"] == sc]
            by[sc] = {"n": len(R), "tempo": med([r["tempo"] for r in R if r.get("tempo")]), "kick lock": med([r["kick_lock"] for r in R if r.get("kick_lock") is not None]),
                      "kicks per beat": med([r["kicks_per_beat"] for r in R if r.get("kicks_per_beat") is not None]),
                      "hats off beat": med([r["hats_offbeat"] for r in R if r.get("hats_offbeat") is not None]),
                      "hats with kick": med([r["hats_with_kick"] for r in R if r.get("hats_with_kick") is not None]),
                      "kick-hat overlap": med([r["kick_hat_overlap"] for r in R if r.get("kick_hat_overlap") is not None])}
        shares = {k: med([r["share"].get(k, 0) for r in ok]) for k in ("kick", "snare", "toms", "hh", "ride", "crash")}
        print("::notice title=drum pilot::" + json.dumps({"records": len(rows), "separated": len(ok), "errors": dict(errs.most_common(3)),
              "seconds per record": med([r["secs"] for r in ok]), "median energy shares": shares}))
        print("::notice title=drum pilot by scene::" + json.dumps(by))
    if stage == "promote":
        print("::notice title=promoted::" + json.dumps(promote.remote(tag)))
    if stage == "confusion":
        res = confusion.remote(man, tag or "aug")
        # notes are cut at 4,096 characters: one compact line per genre, in several notes
        lines = [f"{g}|{v['n']}|{v['right']}|" + ";".join(f"{x[0]}={x[1]}" for x in v["mistaken_for"]) for g, v in res["genres"].items()]
        chunk, k = [], 0
        for ln in lines + ["END"]:
            if sum(len(x) + 1 for x in chunk) + len(ln) > 3500 or ln == "END":
                k += 1; print(f"::notice title=confusion-{k}::" + " ".join(chunk)); chunk = []
            if ln != "END": chunk.append(ln)
    if stage == "restore":
        print("::notice title=restored::" + json.dumps(restore_live.remote()))
    if stage == "calibrate":
        res = calibrate.remote(man, tag, ckpt); print("::notice title=calibration::" + json.dumps({"tag": tag, **res}))
    if stage == "split":
        res = split.remote(man); json.dump(res, open("split.json", "w")); print("split:", len(res["train"]), "train,", len(res["test"]), "test")
    if stage in ("all", "train", "traincal"):
        res = train.remote(man, epochs, bool(aug), tag); print("::notice title=embedding pilot::" + json.dumps({"tag": tag, "augmented": bool(aug), **res}))
    if stage == "traincal":
        res = calibrate.remote(man, tag, ckpt); print("::notice title=calibration::" + json.dumps({"tag": tag, **res}))
