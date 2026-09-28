"""The learned model in the worker: the scene call from spectrogram patches of the clip, calibrated.
The model and its calibration live on the Modal volume sonic-embed as embed_live.pt."""
import os, numpy as np
_M = None
W = 188

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


def _load():
    """Load once found; until then look again on every reading, refreshing the volume, since a warm
    container does not see files written to the volume after it started."""
    global _M
    if _M:
        try:
            if os.path.getmtime("/embed/embed_live.pt") == _M.get("_mtime"): return _M   # reload when the live file changes
        except Exception:
            return _M
    import torch
    fp = "/embed/embed_live.pt"
    if not os.path.exists(fp):
        try:
            import modal; modal.Volume.from_name("sonic-embed").reload()
        except Exception:
            pass
    if not os.path.exists(fp): return None
    C = torch.load(fp, map_location="cpu"); net = make_net(len(C["scenes"])); net.load_state_dict(C["state"]); net.eval(); C["net"] = net; C["_mtime"] = os.path.getmtime(fp); _M = C
    return _M

_P = None
def _proj():
    """The 16-direction projection ear_walk saved beside the model (mean, directions, scale), so a clip lands in the
    same space as the catalogue's ear in dj-index. Reloaded when the file changes; None until it exists."""
    global _P
    fp = "/embed/ear_proj.npz"
    try:
        if _P and os.path.getmtime(fp) == _P.get("_mtime"): return _P
    except Exception:
        return _P
    if not os.path.exists(fp):
        try:
            import modal; modal.Volume.from_name("sonic-embed").reload()
        except Exception:
            pass
    if not os.path.exists(fp): return None
    Z = np.load(fp, allow_pickle=True); _P = {"mean": Z["mean"].astype(np.float32), "V": Z["V"].astype(np.float32), "scale": Z["scale"].astype(np.float32),
                                            "built": str(Z["built"]) if "built" in Z else "", "_mtime": os.path.getmtime(fp)}
    return _P

def learned_call(wav_path):
    C = _load()
    if not C: return None
    import librosa, torch
    y, _ = librosa.load(wav_path, sr=16000, mono=True, duration=120)
    M = np.log1p(1000 * librosa.feature.melspectrogram(y=y, sr=16000, n_fft=512, hop_length=256, n_mels=96)).astype(np.float32)
    if M.shape[1] < 2 * W: return None
    x = np.stack([M[:, s:s + W] for s in np.linspace(0, M.shape[1] - W, 8).astype(int)])
    with torch.no_grad():
        xn = (torch.from_numpy(x) - C["mu"]) / C["sd"]
        p = torch.softmax(C["net"](xn), 1).mean(0).numpy()
        e = C["net"].embed(xn).mean(0).numpy(); e = e / (np.linalg.norm(e) + 1e-9)   # the clip's place in the learned ear
    ear = None; P = _proj()
    if P is not None and P["V"].shape[1] == e.shape[0]:
        yv = (e - P["mean"]) @ P["V"].T   # the same 16 directions as the catalogue's ear; the site normalises before comparing
        ear = {"v": [round(float(v), 5) for v in yv], "dims": int(P["V"].shape[0]), "built": P["built"]}
    q = np.power(np.clip(p, 1e-9, 1), 1 / C.get("temperature", 1.0)); q = q / q.sum(); order = np.argsort(-q); top = C["scenes"][order[0]]; conf = float(q[order[0]])
    # the full reading always sends a 60-second clip: its reliability is measured on 60-second clips
    rel, basis = None, "overall"
    for row in (C.get("condition_tiers") or {}).get("clip 60 s", []):
        lo, hi, r = row[0], row[1], row[2]
        if lo <= conf < hi and r is not None: rel, basis = r, "60-second clips"
    if rel is None:
        for lo, hi, r in (C.get("scene_tiers") or {}).get(top, []):
            if lo <= conf < hi and r is not None: rel, basis = r, "scene"
    if rel is None:
        for lo, hi, r in C.get("tiers") or []:
            if lo <= conf < hi and r is not None: rel = r
    return {"scenes": [[C["scenes"][i], round(float(q[i]), 3)] for i in order[:5]], "confidence": round(conf, 3), "right_at_this_confidence": rel, "reliability_basis": basis,
            "inputs": "the whole mix, heard by the learned model", "model": {"trained_on": C.get("trained_on"), "built": C.get("built"), "held_out_accuracy": C.get("held_out_accuracy")},
            "ear": ear}



def learned_from_slices(x, condition="phone"):
    """The learned model's call from spectrogram slices a device computed itself (8 x 96 x 188, the same slices the
    model trained on), with the reliability measured for that condition; and the phone map's drum-led to vocal-led
    score, from the 16 most phone-stable directions of the model's description."""
    C = _load()
    if not C: return None
    import json, os, torch
    x = np.asarray(x, np.float32)
    if x.shape != (8, 96, W): return {"error": f"slices must be 8 x 96 x {W}, got {list(x.shape)}"}
    with torch.no_grad():
        xn = (torch.from_numpy(x) - C["mu"]) / C["sd"]
        p = torch.softmax(C["net"](xn), 1).mean(0).numpy(); e = C["net"].embed(xn).mean(0).numpy()
    q = np.power(np.clip(p, 1e-9, 1), 1 / C.get("temperature", 1.0)); q = q / q.sum(); order = np.argsort(-q); top = C["scenes"][order[0]]; conf = float(q[order[0]])
    rel = None
    for row in (C.get("condition_tiers") or {}).get(condition, []):
        if row[0] <= conf < row[1] and row[2] is not None: rel = row[2]
    out = {"scene_learned": {"scenes": [[C["scenes"][i], round(float(q[i]), 4)] for i in order[:3]], "confidence": round(conf, 4), "reliability": rel, "condition": condition}}
    fp = "/embed/phone_axis.json"
    if os.path.exists(fp):
        A = json.load(open(fp)); f = ((e - np.array(A["mu"])) @ np.array(A["V"]) - np.array(A["fm"])) / np.array(A["fs"]); w = np.array(A["w"])
        out["drum_voice"] = round(float(f @ w[:-1] + w[-1]), 4)
    return out
