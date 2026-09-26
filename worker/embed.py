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
    if _M: return _M
    import torch
    fp = "/embed/embed_live.pt"
    if not os.path.exists(fp):
        try:
            import modal; modal.Volume.from_name("sonic-embed").reload()
        except Exception:
            pass
    if not os.path.exists(fp): return None
    C = torch.load(fp, map_location="cpu"); net = make_net(len(C["scenes"])); net.load_state_dict(C["state"]); net.eval(); C["net"] = net; _M = C
    return _M

def learned_call(wav_path):
    C = _load()
    if not C: return None
    import librosa, torch
    y, _ = librosa.load(wav_path, sr=16000, mono=True, duration=120)
    M = np.log1p(1000 * librosa.feature.melspectrogram(y=y, sr=16000, n_fft=512, hop_length=256, n_mels=96)).astype(np.float32)
    if M.shape[1] < 2 * W: return None
    x = np.stack([M[:, s:s + W] for s in np.linspace(0, M.shape[1] - W, 8).astype(int)])
    with torch.no_grad():
        p = torch.softmax(C["net"]((torch.from_numpy(x) - C["mu"]) / C["sd"]), 1).mean(0).numpy()
    q = np.power(np.clip(p, 1e-9, 1), 1 / C.get("temperature", 1.0)); q = q / q.sum(); order = np.argsort(-q); top = C["scenes"][order[0]]; conf = float(q[order[0]])
    rel, basis = None, "overall"
    for lo, hi, r in (C.get("scene_tiers") or {}).get(top, []):
        if lo <= conf < hi and r is not None: rel, basis = r, "scene"
    if rel is None:
        for lo, hi, r in C.get("tiers") or []:
            if lo <= conf < hi and r is not None: rel = r
    return {"scenes": [[C["scenes"][i], round(float(q[i]), 3)] for i in order[:5]], "confidence": round(conf, 3), "right_at_this_confidence": rel, "reliability_basis": basis,
            "inputs": "the whole mix, heard by the learned model", "model": {"trained_on": C.get("trained_on"), "built": C.get("built"), "held_out_accuracy": C.get("held_out_accuracy")}}
