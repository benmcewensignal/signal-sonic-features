"""Where a record's kicks sit against Swap's beat grid: the drum part split as Swap splits it, the grid read as Swap
reads it, then every kick attack (30-120 Hz) and every kick-or-snare attack (30-500 Hz) timed in the waveform and placed
against the nearest beat. On a four on the floor the kicks should sit on the beats; an offset common to all of them is
how far the grid is out. The corpus's drum tracker's beats are placed the same way. Annotations only.
  modal run tools/swap_griddiag.py --only tech-house"""
import json, modal

app = modal.App("sonic-swap-griddiag")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("ffmpeg", "libsndfile1")
         .pip_install("numpy<2", "librosa==0.10.2", "soundfile", "demucs==4.0.1", "torch==2.3.1",
                      "torchaudio==2.3.1", "essentia-tensorflow", "fastapi[standard]", "scikit-learn==1.8.0")
         .apt_install("rubberband-cli")
         .add_local_python_source("features")
         .add_local_dir("worker", "/root/worker"))


@app.function(image=image, gpu="T4", cpu=4.0, memory=12288, timeout=900)
def diag(rec):
    import sys, os, tempfile, urllib.request, numpy as np, librosa
    sys.path.insert(0, "/root")
    from features import stems as S
    from worker import swap as SW
    from scipy.signal import butter, sosfiltfilt
    w = tempfile.mkdtemp(); mp3 = os.path.join(w, "r.mp3"); urllib.request.urlretrieve(rec["preview"], mp3)
    wav = os.path.join(w, "r.wav"); SW.decode_to_wav(mp3, wav)
    st = S.separate(wav, os.path.join(w, "sep")); X = {k: SW.read_wav(st[k])[0] for k in SW.PARTS}
    n = min(len(v) for v in X.values()); X = {k: v[:n] for k, v in X.items()}; dur = n / SW.SR
    T0, tbeats = SW.rough_tempo(X["drums"]); a, p, sure = SW.beat_grid(X["drums"], sum(X.values()), dur, T0)
    y22 = librosa.resample(X["drums"].mean(1).astype(np.float32), orig_sr=SW.SR, target_sr=22050)
    out = {"label": rec["scene"] + ": " + rec["name"], "tempo": round(60 / p, 2), "sure": round(sure, 2)}
    for name, hi in (("kick 30-120 Hz", 120), ("kick and snare 30-500 Hz", 500), ("all", None)):
        sig = np.abs(sosfiltfilt(butter(4, hi, "low", fs=22050, output="sos"), y22)) if hi else np.abs(y22)
        hop = 32; e = sig[: len(sig) // hop * hop].reshape(-1, hop).max(1); r = np.maximum(0, np.diff(e, prepend=e[:1]))
        fr = 22050 / hop; pk = SW._peaks(r / (r.max() + 1e-12), w=6, avg=40, delta=0.05, wait=int(0.12 * fr))
        tp = SW._attack_times(y22, pk / fr, low=hi)
        rel = ((tp - a + p / 2) % p) - p / 2   # each attack against its nearest beat, in seconds
        hist, edges = np.histogram(1000 * rel, bins=np.arange(-1000 * p / 2, 1000 * p / 2 + 1, 20))
        top = np.argsort(-hist)[:4]
        out[name] = {"attacks": int(len(tp)), "within 15 ms of a beat": round(float(np.mean(np.abs(rel) <= 0.015)), 3),
                     "busiest 20 ms bins (start ms, count)": [[int(edges[i]), int(hist[i])] for i in top]}
    if len(tbeats):
        rel = ((np.asarray(tbeats) - a + p / 2) % p) - p / 2
        out["tracker beats vs grid (ms) median"] = round(1000 * float(np.median(rel)), 1)
    return out


@app.local_entrypoint()
def main(only: str = ""):
    recs = [r for r in json.load(open("tools/swap-test-records.json")) if only.lower() in (r["scene"] + " " + r["name"]).lower()]
    for o in diag.map(recs, return_exceptions=True):
        print("::notice title=grid diag::" + (json.dumps(o)[:1800] if isinstance(o, dict) else str(o)[:300]))
