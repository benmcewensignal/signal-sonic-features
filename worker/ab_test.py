"""The before-and-after on a real record: its bass replaced by the licensed loop nearest a target style's bass.
Checks both clips come back, twenty seconds each, differing where they should. Writes data/ab-test.json.   modal run worker/ab_test.py"""
import json, modal
app = modal.App("sonic-ab-test")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg", "libsndfile1")
         .pip_install("numpy<2", "librosa==0.10.2", "soundfile", "demucs==4.0.1", "torch==2.3.1", "torchaudio==2.3.1", "essentia-tensorflow", "scikit-learn==1.8.0")
         .add_local_python_source("features").add_local_dir("worker", "/root/worker").add_local_file("data/ab-test-input.json", "/root/data/ab-test-input.json"))
@app.function(image=image, cpu=4.0, memory=8192, timeout=900, volumes={"/embed": modal.Volume.from_name("sonic-embed")})
def run():
    import os, sys, subprocess, tempfile, urllib.request, base64, io
    import numpy as np, soundfile as sf
    sys.path.insert(0, "/root")
    from features import stems as S
    from worker.parts import ab_render
    I = json.load(open("/root/data/ab-test-input.json")); w = tempfile.mkdtemp(); mp3, wav = os.path.join(w, "r.mp3"), os.path.join(w, "r.wav")
    urllib.request.urlretrieve(I["url"], mp3)
    dur = float(subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", mp3], capture_output=True, text=True).stdout.strip() or 0)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-ss", str(max(0.0, (dur - 60) / 2)), "-t", "60", "-i", mp3, "-ac", "2", "-ar", "44100", wav], check=True)
    st = S.separate(wav, os.path.join(w, "sep")); out = {}
    for part, tc in I["targets"].items():
        r = ab_render(st, {"part": part, "tc": tc, "bpm": I.get("bpm"), "key": I.get("key")})
        if r.get("error"): out[part] = r; continue
        def dec(b):
            p_ = os.path.join(w, part + "_x.mp3"); open(p_, "wb").write(base64.b64decode(b))
            subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-i", p_, p_ + ".wav"], check=True); x, sr = sf.read(p_ + ".wav", always_2d=True); return x, sr
        A, sr = dec(r["a"]); B, _ = dec(r["b"]); m = min(len(A), len(B))
        out[part] = {"loop": r["loop"], "closeness": r["closeness"], "from_s": r["from_s"], "tempo": r["tempo"], "key": r["key"],
                     "a_seconds": round(len(A) / sr, 1), "b_seconds": round(len(B) / sr, 1), "kb_each": round(len(r["a"]) * 3 / 4 / 1024),
                     "difference": round(float(np.sqrt(np.mean((A[:m] - B[:m]) ** 2)) / (np.sqrt(np.mean(A[:m] ** 2)) + 1e-9)), 3)}
    return out
@app.local_entrypoint()
def main():
    r = run.remote(); json.dump(r, open("data/ab-test.json", "w"))
    print("::notice title=ab test::" + json.dumps({k: (v.get("error") or f"{v['a_seconds']}s / {v['b_seconds']}s, {v['kb_each']} KB each, difference {v['difference']}, loop {str((v.get('loop') or {}).get('name'))[:30]} at {(v.get('loop') or {}).get('tempo')} BPM") for k, v in r.items()}))
