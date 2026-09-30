"""A real reading for the pitch walkthrough: a released record's middle minute read exactly as a bounce is (worker/parts.py
read), its nearest style found from that reading's place in the ear, and the leverage reading run toward that style with
the style's most central record that has a preview. Writes data/demo-reading.json.   modal run worker/demo_reading.py"""
import json, modal
app = modal.App("sonic-demo-reading")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg", "libsndfile1")
         .pip_install("numpy<2", "librosa==0.10.2", "soundfile", "demucs==4.0.1", "torch==2.3.1", "torchaudio==2.3.1", "essentia-tensorflow", "scikit-learn==1.8.0")
         .add_local_python_source("features").add_local_dir("worker", "/root/worker").add_local_file("data/demo-reading-input.json", "/root/data/demo-reading-input.json"))

@app.function(image=image, cpu=4.0, memory=8192, timeout=1500, volumes={"/embed": modal.Volume.from_name("sonic-embed")})
def run():
    import os, subprocess, tempfile, urllib.request, sys, numpy as np
    sys.path.insert(0, "/root")
    from worker.parts import read
    I = json.load(open("/root/data/demo-reading-input.json")); w = tempfile.mkdtemp(); mp3, wav = os.path.join(w, "b.mp3"), os.path.join(w, "b.wav")
    urllib.request.urlretrieve(I["bounce_url"], mp3)
    dur = float(subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", mp3], capture_output=True, text=True).stdout.strip() or 0)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-ss", str(max(0.0, (dur - 60) / 2)), "-t", "60", "-i", mp3, "-ac", "1", "-ar", "44100", wav], check=True, timeout=120)
    out = read(wav, False)
    ev = np.array((((out.get("scene_learned") or {}).get("ear") or {}).get("v") or [])[:16], float)
    if len(ev) != 16: return {"error": "no ear", "reading": out}
    ev /= np.linalg.norm(ev) or 1
    S = sorted(I["styles"], key=lambda s: -float(np.dot(ev, np.array(s["c"]) / (np.linalg.norm(s["c"]) or 1))))
    st = next(s for s in S if s["donor"] != I["bounce"])
    lv = read(wav, False, {"url": st["donor_url"], "bpm": st["donor_bpm"], "id": st["donor"], "centre": st["c"]}, True)
    out.pop("part_audio", None)
    return {"bounce": I["bounce"], "style": st["id"], "reading": out, "leverage": lv.get("leverage")}

@app.local_entrypoint()
def main():
    r = run.remote(); json.dump(r, open("data/demo-reading.json", "w"))
    lv = (r.get("leverage") or {}).get("parts") or {}
    print("::notice title=demo reading::" + json.dumps({"bounce": r.get("bounce"), "style": r.get("style"), "leverage": {k: v.get("share") for k, v in lv.items()}, "error": r.get("error")}))
