"""The demo for Track everything (earlysignal.live/track): a released electro record's middle minute read exactly as an
upload is (worker/parts.py read), without its parts' audio, since a released record's parts are not ours to play.
Writes data/track-demo.json.   modal run worker/track_demo.py"""
import json, modal
app = modal.App("sonic-track-demo")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg", "libsndfile1")
         .pip_install("numpy<2", "librosa==0.10.2", "soundfile", "demucs==4.0.1", "torch==2.3.1", "torchaudio==2.3.1", "essentia-tensorflow", "scikit-learn==1.8.0")
         .add_local_python_source("features").add_local_dir("worker", "/root/worker"))
REC = {"id": "bp:18842188", "name": "Method to My Madness", "artists": ["DJ Godfather"], "label": "Databass",
       "preview": "https://geo-samples.beatport.com/track/25c8ab71-e987-4c54-bf66-6d65f6d7123d.LOFI.mp3"}

@app.function(image=image, cpu=4.0, memory=8192, timeout=1500, volumes={"/embed": modal.Volume.from_name("sonic-embed")})
def run():
    import os, subprocess, tempfile, urllib.request, sys
    sys.path.insert(0, "/root")
    from worker.parts import read
    w = tempfile.mkdtemp(); mp3, wav = os.path.join(w, "b.mp3"), os.path.join(w, "b.wav")
    urllib.request.urlretrieve(REC["preview"], mp3)
    dur = float(subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", mp3], capture_output=True, text=True).stdout.strip() or 0)
    # the page sends the middle sixty seconds, mono, at 32 kHz: the same here
    subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-ss", str(max(0.0, (dur - 60) / 2)), "-t", "60", "-i", mp3, "-ac", "1", "-ar", "32000", wav], check=True, timeout=120)
    out = read(wav, False)
    out.pop("part_audio", None)
    return dict(REC, reading=out, seconds=min(60.0, dur))

@app.local_entrypoint()
def main():
    import datetime, os
    r = run.remote(); r["built"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ")
    os.makedirs("data", exist_ok=True)
    json.dump(r, open("data/track-demo.json", "w"), separators=(",", ":"))
    rd = r.get("reading") or {}
    print("keys:", sorted(rd.keys()))
    for k in ("scene_error", "mix_error", "learned_error", "part_calls_error", "part_residuals_error"):
        if rd.get(k): print(k, rd[k])
