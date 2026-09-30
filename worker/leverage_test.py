"""The leverage reading end to end, as the reader runs it: a catalogue record's middle minute read as a bounce, with a
donor from another style of its scene (data/part-swap-pairs.json) and that style's centre. Checks the reading returns a
share for each part, how long it takes, and whether drums lead as the part-swap test found. modal run worker/leverage_test.py"""
import json, modal
app = modal.App("sonic-leverage-test")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg", "libsndfile1")
         .pip_install("numpy<2", "librosa==0.10.2", "soundfile", "demucs==4.0.1", "torch==2.3.1", "torchaudio==2.3.1", "essentia-tensorflow", "scikit-learn==1.8.0")
         .add_local_python_source("features").add_local_dir("worker", "/root/worker").add_local_file("data/part-swap-pairs.json", "/root/data/part-swap-pairs.json"))

@app.function(image=image, cpu=4.0, memory=8192, timeout=900, volumes={"/embed": modal.Volume.from_name("sonic-embed")})
def one(rec):
    import os, subprocess, tempfile, urllib.request, sys, time
    sys.path.insert(0, "/root")
    from worker.parts import read
    w = tempfile.mkdtemp(); mp3, wav = os.path.join(w, "b.mp3"), os.path.join(w, "b.wav")
    try:
        urllib.request.urlretrieve(rec["ua"], mp3)
        dur = float(subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", mp3], capture_output=True, text=True).stdout.strip() or 0)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-ss", str(max(0.0, (dur - 60) / 2)), "-t", "60", "-i", mp3, "-ac", "1", "-ar", "32000", wav], check=True, timeout=120)
        t0 = time.time(); out = read(wav, False, {"url": rec["ub"], "bpm": 0, "id": rec["b"], "centre": rec["cb"]}); t = time.time() - t0
        lv = out.get("leverage") or {}
        return {"a": rec["a"], "seconds": round(t, 1), "leverage": {k: v for k, v in lv.items() if k != "loops"}, "loops": {k: [x.get("name") for x in v][:2] for k, v in (lv.get("loops") or {}).items()}}
    except Exception as ex:
        return {"a": rec.get("a"), "error": type(ex).__name__ + ": " + str(ex)[:160]}

@app.local_entrypoint()
def main():
    import numpy as np
    P = json.load(open("data/part-swap-pairs.json"))["pairs"][:8]
    R = list(one.map(P, return_exceptions=True)); ok = [r for r in R if isinstance(r, dict) and (r.get("leverage") or {}).get("parts")]
    bad = [r.get("error") or (r.get("leverage") or {}).get("error") if isinstance(r, dict) else repr(r)[:100] for r in R if r not in ok]
    lead = [max(r["leverage"]["parts"], key=lambda k: (r["leverage"]["parts"][k]["share"] or -9)) for r in ok]
    med = {k: round(float(np.nanmedian([r["leverage"]["parts"].get(k, {}).get("share") if r["leverage"]["parts"].get(k, {}).get("share") is not None else np.nan for r in ok])), 3) for k in ("drums", "bass", "other", "vocals")}
    res = {"read": len(ok), "failed": len(bad), "first_errors": bad[:3], "median_seconds": round(float(np.median([r["seconds"] for r in ok])), 1) if ok else None,
           "median_share": med, "leading_part": {k: lead.count(k) for k in set(lead)}, "loops_sample": ok[0]["loops"] if ok else None, "stretched": sum(1 for r in ok if r["leverage"].get("stretched"))}
    print(json.dumps(res)); print("::notice title=leverage::" + json.dumps(res))
