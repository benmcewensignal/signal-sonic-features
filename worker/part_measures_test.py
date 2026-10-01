"""Two checks before Aim a track turns part measures into advice.

1. Calibration. A bounce's parts are measured by the reader; the target ranges come from the stored record rows. Read the
   same records both ways and see, measure by measure, how far apart they are and whether one predicts the other.
2. Read-back. Change one part of a real record in one known way, mix it back, separate it again, measure that part:
   does the measure move the way the change should move it? (louder: level and share up; darker: brightness down;
   compressed: punch and dynamic range down; noisier: noisiness up; busier: hits per second up.)
Writes data/part-measures-test.json.   modal run worker/part_measures_test.py
"""
import json, modal
app = modal.App("sonic-part-measures")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg", "libsndfile1")
         .pip_install("numpy<2", "librosa==0.10.2", "soundfile", "demucs==4.0.1", "torch==2.3.1", "torchaudio==2.3.1", "essentia-tensorflow", "scikit-learn==1.8.0", "scipy")
         .add_local_python_source("features").add_local_dir("worker", "/root/worker").add_local_file("data/part-swap-pairs.json", "/root/data/part-swap-pairs.json"))
MEAS = ("level", "crest", "dynamic_span", "centroid_hz", "rolloff_hz", "flatness", "onsets_per_s")
TRANS = {"louder": {"level": 1, "share_of_energy": 1}, "darker": {"centroid_hz": -1, "rolloff_hz": -1}, "compressed": {"crest": -1, "dynamic_span": -1},
         "noisier": {"flatness": 1}, "busier": {"onsets_per_s": 1}}

def measure_all(st, S):
    rec = {k: (S.measure_stem(v) or {}) for k, v in st.items()}
    tot = sum(r.get("level", 0) for r in rec.values()) or 1
    for r in rec.values(): r["share_of_energy"] = round(r.get("level", 0) / tot, 4)
    return rec

def transform(x, sr, how, bpm):
    import numpy as np
    from scipy.signal import butter, sosfilt
    if how == "louder": return x * 1.4125
    if how == "darker":
        sos = butter(4, 1500 / (sr / 2), btype="low", output="sos"); return sosfilt(sos, x, axis=0)
    if how == "compressed":
        pk = float(np.max(np.abs(x))) or 1.0; y = np.tanh(3.0 * x / pk) * pk / np.tanh(3.0)
        r0, r1 = float(np.sqrt(np.mean(x ** 2))) or 1.0, float(np.sqrt(np.mean(y ** 2))) or 1.0; return y * (r0 / r1)
    if how == "noisier":
        r0 = float(np.sqrt(np.mean(x ** 2))) or 1e-4; return x + np.random.default_rng(0).normal(0, r0 * 0.1, x.shape)
    if how == "busier":
        d = int(sr * 60.0 / max(bpm or 124.0, 60) / 4); y = x.copy(); y[d:] += 0.7 * x[:-d]; return y
    return x

@app.function(image=image, cpu=4.0, memory=8192, timeout=1800, volumes={"/embed": modal.Volume.from_name("sonic-embed")})
def one(rec):
    import os, subprocess, tempfile, urllib.request, sys
    import numpy as np, soundfile as sf
    sys.path.insert(0, "/root")
    from features import stems as S
    w = tempfile.mkdtemp(); mp3, wav = os.path.join(w, "r.mp3"), os.path.join(w, "r.wav"); out = {"id": rec["id"], "scene": rec["scene"]}
    try:
        urllib.request.urlretrieve(rec["url"], mp3)
        dur = float(subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", mp3], capture_output=True, text=True).stdout.strip() or 0)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-ss", str(max(0.0, (dur - 60) / 2)), "-t", "60", "-i", mp3, "-ac", "2", "-ar", "44100", wav], check=True, timeout=120)
        st = S.separate(wav, os.path.join(w, "base")); out["base"] = {k: {m: v.get(m) for m in MEAS + ("share_of_energy",)} for k, v in measure_all(st, S).items()}
        if rec.get("readback"):
            stems = {k: sf.read(p, always_2d=True) for k, p in st.items()}; sr = next(iter(stems.values()))[1]; rb = {}
            for part in ("drums", "bass", "other", "vocals"):
                if part not in stems: continue
                for how in TRANS:
                    mix = sum((transform(a, sr, how, rec.get("bpm")) if k == part else a) for k, (a, _) in stems.items())
                    pk = float(np.max(np.abs(mix))) or 1.0
                    if pk > 0.99: mix = mix * (0.99 / pk)
                    fp = os.path.join(w, f"{part}_{how}.wav"); sf.write(fp, mix, sr)
                    st2 = S.separate(fp, os.path.join(w, f"sep_{part}_{how}")); m2 = measure_all(st2, S).get(part) or {}
                    rb[f"{part}|{how}"] = {m: m2.get(m) for m in MEAS + ("share_of_energy",)}
            out["readback"] = rb
        return out
    except Exception as ex:
        return {"id": rec.get("id"), "error": type(ex).__name__ + ": " + str(ex)[:160]}

@app.local_entrypoint()
def main():
    import numpy as np
    P = json.load(open("data/part-swap-pairs.json"))["pairs"]; seen = set(); R = []
    for p in P:
        for i, u in ((p["a"], p["ua"]), (p["b"], p["ub"])):
            if i not in seen: seen.add(i); R.append({"id": i, "url": u, "scene": p["scene"], "bpm": p.get("bpm")})
    for k, r in enumerate(R): r["readback"] = k < 10
    res = [r for r in one.map(R, return_exceptions=True) if isinstance(r, dict) and r.get("base")]
    # the read-back verdicts: for each change and the measures it should move, how often the part moved that way
    verdict = {}
    for r in res:
        for key, m2 in (r.get("readback") or {}).items():
            part, how = key.split("|"); b = r["base"].get(part) or {}
            for meas, sign in TRANS[how].items():
                x0, x1 = b.get(meas), m2.get(meas)
                if not isinstance(x0, (int, float)) or not isinstance(x1, (int, float)): continue
                v = verdict.setdefault(meas, {"right": 0, "n": 0, "by_part": {}}); ok = (x1 - x0) * sign > 0
                v["right"] += ok; v["n"] += 1; bp = v["by_part"].setdefault(part, [0, 0]); bp[0] += ok; bp[1] += 1
    json.dump({"note": __doc__.split("Writes")[0].strip(), "records": res, "readback": verdict}, open("data/part-measures-test.json", "w"))
    print("::notice title=part measures::" + json.dumps({"read": len(res), "of": len(R), "readback": {k: f"{v['right']} of {v['n']}" for k, v in verdict.items()}}))
