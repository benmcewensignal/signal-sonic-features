"""How close does the same part of the same record look to the loop matcher? (calibrating "nothing close" per part)

The gap report calls a record's part unmatched when no free loop scores 0.45 with it, one line for all four parts. But
the matcher finds a record's own loop first far less often for bass than for drums, so the line may not mean the same
for each part. Here each record's middle minute is separated, each part cut into its first and last 30 seconds, both
halves measured exactly as a loop is (features.stems.measure_stem: the 45-number embedding plus eight measures), and
the halves scored against each other with the loop index's own standardisation and cosine. The spread of these scores,
part by part, says what "close" can mean for that part. Writes data/own-halves.json.   modal run worker/own_halves_test.py
"""
import json, modal
app = modal.App("sonic-own-halves")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg", "libsndfile1")
         .pip_install("numpy<2", "librosa==0.10.2", "soundfile", "demucs==4.0.1", "torch==2.3.1", "torchaudio==2.3.1", "essentia-tensorflow", "scikit-learn==1.8.0")
         .add_local_python_source("features").add_local_dir("worker", "/root/worker").add_local_file("data/part-swap-pairs.json", "/root/data/part-swap-pairs.json"))

@app.function(image=image, cpu=4.0, memory=8192, timeout=1200, volumes={"/embed": modal.Volume.from_name("sonic-embed")})
def one(rec):
    import os, subprocess, tempfile, urllib.request, sys, pickle
    import numpy as np, soundfile as sf
    sys.path.insert(0, "/root")
    from features import stems as S
    LI = pickle.load(open("/root/worker/loop_index.pkl", "rb"))
    FAM = {"drums": "drums", "bass": "bass", "other": "melody", "vocals": "vocals"}
    SC = ("level", "crest", "dynamic_span", "centroid_hz", "rolloff_hz", "flatness", "onsets_per_s", "share_of_energy")
    w = tempfile.mkdtemp(); mp3, wav = os.path.join(w, "r.mp3"), os.path.join(w, "r.wav")
    try:
        urllib.request.urlretrieve(rec["url"], mp3)
        dur = float(subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", mp3], capture_output=True, text=True).stdout.strip() or 0)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-ss", str(max(0.0, (dur - 60) / 2)), "-t", "60", "-i", mp3, "-ac", "2", "-ar", "44100", wav], check=True, timeout=120)
        st = S.separate(wav, os.path.join(w, "sep")); out = {"id": rec["id"], "scene": rec["scene"], "sim": {}}
        for k, pth in st.items():
            if k not in FAM: continue
            x, sr = sf.read(pth, always_2d=True); h = len(x) // 2; vs = []
            for tag, seg in (("a", x[:h]), ("b", x[h:])):
                fp = os.path.join(w, f"{k}_{tag}.wav"); sf.write(fp, seg, sr); m = S.measure_stem(fp) or {}
                try:
                    from features.analyser_local import LocalAnalyser
                    fv = LocalAnalyser().analyse(fp); e_ = getattr(fv, "embedding", None)
                    if e_ is None and isinstance(fv, dict): e_ = fv.get("embedding")
                    if e_ is not None: m["embedding"] = [float(q) for q in e_]
                    else: out.setdefault("aerr", "no embedding in " + type(fv).__name__)
                except Exception as ex_:
                    import traceback; out.setdefault("aerr", type(ex_).__name__ + ": " + str(ex_)[:160] + " | " + traceback.format_exc().strip().splitlines()[-3][:160])
                e = m.get("embedding")
                if m.get("silent") or not (isinstance(e, list) and len(e) == 45): vs = None; break
                vs.append(np.array([float(v) for v in e] + [float(m.get(c)) if isinstance(m.get(c), (int, float)) else 0.0 for c in SC]))
            if not vs: continue
            L = LI[FAM[k]]; keep, mu, sd = L["keep"], np.asarray(L["mu"], float), np.asarray(L["sd"], float)
            z = [((v[keep] - mu) / sd) for v in vs]; z = [q / (np.linalg.norm(q) + 1e-9) for q in z]
            out["sim"][FAM[k]] = round(float(z[0] @ z[1]), 4)
        return out
    except Exception as ex:
        return {"id": rec.get("id"), "error": type(ex).__name__ + ": " + str(ex)[:120]}

@app.local_entrypoint()
def main():
    import numpy as np
    P = json.load(open("data/part-swap-pairs.json"))["pairs"]; seen = set(); R = []
    for p in P:
        for i, u in ((p["a"], p["ua"]), (p["b"], p["ub"])):
            if i not in seen: seen.add(i); R.append({"id": i, "url": u, "scene": p["scene"]})
    res = list(one.map(R, return_exceptions=True)); ok = [r for r in res if isinstance(r, dict) and r.get("sim")]
    errs = [(r.get("error") or r.get("aerr")) if isinstance(r, dict) else repr(r)[:100] for r in res if r not in ok][:3]
    summ = {}
    for part in ("drums", "bass", "melody", "vocals"):
        v = np.array([r["sim"][part] for r in ok if part in r["sim"]])
        if len(v): summ[part] = {"n": int(len(v)), "p10": round(float(np.percentile(v, 10)), 3), "p25": round(float(np.percentile(v, 25)), 3), "median": round(float(np.median(v)), 3), "below_045": round(float((v < 0.45).mean()), 3)}
    json.dump({"note": __doc__.split("   modal run")[0].strip(), "records": ok, "summary": summ}, open("data/own-halves.json", "w"))
    print("::notice title=own halves::" + json.dumps({"read": len(ok), "of": len(R), "summary": summ, "first_errors": errs}))
