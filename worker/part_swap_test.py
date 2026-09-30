"""Does changing one part move the whole track toward another style? (the part-swap test for Aim a track)

Pairs of records from two styles of the same scene (within 2% in tempo, same or relative key; data/part-swap-pairs.json)
are each separated into drums, bass, melody and voice. For each pair, four mixes of record A are placed in the learned
ear exactly as an upload is (worker/embed.py, learned_call, on the middle 60 seconds at 32 kHz mono): A rebuilt from its
own separated parts (the control: splitting and remixing alone should not move it), A with B's bass, A with B's drums,
and B itself. Scored against the two styles' centres: does the swap move A toward B's style, how far along the way,
and how often does its nearest style flip? Results arrive as run annotations.   modal run worker/part_swap_test.py
"""
import json, modal
app = modal.App("sonic-part-swap-test")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg", "libsndfile1")
         .pip_install("numpy<2", "librosa==0.10.2", "soundfile", "demucs==4.0.1", "torch==2.3.1", "torchaudio==2.3.1", "essentia-tensorflow", "scikit-learn==1.8.0")
         .add_local_python_source("features", "worker").add_local_file("data/part-swap-pairs.json", "/root/data/part-swap-pairs.json"))

@app.function(image=image, cpu=4.0, memory=8192, timeout=1800, volumes={"/embed": modal.Volume.from_name("sonic-embed")})
def pair(rec):
    import os, subprocess, tempfile, urllib.request, sys
    import numpy as np, soundfile as sf
    sys.path.insert(0, "/root")
    from features import stems as S
    from worker.embed import learned_call
    w = tempfile.mkdtemp(); RAW = {}
    def fetch(url, tag):
        mp3, wav = os.path.join(w, tag + ".mp3"), os.path.join(w, tag + ".wav")
        urllib.request.urlretrieve(url, mp3)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-ss", "20", "-t", "60", "-i", mp3, "-ac", "2", "-ar", "44100", wav], check=True, timeout=120)
        raw = os.path.join(w, tag + "_raw32k.wav"); subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-i", wav, "-ac", "1", "-ar", "32000", raw], check=True, timeout=120)
        RAW[tag] = ((learned_call(raw) or {}).get("ear") or {}).get("v")
        st = S.separate(wav, os.path.join(w, "sep_" + tag)); P = {}
        for k, pth in st.items():
            x, sr = sf.read(pth, always_2d=True); P[k] = x.astype(np.float32)
        return P, sr
    def ear_of(parts, sr, tag):
        n = min(len(x) for x in parts.values()); mix = sum(x[:n] for x in parts.values()); pk = float(np.abs(mix).max()) or 1.0
        mp = os.path.join(w, tag + "_mix.wav"); sf.write(mp, mix * (0.89 / pk if pk > 0.89 else 1.0), sr)
        mono = os.path.join(w, tag + "_32k.wav"); subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-i", mp, "-ac", "1", "-ar", "32000", mono], check=True, timeout=120)
        r = learned_call(mono) or {}; return (r.get("ear") or {}).get("v")
    try:
        A, sr = fetch(rec["ua"], "a"); B, _ = fetch(rec["ub"], "b")
        out = {"scene": rec["scene"], "a": rec["a"], "b": rec["b"], "ca": rec["ca"], "cb": rec["cb"]}
        out["raw"] = RAW.get("a"); out["orig"] = ear_of(A, sr, "orig")
        out["bass"] = ear_of({**A, "bass": B["bass"]}, sr, "swap_bass")
        out["drums"] = ear_of({**A, "drums": B["drums"]}, sr, "swap_drums")
        out["donor"] = ear_of(B, sr, "donor")
        return out
    except Exception as ex:
        return {"a": rec.get("a"), "error": type(ex).__name__ + ": " + str(ex)[:160]}

@app.local_entrypoint()
def main():
    import numpy as np
    P = json.load(open("data/part-swap-pairs.json"))["pairs"]
    R = list(pair.map(P, return_exceptions=True)); ok = [r for r in R if isinstance(r, dict) and r.get("orig") and r.get("bass") and r.get("drums") and r.get("donor")]
    bad = [r.get("error") if isinstance(r, dict) else repr(r)[:120] for r in R if r not in ok]
    u = lambda v: (lambda a: a / (np.linalg.norm(a) or 1))(np.array(v, np.float32)[:16])
    res = {"pairs_read": len(ok), "failed": len(bad), "first_errors": bad[:3]}
    ctl = [float(u(r["raw"]) @ u(r["orig"])) for r in ok if r.get("raw")]
    near = []
    for r in ok:
        if r.get("raw"):
            ra, o, ca, cb = u(r["raw"]), u(r["orig"]), u(r["ca"]), u(r["cb"]); near.append(abs(float((o @ cb - o @ ca) - (ra @ cb - ra @ ca))))
    res["control"] = {"cosine_original_to_rebuilt": round(float(np.median(ctl)), 4) if ctl else None, "median_shift_from_rebuilding": round(float(np.median(near)), 4) if near else None}
    for part in ("bass", "drums"):
        toward, frac, flips, closer_donor = [], [], 0, 0
        for r in ok:
            o, s, ca, cb, dn = u(r["orig"]), u(r[part]), u(r["ca"]), u(r["cb"]), u(r["donor"])
            gap_o = float(o @ cb - o @ ca); gap_s = float(s @ cb - s @ ca); gap_d = float(dn @ cb - dn @ ca)
            toward.append(gap_s - gap_o); frac.append((gap_s - gap_o) / (gap_d - gap_o) if abs(gap_d - gap_o) > 1e-3 else np.nan)
            flips += int(gap_o < 0 <= gap_s); closer_donor += int(float(s @ dn) > float(o @ dn))
        toward = np.array(toward); frac = np.array(frac)
        res[part] = {"median_shift": round(float(np.median(np.abs(toward))), 4), "moved_toward_donor_style": f"{int((toward > 0).sum())} of {len(toward)}", "median_share_of_the_way": round(float(np.nanmedian(frac)), 3),
                     "nearest_style_flipped": f"{flips} of {len(toward)}", "closer_to_the_donor_record": f"{closer_donor} of {len(toward)}"}
    print(json.dumps(res)); print("::notice title=part swap::" + json.dumps(res))
