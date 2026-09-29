"""Does the reading move the way a producer's change does? (the Producer tool's sensitivity test)

Twenty records are separated once; their own parts are remixed with one known change each (bass +3 dB, bass -3 dB,
voice +3 dB, drums -3 dB, melody brighter by a +6 dB treble shelf) and read back exactly as an upload is: separated
again and measured by features/stems. The true values are the same measures taken on the clean remixed parts, so the
test asks only whether a reading of the finished mix recovers the change. Results arrive as run annotations.
    modal run worker/sensitivity.py   (the sensitivity workflow)
"""
import json, modal

app = modal.App("sonic-sensitivity")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("ffmpeg", "libsndfile1")
         .pip_install("numpy<2", "librosa==0.10.2", "soundfile", "demucs==4.0.1", "torch==2.3.1",
                      "torchaudio==2.3.1", "essentia-tensorflow", "scikit-learn==1.8.0")
         .add_local_python_source("features"))

RECORDS = [
[
"bp:27975069",
"140-deep-dubstep-grime",
"https://geo-samples.beatport.com/track/25197251-7ade-49bd-952b-db8728e19fdf.LOFI.mp3"
],
[
"bp:1539453",
"afro-house",
"https://geo-samples.beatport.com/track/859e1a3c-ad46-4034-8403-66dbfad76ec8.LOFI.mp3"
],
[
"bp:16496325",
"amapiano",
"https://geo-samples.beatport.com/track/eba43b30-32ce-4710-bce7-d7923629a0ea.LOFI.mp3"
],
[
"bp:5853278",
"ambient-experimental",
"https://geo-samples.beatport.com/track/64c69706-6cad-40b0-adb6-78aeaf541a03.LOFI.mp3"
],
[
"bp:28922906",
"bass-house",
"https://geo-samples.beatport.com/track/c1904c8f-4e80-4608-891c-88ad22a576e9.LOFI.mp3"
],
[
"bp:407864",
"breaks-breakbeat-uk-bass",
"https://geo-samples.beatport.com/track/4a49e2da-938f-49d9-9fd0-77a893ff05e7.LOFI.mp3"
],
[
"bp:269191",
"deep-house",
"https://geo-samples.beatport.com/track/3ae8d0af-e7d1-48f3-ac45-09f3cf184ec5.LOFI.mp3"
],
[
"bp:1380481",
"downtempo",
"https://geo-samples.beatport.com/track/07247856-36e9-4898-8764-939652d3fed0.LOFI.mp3"
],
[
"bp:20037093",
"drum-and-bass",
"https://geo-samples.beatport.com/track/75fafbbd-1d02-4c52-8ea6-fe2420065e14.LOFI.mp3"
],
[
"bp:17194309",
"electro",
"https://geo-samples.beatport.com/track/a130a6c6-54e1-43ed-86d5-104ce9dea9a8.LOFI.mp3"
],
[
"bp:16247303",
"hard-techno",
"https://geo-samples.beatport.com/track/6a65a1f8-446e-4eb8-a120-48478cf4ce10.LOFI.mp3"
],
[
"bp:127533",
"house",
"https://geo-samples.beatport.com/track/46274bfb-3a71-4ef7-81fe-b85ecd69c2bc.LOFI.mp3"
],
[
"bp:19119572",
"indie-dance",
"https://geo-samples.beatport.com/track/7e0708b4-6d32-4deb-aca4-8a579508e40b.LOFI.mp3"
],
[
"bp:23355575",
"jackin-house",
"https://geo-samples.beatport.com/track/c8ed6879-fb04-441f-97b0-521ec3b42f93.LOFI.mp3"
],
[
"bp:20579191",
"latin-electronic",
"https://geo-samples.beatport.com/track/de6ee01a-0e9c-4769-98f5-22aa99418862.LOFI.mp3"
],
[
"bp:20540001",
"mainstage",
"https://geo-samples.beatport.com/track/2f642b45-0e8b-4dc3-b870-f8d93150b083.LOFI.mp3"
],
[
"bp:11474709",
"melodic-house-techno",
"https://geo-samples.beatport.com/track/422b68fa-2a6a-42bc-95b5-7b539b3a54cf.LOFI.mp3"
],
[
"bp:248310",
"minimal-deep-tech",
"https://geo-samples.beatport.com/track/7cbdf529-0116-4868-be6c-14a91ebbd4c5.LOFI.mp3"
],
[
"bp:23981090",
"nu-disco-disco",
"https://geo-samples.beatport.com/track/9a698318-1772-4fde-a192-d16c28635816.LOFI.mp3"
],
[
"bp:6681575",
"organic-house",
"https://geo-samples.beatport.com/track/7c80bed4-cff8-46a0-9b78-e8ccaf0a098e.LOFI.mp3"
]
]
VARIANTS = {"control": {}, "bass +3 dB": {"bass": 3.0}, "bass -3 dB": {"bass": -3.0}, "voice +3 dB": {"vocals": 3.0},
            "drums -3 dB": {"drums": -3.0}, "melody brighter": {"shelf": 6.0}}


@app.function(image=image, cpu=4.0, memory=8192, timeout=1800)
def one(rec):
    import os, subprocess, tempfile, urllib.request
    import numpy as np, soundfile as sf
    from features import stems as S
    tid, scene, url = rec
    w = tempfile.mkdtemp(); mp3, wav = os.path.join(w, "in.mp3"), os.path.join(w, "in.wav")
    try:
        urllib.request.urlretrieve(url, mp3)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-ss", "20", "-t", "60", "-i", mp3, "-ac", "2", "-ar", "44100", wav], check=True, timeout=120)
        st = S.separate(wav, os.path.join(w, "sep0"))
        parts = {}
        for k, p in st.items():
            x, sr = sf.read(p, always_2d=True); parts[k] = (x.astype(np.float32), sr)
        n = min(len(x) for x, _ in parts.values())
        def shelf(x, sr, gain_db, f0=3000.0):
            X = np.fft.rfft(x, axis=0); f = np.fft.rfftfreq(len(x), 1.0 / sr)
            ramp = np.clip((f - f0 / 2) / (f0 / 2), 0, 1); g = 10 ** (gain_db * ramp / 20.0)
            return np.fft.irfft(X * g[:, None], n=len(x), axis=0).astype(np.float32)
        out = {}
        for name, v in VARIANTS.items():
            d = os.path.join(w, name.replace(" ", "_").replace("+", "p").replace("-", "m")); os.makedirs(d, exist_ok=True)
            mixed, true = {}, {}
            for k, (x, sr) in parts.items():
                y = x[:n].copy()
                if k in v: y = y * (10 ** (v[k] / 20.0))
                if k == "other" and "shelf" in v: y = shelf(y, sr, v["shelf"])
                mixed[k] = y; pth = os.path.join(d, "true_" + k + ".wav"); sf.write(pth, y, sr); true[k] = S.measure_stem(pth)
            mix = sum(mixed.values()); pk = float(np.abs(mix).max()) or 1.0
            g = 0.89 / pk if pk > 0.89 else 1.0; mp = os.path.join(d, "mix.wav"); sf.write(mp, mix * g, sr)
            true = {k: (t or {}) for k, t in true.items()}   # shares are ratios, so the mix's peak scaling cancels; a silent part measures as nothing
            st2 = S.separate(mp, os.path.join(d, "sep")); meas = {k: S.measure_stem(p) for k, p in st2.items()}
            def shares(r):
                t = sum((r[k] or {}).get("level", 0) for k in r) or 1.0
                return {k: round((r[k] or {}).get("level", 0) / t, 4) for k in r}
            out[name] = {"measured": shares(meas), "true": shares(true),
                         "other_centroid": (meas.get("other") or {}).get("centroid_hz"), "other_centroid_true": (true.get("other") or {}).get("centroid_hz")}
        return {"id": tid, "scene": scene, "results": out}
    except Exception as e:
        return {"id": tid, "scene": scene, "error": type(e).__name__ + ": " + str(e)[:160]}


@app.local_entrypoint()
def main():
    import statistics as st
    R = list(one.map(RECORDS, return_exceptions=True))
    ok = [r for r in R if isinstance(r, dict) and "results" in r]
    errs = [r.get("error") if isinstance(r, dict) else repr(r)[:120] for r in R if not (isinstance(r, dict) and "results" in r)]
    print("::notice title=sensitivity records::" + json.dumps({"read": len(ok), "failed": len(errs), "first_errors": errs[:3]}))
    PART = {"bass +3 dB": "bass", "bass -3 dB": "bass", "voice +3 dB": "vocals", "drums -3 dB": "drums"}
    summary = {}
    for name, p in PART.items():
        dm, dt, sign, leak = [], [], 0, []
        for r in ok:
            c, v = r["results"]["control"], r["results"][name]
            a = v["measured"][p] - c["measured"][p]; b = v["true"][p] - c["true"][p]
            dm.append(a); dt.append(b); sign += (a > 0) == (b > 0)
            for q in c["measured"]:
                if q != p and (1 - c["measured"][p]) > 0 and (1 - v["measured"][p]) > 0:
                    leak.append(abs(v["measured"][q] / (1 - v["measured"][p]) - c["measured"][q] / (1 - c["measured"][p])))
        if ok:
            summary[name] = {"changed_part": p, "true_change_pts": round(st.median(dt) * 100, 1), "measured_change_pts": round(st.median(dm) * 100, 1),
                             "right_direction": f"{sign} of {len(ok)}", "recovered_share_of_true_change": round(st.median([a / b for a, b in zip(dm, dt) if abs(b) > 1e-4]), 2),
                             "other_parts_shift_pts": round(st.median(leak) * 100, 1)}
    cm, ct, sg = [], [], 0
    for r in ok:
        c, v = r["results"]["control"], r["results"]["melody brighter"]
        if None in (c["other_centroid"], v["other_centroid"], c["other_centroid_true"], v["other_centroid_true"]): continue
        a = v["other_centroid"] - c["other_centroid"]; b = v["other_centroid_true"] - c["other_centroid_true"]; cm.append(a); ct.append(b); sg += (a > 0) == (b > 0)
    if cm: summary["melody brighter"] = {"true_centroid_change_hz": round(st.median(ct)), "measured_centroid_change_hz": round(st.median(cm)), "right_direction": f"{sg} of {len(cm)}"}
    acc = {}
    for q in ("drums", "bass", "other", "vocals"):
        e = [abs(r["results"]["control"]["measured"].get(q, 0) - r["results"]["control"]["true"].get(q, 0)) for r in ok]
        if e: acc[q] = round(st.median(e) * 100, 1)
    summary["control_share_error_pts"] = acc
    for k, v in summary.items(): print("::notice title=sensitivity " + k + "::" + json.dumps(v))
    print("SENSITIVITY " + json.dumps(summary))
