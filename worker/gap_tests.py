"""Two tests of the Catalogue gap (tools/catalogue_gaps.py), run on Modal; results land in data/gap-tests/.

  modal run worker/gap_tests.py::sep   Separation round trip. 40 openly licensed bass loops, each mixed with a drum loop and
                                       a melody loop at a matching tempo, are separated with the same htdemucs the records
                                       go through and measured with the same functions. If separation alone moves a clean
                                       loop away from itself, part of every gap is the pipeline, not the music.
  modal run worker/gap_tests.py::gen   Generate and measure, on drum and bass's bass gap. An open text-to-music model
                                       (MusicGen medium, non-commercial weights; a test, not a product) makes basslines from
                                       prompts written from the gap's own words; each is measured as a loop is, and also
                                       after separation, as a record is. How many of the far records then have something
                                       close? Clips are kept so a person can listen: the measures cannot say if they are good.
"""
import io, json, math, os, modal

app = modal.App("sonic-gap-tests")
cpu_image = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg", "libsndfile1")
             .pip_install("numpy<2", "librosa==0.10.2", "soundfile", "demucs==4.0.1", "torch==2.3.1", "torchaudio==2.3.1", "scikit-learn==1.8.0", "requests")
             .add_local_python_source("features", "worker"))
gpu_image = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg")
             .pip_install("numpy<2", "torch==2.3.1", "transformers==4.46.3", "soundfile", "scipy", "sentencepiece"))
HF = modal.Volume.from_name("sonic-hf-cache", create_if_missing=True)
SC = ("level", "crest", "dynamic_span", "centroid_hz", "rolloff_hz", "flatness", "onsets_per_s", "share_of_energy")
SR = 44100


def _wav(x, path, sr=SR):
    import soundfile as sf
    sf.write(path, x, sr)
    return path


def _measure(path):
    """The exact vector the gap compares: the 45-number profile plus the eight plain part measures (worker loops and
    separated record parts are both built this way)."""
    from features import stems as S
    m = S.measure_stem(path) or {}
    e = S.analyse_stem(path) or {}
    emb = e.get("embedding")
    if not (isinstance(emb, list) and len(emb) == 45): return None
    return [float(z) for z in emb] + [float(m.get(k)) if isinstance(m.get(k), (int, float)) else 0.0 for k in SC]


def _load(url, tag, w):
    import subprocess, urllib.request, soundfile as sf, numpy as np
    src, wav = os.path.join(w, tag + ".src"), os.path.join(w, tag + ".wav")
    urllib.request.urlretrieve(url, src)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-i", src, "-ac", "2", "-ar", str(SR), wav], check=True, timeout=120)
    x, _ = sf.read(wav, always_2d=True)
    return x.astype(np.float32)


@app.function(image=cpu_image, cpu=4.0, memory=8192, timeout=1500)
def sep_case(c):
    import tempfile, numpy as np
    from features import stems as S
    w = tempfile.mkdtemp(); N = 30 * SR
    try:
        def tile(x): return np.tile(x, (int(math.ceil(N / len(x))), 1))[:N]
        def rms(x, db): return x * (10 ** (db / 20) / (float(np.sqrt(np.mean(x ** 2))) + 1e-9))
        B = rms(tile(_load(c["bass"]["url"], "b", w)), -16.0)
        D = rms(tile(_load(c["drums"]["url"], "d", w)), -14.0)
        M = rms(tile(_load(c["melody"]["url"], "m", w)), -19.0)
        mix = B + D + M; pk = float(np.abs(mix).max()); g = 0.89 / pk if pk > 0.89 else 1.0
        B, D, M, mix = B * g, D * g, M * g, mix * g
        out = {"id": c["bass"]["i"]}
        out["clean"] = {k: _measure(_wav(x, os.path.join(w, k + "_clean.wav"))) for k, x in (("bass", B), ("drums", D), ("melody", M))}
        st = S.separate(_wav(mix, os.path.join(w, "mix.wav")), os.path.join(w, "sep_mix"))
        out["sep"] = {"bass": _measure(st["bass"]), "drums": _measure(st["drums"]), "melody": _measure(st["other"])}
        so = S.separate(os.path.join(w, "bass_clean.wav"), os.path.join(w, "sep_solo"))   # the bass alone through the separator
        out["solo"] = {"bass": _measure(so["bass"])}
        return out
    except Exception as ex:
        return {"id": c["bass"]["i"], "error": type(ex).__name__ + ": " + str(ex)[:200]}


@app.function(image=gpu_image, gpu="A10G", memory=16384, timeout=2400, volumes={"/cache": HF})
def gen_batch(prompts, n=4, seconds=15):
    import torch, numpy as np
    os.environ["HF_HOME"] = "/cache"
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    proc = AutoProcessor.from_pretrained("facebook/musicgen-medium", cache_dir="/cache")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium", cache_dir="/cache").to("cuda")
    sr = model.config.audio_encoder.sampling_rate; res = []
    for p in prompts:
        inp = proc(text=[p] * n, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            a = model.generate(**inp, do_sample=True, guidance_scale=3.0, max_new_tokens=int(seconds * 50))
        for k in range(n):
            x = a[k, 0].float().cpu().numpy(); buf = io.BytesIO()
            import soundfile as sf; sf.write(buf, x, sr, format="WAV"); res.append({"prompt": p, "k": k, "sr": sr, "wav": buf.getvalue()})
    HF.commit()
    return res


@app.function(image=cpu_image, cpu=4.0, memory=8192, timeout=1500)
def gen_measure(clip):
    import tempfile, subprocess, librosa, numpy as np
    from features import stems as S
    w = tempfile.mkdtemp()
    try:
        raw = os.path.join(w, "raw.wav"); open(raw, "wb").write(clip["wav"])
        wav = os.path.join(w, "clip.wav"); subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-i", raw, "-ac", "2", "-ar", str(SR), wav], check=True, timeout=120)
        y, sr = librosa.load(wav, sr=22050, mono=True); bt = librosa.beat.beat_track(y=y, sr=sr)[0]; bt = float(bt[0] if hasattr(bt, "__len__") else bt)
        out = {"prompt": clip["prompt"], "k": clip["k"], "tempo": round(bt, 1), "clean": _measure(wav)}
        st = S.separate(wav, os.path.join(w, "sep")); out["sep_bass"] = _measure(st["bass"])
        mp3 = os.path.join(w, "clip.mp3"); subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-i", wav, "-b:a", "128k", mp3], check=True, timeout=120)
        out["mp3"] = open(mp3, "rb").read()
        return out
    except Exception as ex:
        return {"prompt": clip.get("prompt"), "k": clip.get("k"), "error": type(ex).__name__ + ": " + str(ex)[:200]}


@app.function(image=cpu_image, memory=4096, timeout=600, volumes={"/embed": modal.Volume.from_name("sonic-embed")})
def record_vectors(ids):
    import numpy as np
    R = np.load("/embed/record-parts.npz"); at = {t: i for i, t in enumerate(R["ids"].tolist())}
    return {t: {"bass": [float(z) for z in R["V"][at[t], 1]], "tempo": float(R["tempo"][at[t]])} for t in ids if t in at}


# ---------------------------------------------------------------- local side: choose the cases, score, write the results
def _index():
    import pickle
    return pickle.load(open("worker/loop_index.pkl", "rb"))


def _z(v, L):
    import numpy as np
    a = (np.array(v, np.float32)[L["keep"]] - np.array(L["mu"], np.float32)) / np.array(L["sd"], np.float32)
    return a / (np.linalg.norm(a) + 1e-9)


def _fits(lt, T):
    import numpy as np
    ok = np.zeros(len(lt), bool)
    for f in (1, 2, 0.5): ok |= (lt > 0) & (np.abs(lt * f - T) / T <= 0.08)
    return ok


def _q(a):
    import numpy as np
    a = np.array(a, float)
    return {"n": int(len(a)), "median": round(float(np.median(a)), 3), "p10": round(float(np.percentile(a, 10)), 3), "p90": round(float(np.percentile(a, 90)), 3),
            "under_0.45": round(float((a < 0.45).mean()), 3), "at_least_0.7": round(float((a >= 0.7).mean()), 3)} if len(a) else {"n": 0}


@app.local_entrypoint()
def sep():
    import random, numpy as np
    LI = _index(); rng = random.Random(7); fam = {"bass": "bass", "drums": "drums", "melody": "melody"}
    bass = [(i, m) for i, m in enumerate(LI["bass"]["meta"]) if m.get("tempo") and m.get("preview")]; rng.shuffle(bass); cases = []
    for i, m in bass:
        T = float(m["tempo"]); pick = {}
        for f in ("drums", "melody"):
            c = [(j, x) for j, x in enumerate(LI[f]["meta"]) if x.get("tempo") and x.get("preview") and abs(float(x["tempo"]) - T) / T <= 0.03]
            if c: pick[f] = rng.choice(c)
        if len(pick) == 2:
            cases.append({"bass": {"i": i, "url": m["preview"]}, "drums": {"i": pick["drums"][0], "url": pick["drums"][1]["preview"]},
                          "melody": {"i": pick["melody"][0], "url": pick["melody"][1]["preview"]}})
        if len(cases) >= 40: break
    R = list(sep_case.map(cases, return_exceptions=True))
    ok = [r for r in R if isinstance(r, dict) and not r.get("error") and r.get("sep") and r.get("clean")]
    errs = [r.get("error") if isinstance(r, dict) else repr(r)[:160] for r in R if r not in ok]
    res = {"cases": len(cases), "read": len(ok), "errors": errs[:4], "families": {}}
    for f, F in fam.items():
        L = LI[F]; Z = np.array(L["Z"], np.float32); idx = {"bass": "bass", "drums": "drums", "melody": "melody"}[f]
        own = [c[idx]["i"] for c in cases]
        s_sep, s_lvl, s_tot, s_solo, ranks, plain = [], [], [], [], [], {k: [] for k in ("centroid_hz", "onsets_per_s", "crest", "flatness")}
        for r, c in zip(R, cases):
            if r not in ok: continue
            cl, sp = r["clean"].get(f), r["sep"].get(f)
            if not (cl and sp): continue
            zc, zs = _z(cl, L), _z(sp, L); zi = Z[c[idx]["i"]] / (np.linalg.norm(Z[c[idx]["i"]]) + 1e-9)
            s_sep.append(float(zs @ zc)); s_lvl.append(float(zc @ zi)); s_tot.append(float(zs @ zi))
            ranks.append(int((Z @ zs > float(Z[c[idx]["i"]] @ zs)).sum()) + 1)
            for k in plain: 
                j = 45 + SC.index(k)
                if cl[j] and sp[j]: plain[k].append(sp[j] / cl[j])
            if f == "bass" and (r.get("solo") or {}).get("bass"): s_solo.append(float(_z(r["solo"]["bass"], L) @ zc))
        rk = np.array(ranks) if ranks else np.array([0])
        res["families"][f] = {"separated_vs_same_loop_clean": _q(s_sep), "clean_in_mix_vs_index_entry": _q(s_lvl), "separated_vs_index_entry": _q(s_tot),
                              "own_loop_rank_when_searched_with_separated_part": {"median": int(np.median(rk)), "first": round(float((rk == 1).mean()), 3), "top5": round(float((rk <= 5).mean()), 3), "of": len(Z)},
                              "separated_over_clean_median_ratio": {k: round(float(np.median(v)), 3) for k, v in plain.items() if v}}
        if f == "bass": res["families"][f]["bass_alone_separated_vs_clean"] = _q(s_solo)
    os.makedirs("data/gap-tests", exist_ok=True); json.dump(res, open("data/gap-tests/separation.json", "w"), indent=1)
    print("SEP " + json.dumps(res))


@app.local_entrypoint()
def gen():
    import numpy as np
    LI = _index(); L = LI["bass"]; Z = np.array(L["Z"], np.float32); lt = np.array([m.get("tempo") or 0 for m in L["meta"]], np.float32)
    far = json.load(open("worker/catalogue_far.json"))["scenes"]["drum-and-bass"]["bass"]
    V = record_vectors.remote(far); ids = [t for t in far if t in V]
    target = {t: _z(V[t]["bass"], L) for t in ids}
    base = {t: float((Z[_fits(lt, V[t]["tempo"])] @ target[t]).max()) if V[t]["tempo"] and _fits(lt, V[t]["tempo"]).any() else -1.0 for t in ids}
    P = ["isolated reese bassline, drum and bass, 174 bpm, bright gritty mid-range bass, no drums, F minor",
         "neurofunk bass loop, 174 bpm, punchy distorted mid bass, solo bass, no drums",
         "liquid drum and bass bassline, 174 bpm, warm tonal melodic bass, solo bass, no drums",
         "jump up drum and bass wobble bass, 174 bpm, bright and punchy, solo bass, no drums",
         "rolling drum and bass sub bass with a growl on top, 174 bpm, isolated bass, no drums",
         "drum and bass bass stab riff, 174 bpm, bright tonal bass, no drums",
         "bass loop, 174 bpm", "bass guitar loop, 174 bpm"]
    CONTROL = set(P[-2:])
    clips = gen_batch.remote(P, n=4, seconds=15)
    M = [r for r in gen_measure.map(clips, return_exceptions=True) if isinstance(r, dict) and not r.get("error") and r.get("clean")]
    def best(kind, prompts=None):
        out = {}
        for t in ids:
            T = V[t]["tempo"]; b = -1.0
            for r in M:
                if prompts is not None and r["prompt"] not in prompts: continue
                v = r.get(kind)
                if not v or not r["tempo"] or not T: continue
                if not _fits(np.array([r["tempo"]], np.float32), T)[0]: continue
                b = max(b, float(_z(v, L) @ target[t]))
            out[t] = b
        return out
    tgt = set(P) - CONTROL
    b_clean, b_sep, b_ctl = best("clean", tgt), best("sep_bass", tgt), best("clean", CONTROL)
    nofit = lambda d: sum(1 for v in d.values() if v < -0.5)
    def share(d, lo): return round(float(np.mean([1.0 if d[t] >= lo else 0.0 for t in ids])), 3)
    per = {}
    for r in M:
        zc = _z(r["clean"], L); s = [float(zc @ target[t]) for t in ids]
        per.setdefault(r["prompt"], []).append({"k": r["k"], "tempo": r["tempo"], "near": int(sum(1 for x in s if x >= 0.45)), "best": round(max(s), 3),
                                                "centroid_hz": round(r["clean"][45 + SC.index("centroid_hz")], 1), "onsets_per_s": round(r["clean"][45 + SC.index("onsets_per_s")], 2)})
    res = {"far_records": len(far), "with_vectors": len(ids), "clips": len(clips), "measured": len(M),
           "clips_that_fit_174_bpm": sum(1 for r in M if r["tempo"] and _fits(np.array([r["tempo"]], np.float32), 174.0)[0]),
           "freesound_best": _q([base[t] for t in ids if base[t] > -0.5]), "freesound_no_loop_at_tempo": sum(1 for t in ids if base[t] < -0.5),
           "targeted_clean": {"no_longer_far": share(b_clean, 0.45), "close": share(b_clean, 0.7), "best": _q([v for v in b_clean.values() if v > -0.5]), "no_clip_at_tempo": nofit(b_clean)},
           "targeted_separated": {"no_longer_far": share(b_sep, 0.45), "close": share(b_sep, 0.7), "best": _q([v for v in b_sep.values() if v > -0.5])},
           "control_clean": {"no_longer_far": share(b_ctl, 0.45), "close": share(b_ctl, 0.7), "best": _q([v for v in b_ctl.values() if v > -0.5])},
           "far_median_plain": {k: round(float(np.median([V[t]["bass"][45 + SC.index(k)] for t in ids])), 2) for k in ("centroid_hz", "onsets_per_s")},
           "per_prompt": per}
    os.makedirs("data/gap-tests/clips", exist_ok=True)
    ranked = sorted(M, key=lambda r: -sum(1 for t in ids if float(_z(r["clean"], L) @ target[t]) >= 0.45))
    keep = ranked[:6] + [r for r in M if r["prompt"] in CONTROL][:2]
    res["kept_clips"] = []
    for n, r in enumerate(keep):
        fn = f"clip{n + 1:02d}.mp3"; open(os.path.join("data/gap-tests/clips", fn), "wb").write(r["mp3"])
        res["kept_clips"].append({"file": fn, "prompt": r["prompt"], "tempo": r["tempo"], "control": r["prompt"] in CONTROL,
                                  "far_records_now_near": int(sum(1 for t in ids if float(_z(r["clean"], L) @ target[t]) >= 0.45))})
    json.dump(res, open("data/gap-tests/generate.json", "w"), indent=1)
    print("GEN " + json.dumps({k: v for k, v in res.items() if k != "per_prompt"}))
