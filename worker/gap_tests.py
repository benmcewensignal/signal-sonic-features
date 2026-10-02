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
    # the far set is recomputed from today's record vectors, the same rule as the gap report (best openly licensed bass
    # loop at a workable tempo under 0.45; the key filter left out, as it narrows nothing in practice): the far file in
    # worker/catalogue_far.json predates a remeasure, and nine in ten of its records are no longer far
    stale = json.load(open("worker/catalogue_far.json"))["scenes"]["drum-and-bass"]["bass"]
    pool = sorted(set(r[0] for r in json.load(open("data/scene-records-2026.json"))["scenes"]["drum-and-bass"]) | set(stale))
    V = record_vectors.remote(pool); allids = [t for t in pool if t in V and V[t]["tempo"]]
    zall = {t: _z(V[t]["bass"], L) for t in allids}
    base_all = {t: float((Z[_fits(lt, V[t]["tempo"])] @ zall[t]).max()) if _fits(lt, V[t]["tempo"]).any() else -1.0 for t in allids}
    ids = [t for t in allids if base_all[t] < 0.45]; target = {t: zall[t] for t in ids}; base = {t: base_all[t] for t in ids}
    stale_now_far = sum(1 for t in stale if t in base_all and base_all[t] < 0.45)
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
    res = {"dnb_records_with_bass_vectors": len(allids), "far_now": len(ids), "far_share_now": round(len(ids) / max(1, len(allids)), 3),
           "stale_far_file": len(stale), "stale_far_still_far": stale_now_far, "clips": len(clips), "measured": len(M),
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
    res["clip_vectors"] = [{"prompt": r["prompt"], "k": r["k"], "tempo": r["tempo"], "clean": r["clean"], "sep_bass": r.get("sep_bass")} for r in M]
    res["far_records_detail"] = {t: {"tempo": V[t]["tempo"], "freesound_best": round(base[t], 3), "gen_clean": round(b_clean[t], 3), "gen_sep": round(b_sep[t], 3), "control": round(b_ctl[t], 3)} for t in ids}
    json.dump(res, open("data/gap-tests/generate.json", "w"), indent=1)
    print("GEN " + json.dumps({k: v for k, v in res.items() if k not in ("per_prompt", "clip_vectors", "far_records_detail")}))


# ---------------------------------------------------------------- the practice-mix calibration
# Every library loop is mixed with other library loops at a matching tempo and unmixed with the records' own htdemucs,
# so each loop has a clean measurement (its index entry) and one or more smudged ones. Two fair versions of the gap:
#   A: record parts against the smudged library (both sides have been through the separator)
#   B: a correction learned from the practice pairs (smudged -> clean), checked on loops it never saw, applied to the
#      record parts, which are then compared with the clean library as today
# Limits: practice mixes are three or four loops with no mastering; real records' smudge may differ.
FAMS = ("drums", "bass", "melody", "vocals")
STEM = {"drums": "drums", "bass": "bass", "melody": "other", "vocals": "vocals"}
LEVEL = {"drums": -14.0, "bass": -16.0, "melody": -19.0, "vocals": -17.0}


@app.function(image=cpu_image, cpu=4.0, memory=8192, timeout=900, retries=1)
def mix_case(parts):
    import tempfile, numpy as np
    from features import stems as S
    w = tempfile.mkdtemp(); N = 16 * SR
    try:
        def tile(x): return np.tile(x, (int(math.ceil(N / len(x))), 1))[:N]
        def rms(x, db): return x * (10 ** (db / 20) / (float(np.sqrt(np.mean(x ** 2))) + 1e-9))
        X = {f: rms(tile(_load(u, f, w)), LEVEL[f]) for f, (i, u) in parts.items()}
        mix = sum(X.values()); pk = float(np.abs(mix).max()); g = 0.89 / pk if pk > 0.89 else 1.0
        st = S.separate(_wav(mix * g, os.path.join(w, "mix.wav")), os.path.join(w, "sep"))
        return {f: [i, _measure(st[STEM[f]])] for f, (i, u) in parts.items()}
    except Exception as ex:
        return {"error": type(ex).__name__ + ": " + str(ex)[:160]}


@app.function(image=cpu_image, memory=16384, timeout=1800, volumes={"/embed": modal.Volume.from_name("sonic-embed")})
def cal_gaps(lib, scenes):
    """Far shares per scene and part: today's rule, A (smudged library) and B (corrected record parts)."""
    import numpy as np
    R = np.load("/embed/record-parts.npz"); at = {t: i for i, t in enumerate(R["ids"].tolist())}; V = R["V"].astype(np.float32); TEMPO = R["tempo"]
    PI = {"drums": 0, "bass": 1, "melody": 2, "vocals": 3}; out = {}; dnb = {}
    for sc, ids in scenes.items():
        rows = [at[t] for t in ids if t in at]
        if len(rows) < 15: continue
        out[sc] = {"records": len(rows)}
        for f in FAMS:
            L = lib[f]; keep = np.array(L["keep"]); mu = np.array(L["mu"], np.float32); sd = np.array(L["sd"], np.float32)
            Zc = np.array(L["Z"], np.float32); Zs = np.array(L["Zs"], np.float32); has = np.array(L["has"], bool); lt = np.array(L["tempo"], np.float32)
            W = np.array(L["W"], np.float32)
            far = {"today": 0, "A": 0, "B": 0}; n = 0
            for i in rows:
                T = float(TEMPO[i]) or 0
                ok = np.zeros(len(lt), bool)
                if T:
                    for m in (1, 2, 0.5): ok |= (lt > 0) & (np.abs(lt * m - T) / T <= 0.08)
                else: ok[:] = True
                z = (V[i, PI[f]][keep] - mu) / sd; z /= np.linalg.norm(z) + 1e-9
                zb = np.append(z, 1.0) @ W; zb /= np.linalg.norm(zb) + 1e-9
                okA = ok & has
                b0 = float((Zc[ok] @ z).max()) if ok.any() else -1.0
                bA = float((Zs[okA] @ z).max()) if okA.any() else -1.0
                bB = float((Zc[ok] @ zb).max()) if ok.any() else -1.0
                far["today"] += b0 < 0.45; far["A"] += bA < 0.45; far["B"] += bB < 0.45; n += 1
                if sc == "drum-and-bass" and f == "bass": dnb.setdefault("best", []).append([round(b0, 3), round(bA, 3), round(bB, 3)])
            out[sc][f] = {k: round(v / max(1, n), 3) for k, v in far.items()}
    return {"scenes": out, "dnb_bass_best": dnb.get("best", [])}


@app.local_entrypoint()
def cal():
    import random, collections, numpy as np
    LI = _index(); rng = random.Random(11)
    meta = {f: [(i, m) for i, m in enumerate(LI[f]["meta"]) if m.get("tempo") and m.get("preview")] for f in FAMS}
    anchors = [("bass", x) for x in meta["bass"]] + [("melody", x) for x in meta["melody"]] + [("vocals", x) for x in meta["vocals"]] + \
              [("drums", x) for x in rng.sample(meta["drums"], min(500, len(meta["drums"])))] + [("bass", x) for x in rng.sample(meta["bass"], 250)]
    rng.shuffle(anchors); mixes = []
    for fam, (i, m) in anchors:
        T = float(m["tempo"]); parts = {fam: (i, m["preview"])}
        for f in ("drums", "bass", "melody") + (("vocals",) if rng.random() < 0.3 else ()):
            if f in parts: continue
            c = [x for x in meta[f] if abs(float(x[1]["tempo"]) - T) / T <= 0.03]
            if c: j, x = rng.choice(c); parts[f] = (j, x["preview"])
        if len(parts) >= 3: mixes.append(parts)
    print(f"practice mixes: {len(mixes)}")
    R = list(mix_case.map(mixes, return_exceptions=True))
    good = [r for r in R if isinstance(r, dict) and not r.get("error")]
    errs = collections.Counter((r.get("error") if isinstance(r, dict) else repr(r))[:60] for r in R if r not in good)
    sm = {f: collections.defaultdict(list) for f in FAMS}
    for r in good:
        for f, (i, v) in r.items():
            if v: sm[f][i].append(v)
    res = {"mixes": len(mixes), "separated": len(good), "errors": dict(errs.most_common(3)), "families": {}}
    lib = {}
    for f in FAMS:
        L = LI[f]; Zc = np.array(L["Z"], np.float32); Zc /= np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-9
        loops = sorted(sm[f]); zs = {i: [_z(v, L) for v in sm[f][i]] for i in loops}
        # how consistent is the smudge, and how far does it move a loop from itself
        self_raw = [float(z @ Zc[i]) for i in loops for z in zs[i]]
        twice = [float(zs[i][0] @ zs[i][1]) for i in loops if len(zs[i]) >= 2]
        # B: ridge from smudged to clean, five folds by loop
        X = np.array([np.append(z, 1.0) for i in loops for z in zs[i]], np.float32); Y = np.array([Zc[i] for i in loops for z in zs[i]], np.float32)
        grp = np.array([k % 5 for k, i in enumerate(loops) for z in zs[i]])
        best = None
        for lam in (0.3, 1.0, 3.0, 10.0):
            cv = []
            for k in range(5):
                tr, te = grp != k, grp == k
                W = np.linalg.solve(X[tr].T @ X[tr] + lam * np.eye(X.shape[1]), X[tr].T @ Y[tr])
                P = X[te] @ W; P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-9
                cv += list((P * Y[te]).sum(1))
            if best is None or np.median(cv) > np.median(best[1]): best = (lam, cv)
        lam, cvs = best
        W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
        # does a corrected part find its own loop first in the clean library? (held-out folds)
        first_raw, first_cor = [], []
        for k in range(5):
            tr = grp != k; Wk = np.linalg.solve(X[tr].T @ X[tr] + lam * np.eye(X.shape[1]), X[tr].T @ Y[tr])
            for kk, i in enumerate(loops):
                if kk % 5 != k: continue
                for z in zs[i]:
                    first_raw.append(int(np.argmax(Zc @ z) == i)); c = np.append(z, 1.0) @ Wk; c /= np.linalg.norm(c) + 1e-9; first_cor.append(int(np.argmax(Zc @ c) == i))
        res["families"][f] = {"loops_smudged": len(loops), "of": len(Zc), "smudges": len(self_raw),
                              "smudged_vs_own_clean": _q(self_raw), "two_smudges_of_one_loop": _q(twice),
                              "corrected_vs_own_clean_heldout": _q(cvs), "ridge_lambda": lam,
                              "finds_own_loop_first": {"smudged": round(float(np.mean(first_raw)), 3), "corrected": round(float(np.mean(first_cor)), 3)}}
        Zs = np.zeros_like(Zc); has = np.zeros(len(Zc), bool)
        for i in loops:
            m = np.mean(zs[i], axis=0); Zs[i] = m / (np.linalg.norm(m) + 1e-9); has[i] = True
        lib[f] = {"keep": list(map(int, L["keep"])), "mu": [float(x) for x in L["mu"]], "sd": [float(x) for x in L["sd"]], "Z": Zc.tolist(), "Zs": Zs.tolist(),
                  "has": has.tolist(), "tempo": [float(m.get("tempo") or 0) for m in L["meta"]], "W": W.tolist()}
    scenes = {sc: [r[0] for r in rows] for sc, rows in json.load(open("data/scene-records-2026.json"))["scenes"].items()}
    G = cal_gaps.remote(lib, scenes)
    res["gaps"] = G["scenes"]
    b = np.array(G["dnb_bass_best"]) if G["dnb_bass_best"] else np.zeros((0, 3))
    if len(b): res["dnb_bass"] = {"records": len(b), "far_today": round(float((b[:, 0] < 0.45).mean()), 3), "far_A": round(float((b[:, 1] < 0.45).mean()), 3),
                                  "far_B": round(float((b[:, 2] < 0.45).mean()), 3), "median_best": [round(float(np.median(b[:, k])), 3) for k in range(3)]}
    os.makedirs("data/gap-tests", exist_ok=True); json.dump(res, open("data/gap-tests/calibration.json", "w"), indent=1)
    np.savez_compressed("data/gap-tests/smudged-library.npz", **{f + "_Zs": np.array(lib[f]["Zs"], np.float16) for f in FAMS}, **{f + "_W": np.array(lib[f]["W"], np.float32) for f in FAMS})
    print("CAL " + json.dumps({k: v for k, v in res.items() if k != "gaps"}))
    for sc, d in res["gaps"].items(): print("CALGAP", sc, d["records"], {f: d[f] for f in FAMS if f in d})


# ---------------------------------------------------------------- the fit test: do split records know what goes with what?
# Step 1: predict a record's bass from its drums, melody (and vocals), artists held out whole; pick the true bass out of up
# to 100 basses from the same scene within 4% in tempo, so neither scene nor tempo can give it away. Against chance, a
# model trained on mismatched pairs (bass shuffled within scene), and the practice mixes, where parts were paired at random
# and any success is separation leakage. Step 2: does a record's fit (its bass against what its other parts predict, out of
# fold) say anything about whether it charted, or whether DJs played it, within its scene, beyond how typical the bass is?
K51 = [i for i in range(53) if i not in (45, 52)]   # level and share of energy left out: shares sum to one across parts


def _ridge(Xtr, Ytr, lam):
    import numpy as np
    return np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1], dtype=np.float32), Xtr.T @ Ytr)


def _run(Xin, Y, folds, scene, tempo, nsc, lam=50.0, shuffle=False, seed=5):
    """Out-of-fold ridge, then retrieval among same-scene, near-tempo basses in the test fold."""
    import numpy as np
    rng = np.random.default_rng(seed); N = len(Y); P = np.zeros_like(Y); top1 = top10 = cnt = 0; pct = []; chance = []
    Ys = np.zeros_like(Y)
    for k in range(5):
        tr, te = folds != k, folds == k
        mx, sx = Xin[tr].mean(0), Xin[tr].std(0) + 1e-6; my, sy = Y[tr].mean(0), Y[tr].std(0) + 1e-6
        Xt = (Xin - mx) / sx; Yt = (Y - my) / sy; Ys[te] = Yt[te]
        S1 = np.zeros((N, nsc), np.float32); S1[np.arange(N), scene] = 1.0
        A = np.hstack([Xt, S1]).astype(np.float32); Ytr = Yt[tr].copy()
        if shuffle:
            idx = np.where(tr)[0]
            for s in np.unique(scene[tr]):
                g = idx[scene[idx] == s]; Ytr[np.isin(idx, g)] = Yt[rng.permutation(g)]
        W = _ridge(A[tr], Ytr, lam); P[te] = A[te] @ W
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9); Yn = Ys / (np.linalg.norm(Ys, axis=1, keepdims=True) + 1e-9)
    for k in range(5):
        for s in np.unique(scene):
            g = np.where((folds == k) & (scene == s))[0]
            if len(g) < 20: continue
            S = Pn[g] @ Yn[g].T
            for a, i in enumerate(g):
                c = np.where((np.abs(tempo[g] - tempo[i]) / max(tempo[i], 1) <= 0.04) & (g != i))[0]
                if len(c) < 19: continue
                if len(c) > 99: c = rng.choice(c, 99, replace=False)
                r = 1 + int((S[a, c] > S[a, a]).sum()); n = len(c) + 1
                top1 += r == 1; top10 += r <= 10; cnt += 1; pct.append((r - 1) / (n - 1)); chance.append(min(10, n) / n)
    fitv = (Pn * Yn).sum(1)
    return {"tested": cnt, "top1": round(top1 / max(1, cnt), 4), "top10": round(top10 / max(1, cnt), 4), "chance_top10": round(float(np.mean(chance)) if chance else 0, 4),
            "median_rank_percentile": round(float(np.median(pct)) if pct else 0, 3)}, fitv, Ys


def _auc(score, y):
    import numpy as np
    r = np.argsort(np.argsort(score)) + 1; p = y.sum(); n = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * n) if p and n else float("nan")


def _within_scene_auc(score, y, scene, perm=0, seed=9):
    import numpy as np
    rng = np.random.default_rng(seed); ok = [s for s in np.unique(scene) if 10 <= y[scene == s].sum() <= (scene == s).sum() - 10]
    def wauc(yy):
        a = [(_auc(score[scene == s], yy[scene == s]), yy[scene == s].sum()) for s in ok]
        return float(np.sum([x * w for x, w in a]) / np.sum([w for _, w in a])) if a else float("nan")
    obs = wauc(y)
    if not perm: return obs, None, len(ok)
    null = []
    for _ in range(perm):
        yy = y.copy()
        for s in ok: m = scene == s; yy[m] = rng.permutation(yy[m])
        null.append(wauc(yy))
    null = np.array(null); return obs, float((np.abs(null - 0.5) >= abs(obs - 0.5)).mean()), len(ok)


@app.function(image=cpu_image, cpu=8.0, memory=32768, timeout=3600, volumes={"/embed": modal.Volume.from_name("sonic-embed")})
def fit_records(meta, played, practice):
    import numpy as np, zlib
    R = np.load("/embed/record-parts.npz"); at = {t: i for i, t in enumerate(R["ids"].tolist())}
    rows = [(at[m[0]], m) for m in meta if m[0] in at and float(R["tempo"][at[m[0]]]) > 0]
    V = R["V"][[r for r, _ in rows]].astype(np.float32)[:, :, K51]
    good = np.isfinite(V).all((1, 2)) & (np.abs(V).sum(2) > 0).all(1)
    V = V[good]; rows = [x for x, g in zip(rows, good) if g]
    ids = [m[0] for _, m in rows]; scene = np.array([m[1] for _, m in rows]); nsc = int(scene.max()) + 1
    tempo = np.array([float(R["tempo"][r]) for r, _ in rows], np.float32)
    folds = np.array([zlib.crc32((m[2] or m[0]).encode()) % 5 for _, m in rows]); charted = np.array([m[3] for _, m in rows], int)
    pl = set(played); dj = np.array([1 if t in pl else 0 for t in ids], int)
    D, B, M, Vo = V[:, 0], V[:, 1], V[:, 2], V[:, 3]
    out = {"records": len(ids), "scenes": len(np.unique(scene)), "charted": int(charted.sum()), "dj_played": int(dj.sum())}
    out["records_dm"], fit_dm, _ = _run(np.hstack([D, M]), B, folds, scene, tempo, nsc)
    out["records_dmv"], fit_dmv, Ys = _run(np.hstack([D, M, Vo]), B, folds, scene, tempo, nsc)
    out["records_shuffled_pairs"], _, _ = _run(np.hstack([D, M, Vo]), B, folds, scene, tempo, nsc, shuffle=True)
    # the practice mixes: parts paired at random, so any success here is separation leakage
    pr = [p for p in practice if p.get("drums") and p.get("bass") and p.get("melody")]
    PD = np.array([np.array(p["drums"], np.float32)[K51] for p in pr]); PB = np.array([np.array(p["bass"], np.float32)[K51] for p in pr]); PM = np.array([np.array(p["melody"], np.float32)[K51] for p in pr])
    pt = np.array([p["tempo"] for p in pr], np.float32); pf = np.array([p["grp"] % 5 for p in pr]); ps = np.zeros(len(pr), int)
    out["practice_mixes"] = len(pr)
    out["practice_dm_leakage"], _, _ = _run(np.hstack([PD, PM]), PB, pf, ps, pt, 1)
    # step 2: fit against what people chose, within scene, beside how typical the bass is
    typ = np.zeros(len(ids), np.float32)
    for s in np.unique(scene):
        m = scene == s; c = Ys[m].mean(0); c /= np.linalg.norm(c) + 1e-9; typ[m] = (Ys[m] / (np.linalg.norm(Ys[m], axis=1, keepdims=True) + 1e-9)) @ c
    res = fit_dmv.copy()
    for s in np.unique(scene):
        m = scene == s
        if m.sum() > 3: b = np.polyfit(typ[m], fit_dmv[m], 1); res[m] = fit_dmv[m] - np.polyval(b, typ[m])
    st = {}
    for name, y in (("charted", charted), ("dj_played", dj)):
        st[name] = {}
        for lab, sc in (("fit", fit_dmv), ("typicality", typ), ("fit_beyond_typicality", res)):
            a, p, ns = _within_scene_auc(sc, y, scene, perm=400)
            st[name][lab] = {"within_scene_auc": round(a, 4), "perm_p": round(p, 4) if p is not None else None, "scenes": ns}
    out["fit_vs_choices"] = st
    out["fit_median"] = round(float(np.median(fit_dmv)), 3)
    return out


@app.local_entrypoint()
def fit():
    import random, urllib.request, collections
    LI = _index(); rng = random.Random(23)
    meta_l = {f: [(i, m) for i, m in enumerate(LI[f]["meta"]) if m.get("tempo") and m.get("preview")] for f in FAMS}
    mixes = []
    for _ in range(1300):
        i, m = rng.choice(meta_l["bass"]); T = float(m["tempo"]); parts = {"bass": (i, m["preview"])}
        for f in ("drums", "melody") + (("vocals",) if rng.random() < 0.3 else ()):
            c = [x for x in meta_l[f] if abs(float(x[1]["tempo"]) - T) / T <= 0.03]
            if c: j, x = rng.choice(c); parts[f] = (j, x["preview"])
        if "drums" in parts and "melody" in parts: mixes.append((parts, T, i))
    R = list(mix_case.map([p for p, _, _ in mixes], return_exceptions=True))
    practice = []
    for (parts, T, gi), r in zip(mixes, R):
        if isinstance(r, dict) and not r.get("error"):
            practice.append({f: (r.get(f) or [None, None])[1] for f in FAMS} | {"tempo": T, "grp": gi})
    I = json.loads(urllib.request.urlopen("https://raw.githubusercontent.com/benmcewensignal/signal-sonic-audio/main/out/index.json", timeout=120).read())
    meta = [[t["track_id"], int(t["scene"]), ((t.get("artists") or [""])[0] or "").strip().lower(), 1 if t.get("chart_best") is not None else 0] for t in I["tracks"] if t.get("scene") is not None]
    try:
        S = json.loads(urllib.request.urlopen(urllib.request.Request("https://www.earlysignal.live/data/dj-sets.json", headers={"User-Agent": "sonic-tests"}), timeout=120).read())
        played = set(S.get("next", {}).keys())
        for k, v in S.get("next", {}).items(): played |= {x[0] for x in v}
        for st in S.get("sets", []): played |= set(st[2])
    except Exception as ex:
        print("dj sets unavailable:", ex); played = set()
    res = fit_records.remote(meta, sorted(played), practice)
    res["practice_built"] = len(practice)
    os.makedirs("data/gap-tests", exist_ok=True); json.dump(res, open("data/gap-tests/fit.json", "w"), indent=1)
    print("FIT " + json.dumps(res))


# ---------------------------------------------------------------- the loop-suggestion test
# Fresh practice mixes (new random pairings, not the ones the smudged library was built from) are unmixed; each loop's new
# smudged measurement searches (a) the clean library the site uses today and (b) the smudged library, at a tempo that fits.
# Does it find its own original? Scored on loops the smudged library covers, so both libraries are searched over the same
# candidates; drums are only partly covered, so a mixed library (smudged where there is one, clean otherwise) is scored too.
@app.local_entrypoint()
def loops():
    import random, numpy as np
    LI = _index(); rng = random.Random(31)
    SL = np.load(os.environ.get("SMUDGED", "/tmp/smudged-library.npz"))
    meta = {f: [(i, m) for i, m in enumerate(LI[f]["meta"]) if m.get("tempo") and m.get("preview")] for f in FAMS}
    mixes = []
    for _ in range(900):
        i, m = rng.choice(meta["bass"]); T = float(m["tempo"]); parts = {"bass": (i, m["preview"])}
        for f in ("drums", "melody") + (("vocals",) if rng.random() < 0.4 else ()):
            c = [x for x in meta[f] if abs(float(x[1]["tempo"]) - T) / T <= 0.03]
            if c: j, x = rng.choice(c); parts[f] = (j, x["preview"])
        if len(parts) >= 3: mixes.append(parts)
    R = [r for r in mix_case.map(mixes, return_exceptions=True) if isinstance(r, dict) and not r.get("error")]
    res = {"mixes": len(mixes), "unmixed": len(R), "families": {}}
    for f in FAMS:
        L = LI[f]; Zc = np.array(L["Z"], np.float32); Zc /= np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-9
        Zs = SL[f + "_Zs"].astype(np.float32); has = np.linalg.norm(Zs, axis=1) > 0.5
        lt = np.array([m.get("tempo") or 0 for m in L["meta"]], np.float32)
        Zmix = np.where(has[:, None], Zs, Zc)
        rk = {"clean": [], "smudged": [], "mixed_all": [], "clean_all": []}
        for r in R:
            if f not in r or not r[f][1]: continue
            i, v = r[f]; zq = _z(v, L); ok = _fits(lt, float(L["meta"][i].get("tempo") or 0))
            if not ok[i]: continue
            def rank(Z, mask): s = Z @ zq; return 1 + int((s[mask] > s[i]).sum())
            rk["clean_all"].append(rank(Zc, ok)); rk["mixed_all"].append(rank(Zmix, ok))
            if has[i]:
                m2 = ok & has; rk["clean"].append(rank(Zc, m2)); rk["smudged"].append(rank(Zs, m2))
        def summ(a):
            a = np.array(a)
            return {"n": int(len(a)), "first": round(float((a == 1).mean()), 3), "top3": round(float((a <= 3).mean()), 3), "top10": round(float((a <= 10).mean()), 3), "median_rank": float(np.median(a))} if len(a) else {"n": 0}
        cands = [int(_fits(lt, float(m.get("tempo") or 0)).sum()) for m in L["meta"][:200]]
        res["families"][f] = {k: summ(v) for k, v in rk.items()} | {"library": len(Zc), "smudged_entries": int(has.sum()), "typical_candidates_at_tempo": int(np.median(cands))}
    os.makedirs("data/gap-tests", exist_ok=True); json.dump(res, open("data/gap-tests/loops.json", "w"), indent=1)
    print("LOOPS " + json.dumps(res))
