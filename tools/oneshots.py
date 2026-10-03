"""Public-domain drum one-shots for the scene groovebox: Freesound, Creative Commons 0 only (no credit needed), short single
hits, measured for brightness (spectral centroid), punch (peak over early loudness, and how fast the hit peaks), decay and
how many hits a file holds; rolls, loops, kicks without low end and dull hats are dropped. Each survivor is trimmed,
levelled and encoded small. Output: a pool with measures and audio, from which kits are matched to each scene's measured
character elsewhere.
  FREESOUND_KEY=... python tools/oneshots.py out.json"""
import base64, io, json, os, subprocess, sys, tempfile, time, requests, numpy as np, librosa, soundfile as sf
KEY = os.environ["FREESOUND_KEY"]; OUT = sys.argv[1]; SR = 44100
KINDS = {"kick": (["kick drum one shot", "909 kick", "808 kick", "techno kick", "house kick", "kick sample"], (0.08, 1.5), 0.7),
         "clap": (["clap one shot", "909 clap", "handclap sample", "clap sample"], (0.05, 1.5), 0.6),
         "snare": (["snare one shot", "breakbeat snare", "dnb snare", "snare drum sample"], (0.05, 1.5), 0.55),
         "hat": (["closed hihat", "closed hi hat one shot", "909 hihat closed", "hi hat closed"], (0.02, 0.8), 0.25),
         "openhat": (["open hihat", "open hi hat one shot", "909 open hat"], (0.1, 2.5), 0.9),
         "shaker": (["shaker one shot", "shaker hit"], (0.03, 1.0), 0.35),
         "rim": (["rimshot", "rim click", "rim shot one shot"], (0.02, 1.0), 0.35)}
pool = {}
for kind, (queries, (dmin, dmax), keep_s) in KINDS.items():
    seen = {}
    for q in queries:
        for page in (1, 2):
            try:
                r = requests.get("https://freesound.org/apiv2/search/text/", timeout=40, params={"query": q, "filter": f'duration:[{dmin} TO {dmax}] license:"Creative Commons 0"',
                                 "fields": "id,name,username,license,previews,duration,tags", "page_size": 100, "page": page, "token": KEY})
                if r.status_code != 200: break
                for it in r.json().get("results", []): seen.setdefault(it["id"], it)
            except Exception: break
            time.sleep(0.4)
    print(f"{kind}: {len(seen)} candidates", flush=True); kept = 0
    for sid, it in list(seen.items())[:220]:
        if kept >= 70: break
        url = (it.get("previews") or {}).get("preview-hq-mp3")
        if not url: continue
        try:
            raw = requests.get(url, timeout=30).content; y, _ = librosa.load(io.BytesIO(raw), sr=SR, mono=True)
        except Exception: continue
        if len(y) < SR * 0.03 or np.max(np.abs(y)) < 1e-3: continue
        yt, idx = librosa.effects.trim(y, top_db=40); y = y[idx[0]:]   # keep the decay, drop leading silence
        on = librosa.onset.onset_detect(y=y, sr=SR, units="time", backtrack=False)
        if len([t for t in on if t > 0.05]) > 1: continue          # rolls and loops
        pk = int(np.argmax(np.abs(y[:int(SR * 0.08)])))
        early = y[:int(SR * 0.1)]; crest = float(np.max(np.abs(early)) / (np.sqrt(np.mean(early ** 2)) + 1e-9))
        S = np.abs(librosa.stft(y[:int(SR * 0.25)], n_fft=2048, hop_length=512)); freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
        cent = float(np.sum(freqs[:, None] * S) / (np.sum(S) + 1e-9)); low = float(S[freqs < 150].sum() / (S.sum() + 1e-9))
        envl = np.abs(y); dec = int(np.argmax(envl < np.max(envl) * 0.03)) / SR if np.any(envl < np.max(envl) * 0.03) else len(y) / SR
        if kind == "kick" and low < 0.35: continue
        if kind in ("hat", "openhat", "shaker") and cent < 4000: continue
        if kind in ("clap", "snare", "rim") and not (800 < cent < 9000): continue
        n = min(len(y), int(SR * keep_s)); seg = y[:n] / (np.max(np.abs(y[:n])) + 1e-9) * 0.89; fade = min(int(SR * 0.012), n // 4); seg[-fade:] *= np.linspace(1, 0, fade)
        w = tempfile.mkdtemp(); wav = os.path.join(w, "s.wav"); mp3 = os.path.join(w, "s.mp3"); sf.write(wav, seg, SR)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-i", wav, "-ac", "1", "-b:a", "96k", mp3], check=True)
        pool[f"fs{sid}"] = {"kind": kind, "name": it.get("name"), "user": it.get("username"), "license": it.get("license"), "page": f"https://freesound.org/people/{it.get('username')}/sounds/{sid}/",
                            "centroid": round(cent), "crest": round(crest, 2), "attack_ms": round(pk / SR * 1000, 1), "decay_s": round(dec, 3), "low": round(low, 3),
                            "mp3": base64.b64encode(open(mp3, "rb").read()).decode()}
        kept += 1
    print(f"  {kind}: kept {kept}", flush=True)
json.dump(pool, open(OUT, "w")); print("POOL", len(pool), "samples,", round(os.path.getsize(OUT) / 1e6, 2), "MB")
