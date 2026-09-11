"""Separate each record into stems, then measure each one.

Nine tenths of a scene's movement is production texture: we can name its direction —
fuller mid-range, louder, darker — but not its size, because the named ingredients
explain almost none of it. Every attempt so far has failed for the same reason. We
measure a finished mix, and a mix is four decisions averaged together.

Demucs separates a record into drums, bass, vocals and everything else. Measuring each
stem on its own turns "texture" into four things that can be attributed:

  drums   how hard, how dense, how much room
  bass    weight, and where it sits
  vocals  present or not, wet or dry, how far forward
  other   the synth and chord layer, which is where most of the residual probably lives

One question decides whether this was worth it: the field got less dynamic, on broadcast
loudness as well as our own curve. Which stem flattened?

Results go to out/stems-*.jsonl. This never writes sonic.db.

  python -m features.stems --db sonic.db --limit 60 --budget-minutes 80
  python -m features.stems --self-test          proves the install and reports timing
"""
import argparse, collections, json, os, random, subprocess, sqlite3, sys, tempfile, time, urllib.request

MODEL = "htdemucs"
SR = 44100


def _ensure():
    try:
        import demucs.separate  # noqa
        import torch  # noqa
        return True, None
    except Exception:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "demucs"],
                           capture_output=True, text=True)
        if r.returncode:
            return False, (r.stderr or r.stdout)[-300:]
        try:
            import demucs.separate  # noqa
            return True, None
        except Exception as e:
            return False, repr(e)[:300]


def separate(path, outdir):
    """Run demucs on one file; returns {stem: wav path}."""
    cmd = [sys.executable, "-m", "demucs.separate", "-n", MODEL, "--two-stems", None,
           "-o", outdir, path]
    cmd = [c for c in cmd if c is not None and c != "--two-stems"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    base = os.path.splitext(os.path.basename(path))[0]
    d = os.path.join(outdir, MODEL, base)
    if not os.path.isdir(d):
        raise RuntimeError("demucs produced nothing: " + (r.stderr or r.stdout or "")[-160:])
    return {os.path.splitext(f)[0]: os.path.join(d, f) for f in os.listdir(d) if f.endswith(".wav")}


def measure_stem(path):
    import numpy as np, librosa
    y, sr = librosa.load(path, sr=SR, mono=True)
    if len(y) < sr: return None
    rms = librosa.feature.rms(y=y)[0]
    peak = float(np.max(np.abs(y)))
    if peak <= 0: return {"silent": True}
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    cent = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
    flat = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    roll = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr)))
    onset = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    q = np.percentile(rms, [10, 50, 90])
    return {"level": round(float(np.mean(rms)), 5),
            "crest": round(float(peak / (np.mean(rms) + 1e-9)), 3),
            "dynamic_span": round(float(q[2] - q[0]), 5),
            "centroid_hz": round(cent, 1), "rolloff_hz": round(roll, 1),
            "flatness": round(flat, 5), "onsets_per_s": round(len(onset) / (len(y) / sr), 3),
            "share_of_energy": None}


def pick(db, have, limit):
    """One record per scene-month, round robin. A sweep in insertion order samples
    whichever wave was ingested last, which has bitten three modules already."""
    c = sqlite3.connect(db); c.row_factory = sqlite3.Row
    pool = collections.defaultdict(list)
    for r in c.execute("""select ts.scene, ts.week, ts.track_id from track_scenes ts
                          join tracks t on t.track_id=ts.track_id and t.analyser_id='local'
                          where ts.week like '____-M__' and ts.track_id like 'bp:%'"""):
        if r["track_id"] not in have: pool[(r["scene"], r["week"])].append(r["track_id"])
    keys = sorted(pool); rr = random.Random(4)
    for k in keys: rr.shuffle(pool[k])
    todo = []
    while len(todo) < limit and any(pool[k] for k in keys):
        for k in keys:
            if pool[k] and len(todo) < limit: todo.append(pool[k].pop())
    return todo, sum(len(v) for v in pool.values())


def already(out_dir):
    have = set()
    if not os.path.isdir(out_dir): return have
    for f in os.listdir(out_dir):
        if f.startswith("stems-") and f.endswith(".jsonl"):
            for line in open(os.path.join(out_dir, f)):
                try: have.add(json.loads(line)["track_id"])
                except Exception: pass
    return have


def self_test(out_dir):
    """Prove the install and report timing into a file, since logs are not readable."""
    import numpy as np, soundfile as sf
    os.makedirs(out_dir, exist_ok=True)
    rep = {"ran": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    ok, err = _ensure()
    rep["install_ok"] = ok
    if not ok:
        rep["install_error"] = err
        json.dump(rep, open(os.path.join(out_dir, "stems-selftest.json"), "w"), indent=1); print(json.dumps(rep, indent=1)); return
    t = np.arange(int(SR * 12)) / SR
    y = (0.4 * np.sin(2 * np.pi * 55 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    step = int(SR * 60 / 128)
    for i in range(0, len(y) - int(SR * 0.05), step):
        k = int(SR * 0.05); y[i:i+k] += (np.sin(2*np.pi*50*np.arange(k)/SR) * np.exp(-np.arange(k)/(SR*0.01))).astype("float32")
    d = tempfile.mkdtemp(); src = os.path.join(d, "test.wav")
    sf.write(src, y, SR)
    t0 = time.time()
    try:
        stems = separate(src, d)
        rep["stems"] = sorted(stems)
        rep["seconds_for_12s_of_audio"] = round(time.time() - t0, 1)
        rep["measured"] = {k: measure_stem(v) for k, v in list(stems.items())[:1]}
        rep["ok"] = True
    except Exception as e:
        rep["ok"] = False; rep["error"] = f"{type(e).__name__}: {str(e)[:220]}"
    json.dump(rep, open(os.path.join(out_dir, "stems-selftest.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="sonic.db"); ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--budget-minutes", type=int, default=80); ap.add_argument("--out-dir", default="out")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()
    if a.self_test: return self_test(a.out_dir)
    ok, err = _ensure()
    if not ok:
        print("stems: demucs unavailable:", err, flush=True); return
    from .beatport import get_token, _get
    have = already(a.out_dir)
    todo, remaining = pick(a.db, have, a.limit * max(1, a.of))
    if a.of > 1:
        # split by record, not by scene: stem separation is per record and the scenes are
        # very uneven, so a scene split would leave shards idle while one grinds on.
        todo = [t for i, t in enumerate(todo) if i % a.of == a.shard]
        print(f"shard {a.shard} of {a.of}: {len(todo)} records", flush=True)
    print(f"stems: {len(have)} done, {len(todo)} this run, {remaining} left after", flush=True)
    if not todo: return
    token = get_token()
    n = len([f for f in os.listdir(a.out_dir) if f.startswith("stems-")]) if os.path.isdir(a.out_dir) else 0
    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"-s{a.shard}" if a.of > 1 else ""
    path = os.path.join(a.out_dir, f"stems-{n:03d}{tag}.jsonl")
    t0 = time.time(); done = err_n = 0
    with open(path, "w") as out:
        for tid in todo:
            if (time.time() - t0) / 60 > a.budget_minutes:
                print("budget reached", flush=True); break
            work = None
            try:
                d = _get(f"/catalog/tracks/{tid.split(':')[-1]}/", token)
                url = d.get("sample_url") or (d.get("preview") or {}).get("mp3", {}).get("url")
                if not url: raise ValueError("no preview url")
                work = tempfile.mkdtemp()
                src = os.path.join(work, "a.mp3")
                req = urllib.request.Request(url, headers={"User-Agent": "signal-sonic-features/stems"})
                with urllib.request.urlopen(req, timeout=30) as r, open(src, "wb") as f: f.write(r.read())
                stems = separate(src, work)
                rec = {"track_id": tid, "model": MODEL,
                       "stems": {k: measure_stem(v) for k, v in stems.items()}}
                tot = sum((s or {}).get("level", 0) for s in rec["stems"].values()) or 1
                for k, s in rec["stems"].items():
                    if s: s["share_of_energy"] = round((s.get("level", 0)) / tot, 4)
                out.write(json.dumps(rec) + "\n"); done += 1
                if done % 10 == 0: out.flush(); print(f"  {done}/{len(todo)}, {err_n} failed", flush=True)
            except Exception as e:
                err_n += 1
                if err_n <= 3: print(f"  {tid}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            finally:
                if work: subprocess.run(["rm", "-rf", work], capture_output=True)
    print(f"stems: {done} separated, {err_n} failed, written to {path}", flush=True)


if __name__ == "__main__":
    main()
