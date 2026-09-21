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


def rhythm_of_stem(path):
    """Swing and pulse from a learned beat tracker, on the drum part alone.

    Swing failed three times on onset autocorrelation and was written off. A learned tracker
    reads a synthetic beat correctly on the first try: offbeat position 0.49 straight, 0.56,
    0.63, 0.73 as the swing is turned up. On the isolated drum stem, with no bass or synth to
    confuse the onsets, it should do better still. Two numbers: where the offbeat falls
    between beats, 0.5 straight and 0.67 full triplet, and how sure the tracker was.
    """
    try:
        import numpy as np, librosa
        from essentia.standard import BeatTrackerMultiFeature, MonoLoader
        y = MonoLoader(filename=path, sampleRate=44100)()
        if len(y) < 44100 * 4:
            return None
        beats, conf = BeatTrackerMultiFeature()(y)
        if len(beats) < 8:
            return None
        on = librosa.onset.onset_detect(y=y, sr=44100, units="time")
        # Swing is where the offbeat falls, so look for the onset nearest the half-beat, inside
        # the window a swung eighth can occupy. The first version took the first onset after
        # each beat, which on real drums with sixteenth hats is the sixteenth at a quarter:
        # it read drum and bass at 0.31 and hard techno at 0.51, which is hat density, not
        # swing. Density is worth keeping, so it is reported separately and named for what it is.
        fr, sub = [], []
        for a, b in zip(beats[:-1], beats[1:]):
            span = b - a
            inbeat = [(o - a) / span for o in on if a < o < b]
            sub.append(len(inbeat))
            # the window an offbeat eighth can occupy, straight at 0.5 to full triplet at 0.67;
            # it stops short of 0.75 so a trailing sixteenth cannot stand in for it
            near = [p for p in inbeat if 0.42 <= p <= 0.71]
            if near:
                fr.append(float(np.median(near)))
        out = {"beat_confidence": round(float(conf), 3),
               "beats_per_minute": round(60.0 / float(np.median(np.diff(beats))), 2),
               "hits_per_beat": round(float(np.median(sub)), 2) if sub else None}
        if len(fr) >= 4:
            out["swing"] = round(float(np.median(fr)), 4)
        return out
    except Exception:
        return None


def breakdowns_of_stem(path, bpm=None):
    """Where the kick goes and comes back, on the drum part alone.

    Segmentation failed three times on the mix because a quieter passage of the same music and
    a different passage read alike. On the isolated drums the question is simpler: is the kick
    there or not. Low band energy, smoothed over a bar, runs shorter than two bars folded into
    their neighbours. On synthetic records it finds every breakdown edge and nothing else, and
    it does not claim to find a drop against a groove, which is a matter of degree it cannot
    see. Four numbers: how many breakdowns, what share of the record they are, the longest one
    in bars, and how far in the first one starts.
    """
    try:
        import numpy as np, librosa
        y, sr = librosa.load(path, sr=22050, mono=True)
        if len(y) < sr * 20:
            return None
        if not bpm:
            bpm = float(librosa.beat.tempo(y=y, sr=sr)[0]) or 125.0
        bpm = max(60.0, min(200.0, bpm))
        hop = 512
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        low = S[(freqs >= 40) & (freqs <= 120)].sum(0)
        bar_frames = max(4, int(4 * 60 / bpm * sr / hop))
        env = np.convolve(low, np.ones(bar_frames) / bar_frames, mode="same")
        env = env / (env.max() + 1e-9)
        on = env > 0.4
        runs, i = [], 0
        while i < len(on):
            j = i
            while j < len(on) and on[j] == on[i]:
                j += 1
            runs.append([i, j, bool(on[i])]); i = j
        minlen, changed = 2 * bar_frames, True
        while changed and len(runs) > 1:
            changed = False
            for k, r in enumerate(runs):
                if r[1] - r[0] < minlen:
                    if k > 0:
                        runs[k - 1][1] = r[1]
                    else:
                        runs[k + 1][0] = r[0]
                    runs.pop(k); changed = True; break
        offs = [r for r in runs if not r[2]]
        total = len(on)
        if not offs:
            return {"breakdowns": 0, "breakdown_share": 0.0}
        longest = max(r[1] - r[0] for r in offs) / bar_frames
        return {"breakdowns": len(offs),
                "breakdown_share": round(sum(r[1] - r[0] for r in offs) / total, 4),
                "longest_breakdown_bars": round(float(longest), 1),
                "first_breakdown_at": round(offs[0][0] / total, 4)}
    except Exception:
        return None


def analyse_stem(path):
    """The full analyser, on one separated part.

    Eight summary statistics per stem already beat the entire forty-five number hand-named set
    on genre, 32.2% against 22.9%, and add twelve points on top of it. Those eight are the
    residue of a neural separation whose output we then throw away. This runs the same analyser
    the mix gets, on each part, so a drum stem is described as fully as a record is.
    """
    try:
        from features.analyser_local import LocalAnalyser
    except Exception:
        return None
    try:
        a = LocalAnalyser()
        fv = a.analyse(path)
        e = getattr(fv, "embedding", None)
        if e is None and isinstance(fv, dict):
            e = fv.get("embedding")
        return {"embedding": [round(float(x), 5) for x in e]} if e is not None and len(e) == 45 else None
    except Exception:
        return None


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



def remeasure_todo(out_dir, limit, db=None):
    """Records already separated that carry no embedding anywhere, a scene at a time.

    Two faults in the version before this. It looked only at the first copy of each record, which
    is always the original without the embedding, so every pass redid the same records: 8,227
    remeasured lines were 1,641 unique records, 1,510 of them done five times. And it took them
    in shard-file order, which is mostly two scenes, so nothing could be tested until the end.
    """
    import glob as _g, collections as _c
    finished, seen, order = set(), set(), []
    for f in sorted(_g.glob(os.path.join(out_dir, "stems-*.jsonl"))):
        for line in open(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("track_id")
            if not t:
                continue
            st = d.get("stems") or {}
            if not isinstance(st, dict) or not st:
                continue
            # Finished means carrying everything the remeasure now writes, not only the first
            # thing it wrote. Swing and breakdowns were added after 3,261 records had their
            # forty-five, and an embedding-only test would have called those done and skipped
            # them for good, leaving the new measures on every record but the first three
            # thousand. The drum part is the one that carries all three.
            dr = st.get("drums") if isinstance(st.get("drums"), dict) else {}
            if dr.get("embedding") and "breakdowns" in dr and "beat_confidence" in dr:
                finished.add(t)
            if t not in seen:
                seen.add(t)
                order.append(t)
    need = [t for t in order if t not in finished]
    scene = {}
    if db:
        try:
            import sqlite3 as _s
            for t, sc in _s.connect(db).execute(
                    "select track_id, scene from track_scenes where week like '____-M__'"):
                scene.setdefault(t, sc)
        except Exception:
            scene = {}
    if scene:
        pools = _c.defaultdict(list)
        for t in need:
            pools[scene.get(t, "?")].append(t)
        keys, rot = sorted(pools), []
        while any(pools[k] for k in keys):
            for k in keys:
                if pools[k]:
                    rot.append(pools[k].pop(0))
        need = rot
    return need[:limit], max(0, len(need) - limit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="sonic.db"); ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--budget-minutes", type=int, default=80); ap.add_argument("--out-dir", default="out")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    # Separation is done on 48,945 records and each part carries eight summary numbers. Those
    # eight already beat the whole forty-five number hand-named set on genre, which is why it
    # is worth going back over them with the real analyser rather than only doing the 287 that
    # are still unseparated. This mode ignores the picker and re-measures what is already in
    # the shard files, writing the embedding alongside the eight.
    ap.add_argument("--remeasure", action="store_true")
    a = ap.parse_args()
    # The workflow sets this rather than branching in the shell: a conditional around a
    # GitHub expression is the kind of line that passes review and fails at run time.
    if os.environ.get("STEMS_REMEASURE") == "1":
        a.remeasure = True
    if a.self_test: return self_test(a.out_dir)
    ok, err = _ensure()
    if not ok:
        print("stems: demucs unavailable:", err, flush=True); return
    from .beatport import get_token, _get
    have = already(a.out_dir)
    if a.remeasure:
        todo, remaining = remeasure_todo(a.out_dir, a.limit * max(1, a.of), a.db)
        print(f"remeasure: {len(todo)} this pass, {remaining} still on the eight numbers alone", flush=True)
    else:
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
                # and the full analyser on each part, while the audio is still on disk
                if os.environ.get("STEM_EMBED", "1") != "0":
                    for k, v in stems.items():
                        emb = analyse_stem(v)
                        if emb and isinstance(rec["stems"].get(k), dict):
                            rec["stems"][k].update(emb)
                        if k == "drums" and isinstance(rec["stems"].get(k), dict):
                            rh = rhythm_of_stem(v)
                            if rh:
                                rec["stems"][k].update(rh)
                            bd = breakdowns_of_stem(v, (rh or {}).get("beats_per_minute"))
                            if bd:
                                rec["stems"][k].update(bd)
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
