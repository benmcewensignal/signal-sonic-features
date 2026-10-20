"""Split once: does the full reading come out the same when it measures Swap's split of the whole track instead of
separating the middle minute again? For each track in tools/split_once_tracks.json (a Beatport preview from each of
32 scenes): the track split whole, as Swap splits it (worker/swap.py); its middle minute cut as the phone cuts it for a
reading (worker/swap.py reading_inputs); then the reading run twice on that minute (worker/parts.py read): as today,
separating the minute itself, and from the whole-track split's same minute. Annotations with the agreement; the pairs
of readings (their summaries) in out/split-once.json.
  modal run tools/split_once_check.py"""
import json, modal

app = modal.App("sonic-split-once-check")
img = (modal.Image.debian_slim(python_version="3.12")
       .apt_install("ffmpeg", "libsndfile1")
       .pip_install("numpy<2", "librosa==0.10.2", "soundfile", "demucs==4.0.1", "torch==2.3.1", "torchaudio==2.3.1",
                    "essentia-tensorflow", "fastapi[standard]", "scikit-learn==1.8.0", "scipy")
       .add_local_python_source("features")
       .add_local_dir("worker", "/root/worker"))
vol = modal.Volume.from_name("sonic-embed", create_if_missing=True)
PARTS = ("drums", "bass", "other", "vocals")
PLAIN = ("level", "crest", "centroid_hz", "flatness", "onsets_per_s", "share_of_energy")


def summary(r):
    """What the report is built from, kept small: the ear's call and place, the scene call, key, tempo, each part's
    share and plain measures and 53 numbers, which part pulls away."""
    r = r or {}; L = r.get("scene_learned") or {}; out = {"error": r.get("error")}
    out["learned"] = (L.get("scenes") or [])[:3]; out["ear"] = (L.get("ear") or {}).get("v")
    out["scene"] = ((r.get("scene") or {}).get("scenes") or [])[:3]
    out["key"] = (r.get("key") or {}).get("key"); out["tempo"] = r.get("tempo_read")
    P = r.get("parts") or {}
    out["plain"] = {p: {c: (P.get(p) or {}).get(c) for c in PLAIN} for p in PARTS}
    out["vectors"] = r.get("part_vectors")
    pr = {k: v.get("pct") for k, v in (r.get("part_residuals") or {}).items() if isinstance(v, dict)}
    out["pulls"] = max(pr, key=pr.get) if pr else None   # the part furthest from its scene's typical part
    out["residuals"] = pr
    return out


@app.function(image=img, gpu="T4", cpu=4.0, memory=12288, timeout=1200, volumes={"/embed": vol}, max_containers=12, retries=1)
def both(t):
    import os, sys, tempfile, shutil, urllib.request
    sys.path.insert(0, "/root")
    from worker import swap as SW, parts as PA
    from features import stems as S
    d = tempfile.mkdtemp()
    try:
        src = os.path.join(d, "src.mp3"); open(src, "wb").write(urllib.request.urlopen(t["url"], timeout=60).read())
        wav = os.path.join(d, "track.wav"); SW.decode_to_wav(src, wav)
        st = S.separate(wav, os.path.join(d, "sep"))   # the whole track, as Swap splits it
        X = {k: SW.read_wav(st[k])[0] for k in PARTS}
        n = min(len(v) for v in X.values()); X = {k: v[:n] for k, v in X.items()}
        clip, sb = SW.reading_inputs(wav, X)
        cp = os.path.join(d, "clip.wav"); open(cp, "wb").write(clip)
        old = PA.read(cp, False)                       # today: the minute separated on its own
        sp = {}
        for k, b in sb.items(): p = os.path.join(d, f"s_{k}.wav"); open(p, "wb").write(b); sp[k] = p
        new = PA.read(cp, False, stems=sp)              # split once: the whole-track split's same minute
        return {"id": t["id"], "scene": t["scene"], "old": summary(old), "new": summary(new)}
    except Exception as e:
        return {"id": t["id"], "scene": t["scene"], "error": f"{type(e).__name__}: {str(e)[:200]}"}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def agreement(rows):
    import numpy as np, statistics as st_
    ok = [r for r in rows if not r.get("error") and not r["old"].get("error") and not r["new"].get("error")]
    def same(f): return round(sum(1 for r in ok if f(r["old"]) == f(r["new"]) and f(r["old"]) is not None) / max(1, len(ok)), 3)
    def cos(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float); na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na and nb else None
    ear = [cos(r["old"]["ear"], r["new"]["ear"]) for r in ok if r["old"].get("ear") and r["new"].get("ear")]
    shares = [abs((r["old"]["plain"][p]["share_of_energy"] or 0) - (r["new"]["plain"][p]["share_of_energy"] or 0)) for r in ok for p in PARTS]
    rel = {}
    for c in ("level", "crest", "centroid_hz", "flatness", "onsets_per_s"):
        v = []
        for r in ok:
            for p in PARTS:
                a, b = r["old"]["plain"][p].get(c), r["new"]["plain"][p].get(c)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a) > 1e-9: v.append(abs(b - a) / abs(a))
        rel[c] = round(st_.median(v), 3) if v else None
    vec = {}
    for p in PARTS:
        v = [cos(r["old"]["vectors"][p][:45], r["new"]["vectors"][p][:45]) for r in ok if (r["old"].get("vectors") or {}).get(p) and (r["new"].get("vectors") or {}).get(p)]
        v = [x for x in v if x is not None]
        vec[p] = round(st_.median(v), 4) if v else None
    return {"tracks": len(ok), "of": len(rows), "learned_top_same": same(lambda s: (s["learned"] or [[None]])[0][0]),
            "scene_call_top_same": same(lambda s: (s["scene"] or [[None]])[0][0]), "key_same": same(lambda s: s["key"]),
            "tempo_same_within_1": round(sum(1 for r in ok if r["old"]["tempo"] and r["new"]["tempo"] and abs(r["old"]["tempo"] - r["new"]["tempo"]) <= 1) / max(1, len(ok)), 3),
            "pulls_same": same(lambda s: s["pulls"]),
            "ear_cosine_median": round(st_.median(ear), 4) if ear else None, "ear_cosine_min": round(min(ear), 4) if ear else None,
            "share_abs_diff_median": round(st_.median(shares), 4) if shares else None, "share_abs_diff_p90": round(sorted(shares)[int(0.9 * (len(shares) - 1))], 4) if shares else None,
            "plain_rel_diff_median": rel, "part_embedding_cosine_median": vec}


@app.local_entrypoint()
def main():
    import os
    tracks = json.load(open("tools/split_once_tracks.json"))
    rows = [r for r in both.map(tracks, return_exceptions=True, order_outputs=False) if isinstance(r, dict)]
    A = agreement(rows)
    os.makedirs("out", exist_ok=True)
    json.dump({"note": "split once: the full reading on the middle minute, separated on its own (old) and from the whole track's split (new)",
               "agreement": A, "rows": rows}, open("out/split-once.json", "w"))
    print("::notice title=split once agreement::" + json.dumps(A), flush=True)
    for r in rows:
        if r.get("error"): print(f"::warning title=split once {r['id']}::{r['error']}", flush=True)
