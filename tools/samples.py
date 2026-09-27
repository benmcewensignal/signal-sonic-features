"""Sample matching, first part: openly licensed drum loops from Freesound, measured as Sonic measures a
separated drum part, matched to each scene's drums.  FS_KEY in the environment.
  python tools/samples.py --pages 14 --out sample-matches.json
"""
import os, sys, json, argparse, tempfile, subprocess, numpy as np, requests
from multiprocessing import Pool
sys.path.insert(0, ".")
SC = ("level", "crest", "dynamic_span", "centroid_hz", "rolloff_hz", "flatness", "onsets_per_s", "share_of_energy")
KEY = os.environ.get("FS_KEY", "")

def search(pages):
    out, seen = [], set()
    for q in ("drum loop", "drums loop techno", "house drum loop", "breakbeat loop", "percussion loop"):
        for p in range(1, pages + 1):
            for attempt in range(6):   # Freesound limits requests per minute: wait and retry when told to slow down
                r = requests.get("https://freesound.org/apiv2/search/text/", timeout=40, params={
                "query": q, "filter": 'duration:[2.0 TO 30.0] license:("Creative Commons 0" OR "Attribution")',
                    "fields": "id,name,username,license,previews,tags,duration,ac_analysis", "page_size": 150, "page": p, "token": KEY})
                if r.status_code != 429: break
                import time; time.sleep(15 + 10 * attempt)
            if r.status_code != 200: print("search stopped:", q, p, r.status_code, flush=True); break
            d = r.json()
            for x in d.get("results", []):
                tags = set(x.get("tags") or [])
                if x["id"] in seen or not (tags & {"drums", "drum", "drum-loop", "drumloop", "beat", "breakbeat", "percussion", "loop"}): continue
                seen.add(x["id"]); out.append(x)
            if not d.get("next"): break
    return out

def measure(x):
    try:
        from features import stems as S   # loaded only where loops are measured
        url = (x.get("previews") or {}).get("preview-hq-mp3")
        if not url: return None
        a = requests.get(url, timeout=40).content
        with tempfile.TemporaryDirectory() as t:
            mp3, wav = os.path.join(t, "a.mp3"), os.path.join(t, "a.wav")
            open(mp3, "wb").write(a)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-i", mp3, "-ar", "44100", "-ac", "2", wav], check=True, timeout=60)
            m = S.measure_stem(wav) or {}; e = S.analyse_stem(wav) or {}; m.update(e)
        emb = m.get("embedding")
        if not (isinstance(emb, list) and len(emb) == 45): return None
        v = [float(z) for z in emb] + [float(m.get(k)) if isinstance(m.get(k), (int, float)) else 0.0 for k in SC]
        tempo = ((x.get("ac_analysis") or {}).get("ac_tempo"))
        return {"id": x["id"], "name": x["name"], "user": x["username"], "license": x["license"], "preview": url, "tempo": tempo, "duration": x.get("duration"), "v": v}
    except Exception as ex:
        return None

def measure_stage(a):
    """One shard: every loop whose Freesound id falls to this shard, measured until the time budget runs out."""
    import time, glob
    if not KEY: sys.exit("no Freesound key")
    found = [x for x in search(a.pages) if x["id"] % a.of == a.shard][:a.max]
    print(f"shard {a.shard}: {len(found)} loops to measure", flush=True)
    rows, t0 = [], time.time()
    pool = Pool(4)
    try:
        for i, r in enumerate(pool.imap_unordered(measure, found, chunksize=2)):
            if r: rows.append(r)
            if i % 25 == 0:
                print(f"measured {i} of {len(found)} in {int(time.time() - t0)} s", flush=True)
                json.dump(rows, open(f"loops-{a.shard}.json", "w"))   # saved as it goes
            if time.time() - t0 > a.budget * 60: print("time budget reached; keeping what is measured", flush=True); break
    finally:
        pool.terminate()
    json.dump(rows, open(f"loops-{a.shard}.json", "w")); print(f"shard {a.shard}: {len(rows)} loops measured", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pages", type=int, default=14); ap.add_argument("--max", type=int, default=2000); ap.add_argument("--out", default="sample-matches.json")
    ap.add_argument("--stage", default="all"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--of", type=int, default=1); ap.add_argument("--budget", type=float, default=80)
    a = ap.parse_args()
    if a.stage == "measure": return measure_stage(a)
    if a.stage == "match":
        import glob
        rows = [r for f in sorted(glob.glob("loops-*.json")) for r in json.load(open(f))]
        print(f"loops measured across shards: {len(rows)}", flush=True)
    else:
        if not KEY: sys.exit("no Freesound key")
        found = search(a.pages)[:a.max]; print(f"loops found: {len(found)}", flush=True)
        with Pool(4) as pool: rows = [r for r in pool.map(measure, found) if r]
    print(f"loops measured: {len(rows)}", flush=True)
    P = json.load(open("data/drum-profiles.json")); keep = P["keep"]; mu, sd = np.array(P["mu"]), np.array(P["sd"])
    V = np.array([np.array(r["v"])[keep] for r in rows]); Z = (V - mu) / sd; Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    out = {"part": "drums", "loops": len(rows), "source": "Freesound (Creative Commons 0 and Attribution)", "scenes": {}}
    for s, m in P["scenes"].items():
        sim = Z @ np.array(m); order = np.argsort(-sim)[:25]
        out["scenes"][s] = [{k: rows[i][k] for k in ("id", "name", "user", "license", "preview", "tempo", "duration")} | {"sim": round(float(sim[i]), 3)} for i in order]
    json.dump(out, open(a.out, "w"), separators=(",", ":"))
    for s in ("drum-and-bass", "techno-peak-time", "deep-house", "amapiano"):
        if s in out["scenes"]: print(f"::notice title=matches {s}::" + "; ".join(f"{r['name'][:40]} ({round(r['tempo']) if r['tempo'] else '?'} BPM)" for r in out["scenes"][s][:5]))
    tempos = {s: [r["tempo"] for r in v[:25] if r["tempo"]] for s, v in out["scenes"].items()}
    print("::notice title=median tempo of each scene's matches::" + json.dumps({s: round(float(np.median(t))) for s, t in tempos.items() if t}))

if __name__ == "__main__": main()
