"""Sample matching, first part: openly licensed drum loops from Freesound, measured as Sonic measures a
separated drum part, matched to each scene's drums.  FS_KEY in the environment.
  python tools/samples.py --pages 14 --out sample-matches.json
"""
import os, sys, json, argparse, tempfile, subprocess, numpy as np, requests
from multiprocessing import Pool
sys.path.insert(0, ".")
from features import stems as S
SC = ("level", "crest", "dynamic_span", "centroid_hz", "rolloff_hz", "flatness", "onsets_per_s", "share_of_energy")
KEY = os.environ.get("FS_KEY", "")

def search(pages):
    out, seen = [], set()
    for q in ("drum loop", "drums loop techno", "house drum loop", "breakbeat loop", "percussion loop"):
        for p in range(1, pages + 1):
            r = requests.get("https://freesound.org/apiv2/search/text/", timeout=40, params={
                "query": q, "filter": 'duration:[2.0 TO 30.0] license:("Creative Commons 0" OR "Attribution")',
                "fields": "id,name,username,license,previews,tags,duration,ac_analysis", "page_size": 150, "page": p, "token": KEY})
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

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pages", type=int, default=14); ap.add_argument("--max", type=int, default=2000); ap.add_argument("--out", default="sample-matches.json"); a = ap.parse_args()
    if not KEY: sys.exit("no Freesound key")
    found = search(a.pages); print(f"loops found: {len(found)}", flush=True)
    found = found[:a.max]   # a cap, so a run cannot outlast its time limit and lose everything
    rows = []
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(measure, found, chunksize=4)):
            if r: rows.append(r)
            if i % 200 == 0: print(f"measured {i} of {len(found)}", flush=True)
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
