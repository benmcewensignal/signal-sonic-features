"""The DJ tool's index (site: data/dj-index.json): every recognisable, separated record's character, tempo, key,
scene and map position, plus its place in the learned ear. Rebuild after the recognition index, the separated
records' parts file or the learned ear grows, so new records (the classics) reach "where next".

  python tools/dj_index.py --index ../signal-sonic-audio/out/index.json --detail ../signal-sonic-audio/out/detail \
      --parts record-parts.npz --ear data/walk_ear.json --out ../signalgood/data/dj-index.json

Inputs: the lean recognition index and its detail files (audio repo), record-parts.npz (built by
tools/build_record_parts.py in worker-deploy), and data/walk_ear.json (ear_walk in embed/modal_embed.py).
"""
import argparse, base64, glob, json, os
import numpy as np

CAM = ["G#m","D#m","A#m","Fm","Cm","Gm","Dm","Am","Em","Bm","F#m","C#m","B","F#","C#","G#","D#","A#","F","C","G","D","A","E"]
SP = [-0.636544, 0.448711, -0.4145, 0.551105]   # the map's axis spans (driving, defined), as the site draws them

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True); ap.add_argument("--detail", required=True); ap.add_argument("--parts", required=True)
    ap.add_argument("--ear", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    I = json.load(open(a.index)); T = I["tracks"]; SCN = I.get("scenes") or []
    RP = np.load(a.parts); at = {t: i for i, t in enumerate(RP["ids"].tolist())}; V = RP["V"].astype(np.float32)
    det = {}
    for f in glob.glob(os.path.join(a.detail, "*.json")):
        for k, v in json.load(open(f)).items():
            det[k] = {"key": (v.get("key") or {}).get("key") if isinstance(v.get("key"), dict) else None, "pos": v.get("pos")}
    rows = []
    for t in T:
        i = at.get(t["track_id"])
        if i is None or not t.get("tm"): continue
        sh = np.clip(V[i, :, 52], 0, None); tot = sh.sum()
        if tot <= 0: continue
        bright = float((sh * V[i, :, 48]).sum() / tot)
        d = det.get(t["track_id"]) or {}; p = d.get("pos") or {}
        mx = my = None
        dv = p.get("defined") if p.get("defined") is not None else p.get("melodic")
        if p.get("driving") is not None and dv is not None:
            mx = int(np.clip((p["driving"] - SP[0]) / (SP[1] - SP[0]) * 254 - 127, -127, 127)); my = int(np.clip((dv - SP[2]) / (SP[3] - SP[2]) * 254 - 127, -127, 127))
        sc = t.get("scene"); sci = sc if isinstance(sc, int) else (SCN.index(sc) if sc in SCN else 255)
        rows.append((t["track_id"], float(t["tm"]), CAM.index(d["key"]) if d.get("key") in CAM else 255, sci, bright,
                     sh[3] / tot, sh[0] / tot, sh[1] / tot, float(V[i, 0, 51]), float(V[i, 0, 46]), mx, my))
    n = len(rows)
    pct = lambda vals: np.round(np.asarray(vals, float).argsort().argsort() / (len(vals) - 1) * 99).astype(np.uint8)
    F = {"bright": pct([r[4] for r in rows]), "vocal": pct([r[5] for r in rows]), "drums": pct([r[6] for r in rows]),
         "bass": pct([r[7] for r in rows]), "busy": pct([r[8] for r in rows]), "punch": pct([r[9] for r in rows])}
    b64 = lambda x: base64.b64encode(np.asarray(x).tobytes()).decode()
    ids = [r[0] for r in rows]
    out = {"note": "every recognisable, separated record's character for DJ walks: percentiles (0 to 99) across these records of brightness (energy-weighted across parts), vocal, drum and bass share of the energy, drum hit density and punch; tempo, Camelot key index (255 unknown), scene index, map position (int8, 127 = unknown); ear: the learned whole-record ear, int8 per dimension with earscale, earhas marking records it has heard",
           "n": n, "ids": ids, "scenes": SCN, "camelot": CAM,
           "tempo": b64(np.array([round(r[1] * 10) for r in rows], dtype=np.uint16)), "key": b64(np.array([r[2] for r in rows], dtype=np.uint8)),
           "scene": b64(np.array([r[3] for r in rows], dtype=np.uint8)),
           "mx": b64(np.array([r[10] if r[10] is not None else 127 for r in rows], dtype=np.int8)), "my": b64(np.array([r[11] if r[11] is not None else 127 for r in rows], dtype=np.int8)),
           **{k: b64(v) for k, v in F.items()}}
    # the learned ear, aligned to these records
    W = json.load(open(a.ear)); D = W["dims"]; Q = np.frombuffer(base64.b64decode(W["ear"]), dtype=np.int8).reshape(len(W["ids"]), D)
    pos = {t: k for k, t in enumerate(ids)}; E = np.zeros((n, D), np.int8); H = np.zeros(n, np.uint8)
    for r, t in enumerate(W["ids"]):
        k = pos.get(t)
        if k is not None: E[k] = Q[r]; H[k] = 1
    out.update({"ear": b64(E), "eardims": D, "earscale": [float(x) for x in W["scale"]], "earhas": b64(H), "earbuilt": W.get("built", "")})
    json.dump(out, open(a.out, "w"), separators=(",", ":"))
    print(json.dumps({"records": n, "with_key": sum(1 for r in rows if r[2] != 255), "with_map": sum(1 for r in rows if r[10] is not None),
                      "in_the_ear": int(H.sum()), "mb": round(os.path.getsize(a.out) / 1e6, 2)}))

if __name__ == "__main__":
    main()
