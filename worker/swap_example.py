"""The Swap page's example: a short tech house sketch made only from CC0 loops in Sonic's licensed library (the loops
nearest the average of tech house records, part by part), so it can be split and swapped on the page without anyone's
record in it. It is read by the deployed swap reader exactly as an upload is; the page loads the reading and the parts
from signalgood/data/swap-example/ and leaves the sketch's own loops out of its swaps.
  modal run worker/swap_example.py --out swap-example/"""
import json, os, hashlib
import modal

app = modal.App("sonic-swap-example")
img = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg", "rubberband-cli", "libsndfile1")
       .pip_install("numpy<2", "soundfile")
       .add_local_file("tools/swap-example-ids.json", "/root/ids.json")
       .add_local_dir("worker", "/root/worker"))
vol = modal.Volume.from_name("sonic-embed")
TEMPO, KEY, BARS = 125.0, "Am", 64
# which bars each part plays: intro on drums, the bass in at 9, the melody at 17, a breakdown at 33 with the voice,
# everything from 41, drums and bass out from 57
PLAN = {"drums": [(0, 32), (40, 64)], "bass": [(8, 32), (40, 64)], "other": [(16, 56)], "vocals": [(32, 56)]}
LEVEL_DB = {"drums": -15.0, "bass": -17.0, "other": -21.0, "vocals": -22.0}


@app.function(image=img, volumes={"/embed": vol}, timeout=1200, cpu=2.0, memory=4096)
def build():
    import sys, numpy as np
    sys.path.insert(0, "/root")
    from worker import swap as SW
    ids = json.load(open("/root/ids.json"))["ids"]
    RP = SW.record_parts(); rows = [RP["at"][t] for t in ids if t in RP["at"]]
    picks, cycles = {}, {}
    from concurrent.futures import ThreadPoolExecutor
    def pick(p):
        c = RP["V"][rows, SW.PARTS.index(p)].astype(float).mean(0)
        L = SW.loop_index()[SW.FAM[p]]
        q, zn = SW._space(L, c); sim = np.asarray(L[zn], np.float32) @ q
        tries = 0
        for i in np.argsort(-sim)[:600]:
            m = L["meta"][int(i)]
            if "publicdomain/zero" not in str(m.get("license")): continue      # CC0 only: the example owes no one a credit
            lt = float(m.get("tempo") or 0)
            if not lt or abs(lt / TEMPO - 1) > 0.05: continue
            s = SW.key_shift(m.get("key"), KEY) if p != "drums" else 0
            if p != "drums" and (not m.get("key") or abs(s) > 3): continue
            tries += 1
            if tries > 40: break
            try:   # a clean, straight loop: the example is the page's first impression (voices are rarely quantised)
                y, prep = SW.prepare(m, 1.0, s, TEMPO, 0.45 if p == "vocals" else 0.6)
            except Exception:
                continue
            if prep["rms_db"] < -45: continue
            return p, {"id": m.get("id"), "name": m.get("name"), "user": m.get("user"), "licence": "CC0", "page": f"https://freesound.org/s/{m.get('id')}/",
                       "closeness": round(float(sim[i]), 3), "prep": prep}, (y, prep)
        return p, None, None
    with ThreadPoolExecutor(4) as ex:
        for p, meta_, cyc in ex.map(pick, SW.PARTS):
            if meta_: picks[p] = meta_; cycles[p] = cyc
    bar = 240 / TEMPO; g0 = 0.5; n = int((g0 + BARS * bar + 2.0) * SW.SR)
    out = np.zeros((n, 2), np.float32)
    for p, (y, prep) in cycles.items():
        act = [any(a <= b < z for a, z in PLAN[p]) for b in range(BARS)]
        gain = 10 ** ((LEVEL_DB[p] - prep["rms_db"]) / 20)
        out += SW.lay(y, n, g0, bar, [gain if a else 0.0 for a in act], [list(r) for r in PLAN[p]])
    pk = float(np.abs(out).max()) or 1.0; out *= 0.89 / pk
    mp3 = SW.mp3_bytes(out, kbps=192, meta={"title": "Swap example (CC0 loops)"})
    return {"mp3": mp3, "picks": picks, "tempo": TEMPO, "key": KEY, "bars": BARS}


@app.local_entrypoint()
def main(out: str = "swap-example"):
    os.makedirs(out, exist_ok=True)
    b = build.remote()
    open(os.path.join(out, "source.mp3"), "wb").write(b["mp3"])
    sha = hashlib.sha256(b["mp3"]).hexdigest()
    print("::notice title=swap example::" + json.dumps({"sha256": sha, "picks": {p: f"{x['name']} by {x['user']} (closeness {x['closeness']})" for p, x in b["picks"].items()}}))
    read = modal.Function.from_name("sonic-parts", "swap_read")
    r = read.remote(b["mp3"], "Example: a tech house sketch from CC0 loops")
    if r.get("error"): print("::error title=swap example::" + r["error"]); raise SystemExit(1)
    audio = r.pop("audio")
    for k, v in audio.items(): open(os.path.join(out, f"{k}.mp3"), "wb").write(v)
    r["made_of"] = [str(x["id"]) for x in b["picks"].values()]
    r["made_from"] = b["picks"]; r["sha256"] = sha; r["example"] = True
    json.dump(r, open(os.path.join(out, "reading.json"), "w"), separators=(",", ":"))
    print("::notice title=swap example reading::" + json.dumps({k: r.get(k) for k in ("tempo", "key", "grid", "window", "timing")}))
