"""Do the loops Swap lays sound? On the example the page opens, for each of its four parts and every target the page
offers there (Nearest, the five artists it suggests, three scenes), the five loops /loop lists and each one's prepared
cycle (/loopaudio): its level, how much of it is silence, whether its sound sits in a hit or two (a one-shot read as a
loop), and how loud the page lays it against your part in the bars your part plays (the page lifts a loop by at most
12 dB towards your part's level there, so a quiet loop stays under it). Runs against the deployed worker as the page
does. Annotations; loop-audit.json for the results branch.
  PARTS_KEY=... python tools/swap_loop_audit.py out/"""
import os, sys, io, json, time, hmac, hashlib, secrets, base64, math, wave, statistics
from concurrent.futures import ThreadPoolExecutor
import numpy as np, requests

BASE = os.environ.get("SWAP_URL", "https://benmcewensignal--sonic-parts-swap.modal.run").rstrip("/")
SITE = os.environ.get("SITE", "https://earlysignal.live").rstrip("/")
KEY = os.environ.get("PARTS_KEY", "")
OUT = sys.argv[1] if len(sys.argv) > 1 else "audit-out"; os.makedirs(OUT, exist_ok=True)
PARTS = ("drums", "bass", "other", "vocals")
SCENES = ("uk-garage-speed-garage", "tech-house", "techno-peak-time")


def tok(scope="e"):
    exp = int(time.time()) + 3600; nonce = secrets.token_hex(6)
    return f"{exp}.{nonce}.{scope}." + hmac.new(KEY.encode(), f"swap|{exp}|{nonce}|{scope}".encode(), hashlib.sha256).hexdigest()[:32]


def note(title, msg): print(f"::notice title={title}::{msg}", flush=True)


def post(path, body, tries=2):
    for k in range(tries):
        try:
            r = requests.post(f"{BASE}{path}", params={"t": tok()}, json=body, timeout=900)
            return r.json() if r.headers.get("content-type", "").startswith("application/json") else {"error": f"http {r.status_code}"}
        except requests.RequestException as e:
            err = f"{type(e).__name__}"; time.sleep(3)
    return {"error": err}


def mono(wav_b64):
    w = wave.open(io.BytesIO(base64.b64decode(wav_b64)))
    sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
    x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    return x.reshape(-1, ch).mean(1), sr


def levels(R):
    """The page's bar levels and the bars each part plays in, over the window (swap.html: levels, activeBars)."""
    g = R["grid"]; w = R["window"]; beat = g["beat"]; bar = g["bar"]; a = g["first_beat"]; g0 = g["g0"]
    while g0 >= bar - 1e-6: g0 -= bar
    j0 = round((g0 - a) / beat); nb = (g["nbeats"] - j0) // 4
    bA = max(0, math.floor((w["start_s"] - g0) / bar + 1e-3)); bB = min(nb, math.ceil((w["end_s"] - g0) / bar - 1e-3))
    out = {}
    for p in PARTS:
        bl = R["beat_levels"][p]; lv = []
        for b in range(nb):
            e = n = 0
            for q in range(4):
                i = j0 + 4 * b + q
                if 0 <= i < len(bl) and bl[i] is not None: e += 10 ** (bl[i] / 10); n += 1
            lv.append(10 * math.log10(e / n + 1e-12) if n else -120)
        amp = [10 ** (d / 20) for d in lv]; ref = float(np.percentile(amp, 90)); thr = max(0.2 * ref, 10 ** (-50 / 20))
        out[p] = [(lv[b], amp[b] >= thr) for b in range(bA, bB)]
    return out


def analyse(x, sr, prep, lv):
    """Its level, its silence, how much of its sound sits in its loudest tenth (one hit in a bar of decay reads near 1;
    a loop that plays through, near 0.1 to 0.3), and how far under your part the page lays it."""
    rms = float(np.sqrt(np.mean(x ** 2))) or 1e-9; fr = int(0.05 * sr); F = x[: len(x) // fr * fr].reshape(-1, fr)
    e = (F ** 2).mean(1); db = 10 * np.log10(e + 1e-12)
    top = np.sort(e)[::-1]; k = max(1, len(top) // 10)
    rd = float(prep.get("rms_db", 20 * math.log10(rms)))
    act = [d for d, on in lv if on]
    under = [max(0.0, (d - rd) - 12.04) for d in act]   # dB the page leaves it under your part (gain capped at 4)
    return {"seconds": round(len(x) / sr, 2), "rms_db": round(20 * math.log10(rms), 1), "peak_db": round(20 * math.log10(float(np.abs(x).max()) or 1e-9), 1),
            "silent_share": round(float(np.mean(db < -50)), 3), "loudest_tenth_share": round(float(top[:k].sum() / (e.sum() or 1)), 3),
            "under_your_part_db": round(statistics.median(under), 1) if under else None}


def main():
    R = requests.get(f"{SITE}/data/swap-example/reading.json", timeout=60).json()
    LV = levels(R); made = [str(x) for x in (R.get("made_of") or [])]
    arts = [a[0] for a in ((R.get("like") or {}).get("all") or [])[:5]]
    targets = [("nearest", None)] + [("artist", a) for a in arts] + [("scene", s) for s in SCENES]
    jobs = [(p, kind, name) for p in PARTS for kind, name in targets]

    def run(job):
        p, kind, name = job
        tgt = {"kind": "yours", "vector": (R.get("vectors") or {}).get(p)} if kind == "nearest" else {"kind": kind, "name": name}
        t0 = time.time(); j = post("/loop", {"part": p, "target": tgt, "tempo": R["tempo"], "key": (R.get("key") or {}).get("key"), "exclude": made, "n": 5})
        row = {"part": p, "kind": kind, "target": name, "seconds": round(time.time() - t0, 1)}
        if j.get("error") or not j.get("wav"): row["error"] = j.get("error") or "no wav"; return row
        ents = [dict(j, wav=None)] + (j.get("others") or []); loops = []
        for i, e in enumerate(ents):
            wb = j["wav"] if i == 0 else post("/loopaudio", {"part": p, "id": e["loop"]["id"], "f": e["prep"]["f"], "shift": e["prep"]["shift"], "tempo": R["tempo"]}).get("wav")
            L = {"rank": i + 1, "id": e["loop"]["id"], "name": e["loop"]["name"], "user": e["loop"]["user"], "closer_than": e.get("closer_than"),
                 "prep": {k: e["prep"].get(k) for k in ("bars", "cycle_bars", "f", "loop_tempo", "rms_db", "tail_bars", "grid_fit", "lead_ms")}}
            if not wb: L["error"] = "no audio"; loops.append(L); continue
            x, sr = mono(wb); L.update(analyse(x, sr, e["prep"], LV[p])); loops.append(L)
        row["loops"] = loops; row["artist"] = j.get("artist"); return row

    t_all = time.time()
    with ThreadPoolExecutor(3) as ex:
        rows = list(ex.map(run, jobs))
    for r in rows: print(json.dumps({k: r[k] for k in r if k != "loops"} | {"first": (r.get("loops") or [{}])[0]}), flush=True)
    json.dump({"base": BASE, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t_all), "artists": arts, "rows": rows},
              open(os.path.join(OUT, "loop-audit.json"), "w"), indent=1)
    # per part and kind of target: how many of the loops sound like a loop, and how far under your part the page lays them
    for p in PARTS:
        for kind in ("nearest", "artist", "scene"):
            L = [l for r in rows if r["part"] == p and r["kind"] == kind for l in (r.get("loops") or []) if "rms_db" in l]
            F = [l for r in rows if r["part"] == p and r["kind"] == kind for l in (r.get("loops") or [])[:1] if "rms_db" in l]
            errs = [r.get("error") for r in rows if r["part"] == p and r["kind"] == kind and r.get("error")]
            if not L: note(f"loops {p} {kind}", f"none came back {errs[:3]}"); continue
            hit = lambda l: l["loudest_tenth_share"] >= 0.6 or l["silent_share"] >= 0.6
            note(f"loops {p} {kind}", f"{len(L)} loops: {sum(1 for l in L if hit(l))} mostly one hit or silence (first listed: {sum(1 for l in F if hit(l))} of {len(F)}); "
                 f"median level {statistics.median(l['rms_db'] for l in L):.1f} dB; laid under your part by a median {statistics.median([l['under_your_part_db'] or 0 for l in L]):.1f} dB, "
                 f"6 dB or more for {sum(1 for l in L if (l['under_your_part_db'] or 0) >= 6)}. Firsts: " + "; ".join(f"{l['name'][:40]} ({l['rms_db']} dB, top tenth {l['loudest_tenth_share']}, silent {l['silent_share']}, under {l['under_your_part_db']})" for l in F[:6])
                 + (f". Errors: {errs[:3]}" if errs else ""))


if __name__ == "__main__":
    main()
