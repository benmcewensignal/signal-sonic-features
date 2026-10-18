"""Swap end to end through the deployed web app, exactly as the page uses it (worker/modal_app.py, swap): a token signed
as the site signs it; the cross-origin preflight; a synthetic track with known beats and bars, and real records from
several scenes; each one read (separation on a GPU, the grid, the window, the measures), a loop for each part against
your own part and against the nearest artist, a plan built as the page builds it, the whole track rendered and
downloaded, and where the swapped drums land against the grid. Annotations, and swap-test.json plus the synthetic
track's render in the folder given (the workflow keeps them on the swap-test-results branch).
  PARTS_KEY=... python tools/swap_e2e.py out/"""
import os, sys, json, time, hmac, hashlib, secrets, subprocess, math
import numpy as np, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swap_synth import track

BASE = os.environ.get("SWAP_URL", "https://benmcewensignal--sonic-parts-swap.modal.run").rstrip("/")
KEY = os.environ.get("PARTS_KEY", "")
OUT = sys.argv[1] if len(sys.argv) > 1 else "swap-out"; os.makedirs(OUT, exist_ok=True)
SR = 44100
PARTS = ("drums", "bass", "other", "vocals")


def tok(scope="t"):
    exp = int(time.time()) + 3600; nonce = secrets.token_hex(6)
    return f"{exp}.{nonce}.{scope}." + hmac.new(KEY.encode(), f"swap|{exp}|{nonce}|{scope}".encode(), hashlib.sha256).hexdigest()[:32]


def note(title, msg): print(f"::notice title={title}::{msg}", flush=True)
def warn(title, msg): print(f"::warning title={title}::{msg}", flush=True)


def decode(data):
    p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "2", "-ar", str(SR), "pipe:1"], input=data, capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float32).reshape(-1, 2)


def wav_file(x, path):
    import soundfile as sf
    sf.write(path, np.clip(x, -1, 1), SR, subtype="PCM_16")


def runs_of(act):
    """Runs of active bars, joined across single silent bars (the cycle keeps its place through a one-bar gap)."""
    runs, b, n = [], 0, len(act)
    while b < n:
        if not act[b]: b += 1; continue
        s = b
        while b < n and act[b]: b += 1
        runs.append([s, b])
    j = []
    for r in runs:
        if j and r[0] - j[-1][1] <= 1: j[-1][1] = r[1]
        else: j.append(list(r))
    return j


def plan_for(res, loops):
    """The page's rule (swap.html), mirrored: a bar plays the loop where your part's level is within 14 dB of its loud
    bars (and above -50 dB), at your part's level in that bar, at most 12 dB up."""
    g = res["grid"]; parts = {}
    for p, L in loops.items():
        lv = np.array(res["levels"][p], float); amp = 10 ** (lv / 20); ref = float(np.percentile(amp, 90))
        act = amp >= max(0.2 * ref, 10 ** (-50 / 20))
        gains = np.where(act, np.minimum(4.0, 10 ** ((lv - L["prep"]["rms_db"]) / 20)), 0.0)
        parts[p] = {"loop": L["loop"]["id"], "f": L["prep"]["f"], "shift": L["prep"]["shift"], "gains": [round(float(x), 4) for x in gains], "runs": runs_of(act)}
    return {"grid": {"g0": g["g0"], "bar": g["bar"], "nb": g["nb"]}, "tempo": res["tempo"], "name": res["name"], "parts": parts}


def wait(id_, limit=900):
    t0 = time.time()
    while time.time() - t0 < limit:
        r = requests.get(f"{BASE}/result", params={"id": id_, "t": tok()}, timeout=120).json()
        if r.get("status") != "running": return r, time.time() - t0
        time.sleep(3)
    return {"status": "timeout"}, time.time() - t0


def grid_offset_ms(x, a, p, mult=1, band=150):
    """Where the attacks of x sit against the grid (first beat a, period p): the attacks' Fourier phase at mult times the
    beat rate (1: beats, from the low band under band Hz; 4: sixteenths, from the whole band), as milliseconds after the
    grid's lines, and how strongly periodic they are there (0 to 1)."""
    from scipy.signal import butter, sosfiltfilt
    m = x.mean(1)
    lo = np.abs(sosfiltfilt(butter(4, band, "low", fs=SR, output="sos"), m)) if mult == 1 else np.abs(m)
    hop = 64; e = lo[: len(lo) // hop * hop].reshape(-1, hop).max(1); r = np.maximum(0, np.diff(e, prepend=e[:1]))
    t = np.arange(len(r)) * hop / SR; q = p / mult
    z = np.sum(r * np.exp(-2j * np.pi * t / q))
    at = (-np.angle(z) / (2 * np.pi) * q) % q
    d = (at - a) % q
    return round(1000 * (d - q if d > q / 2 else d), 1), round(float(abs(z) / (r.sum() + 1e-9)), 3)


def probe_offset(data):
    x = decode(data).mean(1); i = int(np.argmax(np.abs(x)))
    return round(1000 * (i / SR - 0.5), 2)


def one(label, data, truth=None, keep_render=False):
    out = {"label": label, "bytes": len(data)}
    t0 = time.time()
    r = requests.post(f"{BASE}/start", params={"t": tok(), "name": label}, data=data, timeout=600)
    out["upload_s"] = round(time.time() - t0, 1)
    if r.status_code != 200: out["error"] = f"start {r.status_code}: {r.text[:200]}"; warn("swap start", f"{label}: {out['error']}"); return out
    jid = r.json()["id"]
    res, took = wait(jid)
    out["read_s"] = round(took, 1)
    if res.get("status") != "done": out["error"] = str(res)[:300]; warn("swap read", f"{label}: {out['error']}"); return out
    R = res["result"]; g = R["grid"]
    out.update({"tempo": R["tempo"], "key": (R.get("key") or {}).get("key"), "grid": g, "window": R["window"], "timing": R.get("timing"),
                "like": (R.get("like") or {}).get("all", [])[:3], "duration": R["duration"], "beat_levels": bool(R.get("beat_levels"))})
    try:   # the original against its own grid: what an aligned render should match
        x0 = decode(data); out["original_vs_grid_ms"] = {"beats": grid_offset_ms(x0, g["first_beat"], g["beat"]), "sixteenths": grid_offset_ms(x0, g["first_beat"], g["beat"], 4)}
    except Exception as e:
        out["original_vs_grid_ms"] = str(e)[:80]
    if truth:
        out["truth"] = {"bpm": truth["bpm"], "g0": round(truth["g0"], 4)}
        out["downbeat_error_beats"] = round(((g["g0"] - truth["g0"]) / (60 / truth["bpm"])) % 4, 3)
        out["tempo_error_pct"] = round(100 * (R["tempo"] / truth["bpm"] - 1), 3)
    # the parts over the window, and the click that says how the MP3 decoder shifts them
    sizes = {}
    for p in PARTS + ("probe",):
        s = requests.get(f"{BASE}/stem", params={"id": jid, "part": p, "t": tok()}, timeout=120)
        sizes[p] = len(s.content) if s.status_code == 200 else f"{s.status_code}"
        if p == "probe" and s.status_code == 200: out["probe_ms"] = probe_offset(s.content)
    out["stem_bytes"] = sizes
    # a loop for every part against your own part, and drums and bass against the nearest artist
    loops, picks = {}, {}
    for p in PARTS:
        v = (R.get("vectors") or {}).get(p)
        if not v: continue
        t1 = time.time()
        L = requests.post(f"{BASE}/loop", params={"t": tok()}, json={"part": p, "target": {"kind": "yours", "vector": v}, "tempo": R["tempo"], "key": out["key"]}, timeout=300)
        if L.status_code != 200: picks[p] = {"error": L.text[:160]}; continue
        L = L.json(); loops[p] = L
        picks[p] = {"loop": f"{L['loop']['name']} by {L['loop']['user']} ({L['loop']['licence']})", "closeness": L["closeness"], "rank": L["rank"], "of": L["of"],
                    "prep": L["prep"], "skipped": L.get("skipped"), "s": round(time.time() - t1, 1)}
    out["yours"] = picks
    art = (out["like"] or [[None]])[0][0]
    if art:
        ap = {}
        for p in ("drums", "bass"):
            t1 = time.time()
            L = requests.post(f"{BASE}/loop", params={"t": tok()}, json={"part": p, "target": {"kind": "artist", "name": art}, "tempo": R["tempo"], "key": out["key"]}, timeout=300)
            j = L.json() if L.headers.get("content-type", "").startswith("application/json") else {"error": L.text[:120]}
            ap[p] = {"error": j["error"]} if j.get("error") else {"loop": f"{j['loop']['name']} by {j['loop']['user']}", "artist": j.get("artist"), "s": round(time.time() - t1, 1)}
        out["artist"] = {"name": art, "parts": ap}
    # the whole track with the drums and bass swapped for the loops nearest your own
    sw = {p: loops[p] for p in ("drums", "bass") if p in loops}
    if sw:
        plan = plan_for(R, sw)
        t1 = time.time()
        rr = requests.post(f"{BASE}/render", params={"t": tok()}, files={"file": ("track", data), "plan": (None, json.dumps(plan))}, timeout=600)
        if rr.status_code != 200: out["render_error"] = rr.text[:200]
        else:
            rid = rr.json()["id"]; st, took = wait(rid)
            f = requests.get(f"{BASE}/file", params={"id": rid, "t": tok()}, timeout=300)
            out["render"] = {"s": round(time.time() - t1, 1), "status": st.get("status"), "credits": (st.get("result") or {}).get("credits"),
                             "file": f.status_code, "bytes": len(f.content), "disposition": f.headers.get("content-disposition")}
            if f.status_code == 200:
                x = decode(f.content)
                out["render"]["seconds"] = round(len(x) / SR, 1)
                out["render"]["vs_grid_ms"] = {"beats": grid_offset_ms(x, g["first_beat"], g["beat"]), "sixteenths": grid_offset_ms(x, g["first_beat"], g["beat"], 4)}
                if keep_render:
                    open(os.path.join(OUT, f"{label}-swapped.mp3"), "wb").write(f.content)
    print(json.dumps(out, indent=1)[:4000], flush=True)
    return out


def main():
    results = {"base": BASE, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    # the door: a preflight from the site, and no token
    pf = requests.options(f"{BASE}/loop", headers={"Origin": "https://www.earlysignal.live", "Access-Control-Request-Method": "POST",
                                                    "Access-Control-Request-Headers": "content-type"}, timeout=60)
    results["preflight"] = {"status": pf.status_code, "allow_origin": pf.headers.get("access-control-allow-origin")}
    results["no_token"] = requests.get(f"{BASE}/artists", timeout=60).status_code
    results["example_token_cannot_split"] = requests.post(f"{BASE}/start", params={"t": tok("e")}, data=b"x" * 1000, timeout=60).status_code
    a = requests.get(f"{BASE}/artists", params={"t": tok()}, timeout=300)
    results["artists"] = len(a.json().get("artists", [])) if a.status_code == 200 else a.text[:200]
    note("swap door", json.dumps({k: results[k] for k in ("preflight", "no_token", "example_token_cannot_split", "artists")}))
    runs = []
    # a synthetic track, six minutes, 63 MB as WAV: known tempo and bars, and the size of a producer's bounce
    st, truth = track(bpm=124.0, offset=0.37, pickup_beats=1, bars=180, seed=3)
    mix = sum(st.values()); wp = os.path.join(OUT, "synthetic.wav"); wav_file(mix, wp)
    runs.append(one("synthetic", open(wp, "rb").read(), truth, keep_render=True))
    from concurrent.futures import ThreadPoolExecutor
    def rec_run(rec):
        try:
            data = requests.get(rec["preview"], timeout=60).content
        except Exception as e:
            return {"label": rec["name"], "error": f"preview {type(e).__name__}"}
        o = one(f"{rec['scene']}: {rec['name']}", data)
        o["catalogue_tempo"] = [rec.get("tempo"), rec.get("tempo2")]
        return o
    with ThreadPoolExecutor(4) as ex:
        runs += list(ex.map(rec_run, json.load(open(os.path.join(os.path.dirname(__file__), "swap-test-records.json")))))
    results["runs"] = runs
    json.dump(results, open(os.path.join(OUT, "swap-test.json"), "w"), indent=1)
    for o in runs:
        if o.get("error"): continue
        msg = {k: o.get(k) for k in ("tempo", "catalogue_tempo", "tempo_error_pct", "downbeat_error_beats", "read_s", "upload_s", "probe_ms")}
        msg["phase_margin"] = (o.get("grid") or {}).get("phase_margin")
        msg["render"] = {k: (o.get("render") or {}).get(k) for k in ("s", "seconds", "vs_grid_ms")}
        msg["original_vs_grid_ms"] = o.get("original_vs_grid_ms"); msg["rough_tempo"] = (o.get("grid") or {}).get("rough_tempo")
        msg["drums"] = (o.get("yours") or {}).get("drums", {}).get("loop")
        note(f"swap {o['label']}"[:60], json.dumps(msg)[:900])


main()
