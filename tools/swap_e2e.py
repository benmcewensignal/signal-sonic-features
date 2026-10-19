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


class CallFailed(Exception):
    pass


DROPS = []   # calls that failed and were tried again, as the page tries them again


def again(fn, label):
    """fn() once more after a failed call, as the page does: polling keeps polling, and a loop, a part, a render or
    the file is asked for a second time. The first failure is kept for the summary."""
    try:
        return fn()
    except CallFailed as e:
        DROPS.append(f"{label}: {e}"); time.sleep(2)
        return fn()


def http(method, url, **kw):
    """A request to the worker; a failure says which call it was, how long it had run and when (UTC), so the worker's
    own log for that moment can be found."""
    t0 = time.time()
    try:
        return getattr(requests, method)(url, **kw)
    except requests.RequestException as e:
        path = url.split("?")[0].replace(BASE, "") or url.split("?")[0]
        raise CallFailed(f"{type(e).__name__} on {method.upper()} {path} after {time.time() - t0:.1f} s, at "
                         f"{time.strftime('%H:%M:%S', time.gmtime(t0))} UTC: {str(e)[:160]}") from e


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


def plan_for(res, loops, second=None, moves=None, keep=None, master=True, follow=None, level=None, groove=None, stems=False, kit=None):
    """The page's rule (swap.html), mirrored: a bar plays the loop where your part's level is within 14 dB of its loud
    bars (and above -50 dB), at your part's level in that bar, at most 12 dB up. With second {part: entry}, that part
    gets two layers: the first loop in the first half of the bars and the second in the rest (a section's own choice).
    keep: your own part's share under the swap; moves: cuts and rolls at phrase ends; master: loudness and limiting."""
    g = res["grid"]; parts = {}; nb = g["nb"]
    for p, L in loops.items():
        lv = np.array(res["levels"][p], float); amp = 10 ** (lv / 20); ref = float(np.percentile(amp, 90))
        act = amp >= max(0.2 * ref, 10 ** (-50 / 20))
        def layer(entry, where):
            a = act & where
            gains = np.where(a, np.minimum(4.0, 10 ** ((lv - entry["prep"]["rms_db"]) / 20)), 0.0)
            return {"loop": entry["loop"]["id"], "f": entry["prep"]["f"], "shift": entry["prep"]["shift"], "gains": [round(float(x), 4) for x in gains], "runs": runs_of(a)}
        if second and p in second:
            half = np.arange(nb) < nb // 2
            parts[p] = [layer(L, half), layer(second[p], ~half)]
        else:
            parts[p] = [layer(L, np.ones(nb, bool))]
        if follow and p in follow:
            for lay_ in parts[p]: lay_["follow"] = True
        if kit and p == "drums":
            for lay_ in parts[p]: lay_["kit"] = kit
    out = {"grid": {"g0": g["g0"], "bar": g["bar"], "nb": nb}, "tempo": res["tempo"], "name": res["name"], "parts": parts, "master": master}
    if keep: out["keep"] = keep
    if moves: out["moves"] = moves
    if level: out["level"] = level
    if groove: out["groove"] = groove
    if stems: out["stems"] = True
    return out


def lufs(data):
    """Integrated loudness of an audio file's bytes (ffmpeg ebur128), or None."""
    import re
    p = subprocess.run(["ffmpeg", "-loglevel", "info", "-i", "pipe:0", "-af", "ebur128=framelog=quiet", "-f", "null", "-"], input=data, capture_output=True, timeout=600)
    m = re.search(r"I:\s*(-?[\d.]+) LUFS", p.stderr.decode("utf-8", "replace"))
    return float(m.group(1)) if m else None


def lufs_x(x):
    """Integrated loudness of decoded audio: encoded to WAV for ffmpeg."""
    import io, soundfile as sf
    b = io.BytesIO(); sf.write(b, x, SR, format="WAV", subtype="FLOAT"); return lufs(b.getvalue())


def last_beat_share(x, g0, bar, phrase=8, offset=0, band=150):
    """How much of the low band's attack energy in each phrase's last bar falls in its last beat (a cut leaves it near
    nothing in the swapped layer; a straight loop puts about a quarter there), averaged over the phrases."""
    from scipy.signal import butter, sosfiltfilt
    m = x.mean(1); lo = np.abs(sosfiltfilt(butter(4, band, "low", fs=SR, output="sos"), m))
    hop = 64; e = lo[: len(lo) // hop * hop].reshape(-1, hop).max(1); r = np.maximum(0, np.diff(e, prepend=e[:1])); fr = SR / hop
    shares = []
    nb = int((len(m) / SR - g0) / bar)
    for b in range(nb):
        if (b - offset) % phrase != phrase - 1: continue
        s0, s1 = int((g0 + b * bar) * fr), int((g0 + (b + 1) * bar) * fr); s4 = int((g0 + b * bar + 0.75 * bar) * fr)
        tot = r[s0:s1].sum()
        if tot > 0: shares.append(float(r[s4:s1].sum() / tot))
    return round(float(np.mean(shares)), 3) if shares else None


def wait(id_, limit=900):
    t0 = time.time()
    while time.time() - t0 < limit:
        try:
            rr = http("get", f"{BASE}/result", params={"id": id_, "t": tok()}, timeout=300)
        except CallFailed as e:   # the page polls on through a failed poll
            DROPS.append(f"poll: {e}"); time.sleep(5); continue
        try: r = rr.json()
        except Exception: r = {"status": "running"} if rr.status_code >= 500 else {"status": "failed", "error": f"{rr.status_code} {rr.text[:120]}"}
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


def low_lag_ms(x, y, max_ms=150, band=150):
    """How far y's low band (kicks and bass) sits after x's, in milliseconds: the lag of the largest cross-correlation of
    their attack envelopes within max_ms. A swapped render whose loops sit on the track's beats lags its original by
    about nothing; one laid on a grid a sixteenth off lags by a sixteenth."""
    from scipy.signal import butter, sosfiltfilt
    def env(z):
        lo = np.abs(sosfiltfilt(butter(4, band, "low", fs=SR, output="sos"), z.mean(1)))
        hop = 64; e = lo[: len(lo) // hop * hop].reshape(-1, hop).max(1); r = np.maximum(0, np.diff(e, prepend=e[:1]))
        return (r - r.mean()) / (r.std() + 1e-12)
    a, b = env(x), env(y); n = min(len(a), len(b)); a, b = a[:n], b[:n]
    L = int(max_ms / 1000 * SR / 64)
    xc = [float(np.dot(a[max(0, -k): n - max(0, k)], b[max(0, k): n - max(0, -k)])) for k in range(-L, L + 1)]
    k = int(np.argmax(xc)) - L
    return round(k * 64 / SR * 1000, 1)


def probe_offset(data):
    x = decode(data).mean(1); i = int(np.argmax(np.abs(x)))
    return round(1000 * (i / SR - 0.5), 2)


def one(label, data, truth=None, keep_render=False):
    """One track through the whole path; a failure anywhere is recorded with where it happened, not raised."""
    import traceback
    try:
        return _one(label, data, truth, keep_render)
    except Exception as e:
        tb = traceback.format_exc().strip().splitlines()
        where = " | ".join(x.strip() for x in tb[-4:])[:600]
        print(f"::error title=swap {label}"[:60] + f"::{type(e).__name__}: {str(e)[:500]} | {where[-300:]}", flush=True)
        return {"label": label, "error": f"{type(e).__name__}: {str(e)[:200]}", "where": where}


def _one(label, data, truth=None, keep_render=False):
    out = {"label": label, "bytes": len(data)}
    t0 = time.time()
    r = http("post", f"{BASE}/start", params={"t": tok(), "name": label}, data=data, timeout=600)
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
        s = again(lambda: http("get", f"{BASE}/stem", params={"id": jid, "part": p, "t": tok()}, timeout=120), f"{label} part {p}")
        sizes[p] = len(s.content) if s.status_code == 200 else f"{s.status_code}"
        if p == "drums" and s.status_code == 200: out["_drums_stem"] = s.content
        if p == "probe" and s.status_code == 200: out["probe_ms"] = probe_offset(s.content)
    out["stem_bytes"] = sizes
    # a loop for every part against your own part, and drums and bass against the nearest artist
    loops, picks = {}, {}
    for p in PARTS:
        v = (R.get("vectors") or {}).get(p)
        if not v: continue
        t1 = time.time()
        L = again(lambda: http("post", f"{BASE}/loop", params={"t": tok()}, json={"part": p, "target": {"kind": "yours", "vector": v}, "tempo": R["tempo"], "key": out["key"]}, timeout=600), f"{label} loop {p}")
        if L.status_code != 200: picks[p] = {"error": L.text[:160]}; continue
        L = L.json(); loops[p] = L
        picks[p] = {"loop": f"{L['loop']['name']} by {L['loop']['user']} ({L['loop']['licence']})", "closeness": L["closeness"], "rank": L["rank"], "of": L["of"],
                    "prep": L["prep"], "skipped": L.get("skipped"), "s": round(time.time() - t1, 1), "others": len(L.get("others") or [])}
    out["yours"] = picks
    # following your notes: the bass loop re-pitched to the line read from your bass, the drum loop's kit on your pattern,
    # over the window's bars; the audio must be as long as those bars
    FL = R.get("follow") or {}; out["lines"] = {p: ({"error": v.get("error")} if v.get("error") else ({"kicks": sum(v["kick"]), "snares": sum(v["snare"])} if p == "drums" else {"steps_with_notes": sum(1 for x in v["notes"] if x is not None), "of": len(v["notes"])})) for p, v in FL.items()}
    wb0, wbars = R["window"]["b0"], R["window"]["bars"]
    for p in ("bass", "drums"):
        if p not in loops or p not in FL or FL[p].get("error"): continue
        your = {"kick": FL[p]["kick"][wb0 * 16:(wb0 + wbars) * 16], "snare": FL[p]["snare"][wb0 * 16:(wb0 + wbars) * 16]} if p == "drums" else {"notes": FL[p]["notes"][wb0 * 16:(wb0 + wbars) * 16]}
        Lp = loops[p]; t1 = time.time()
        Fr = again(lambda: http("post", f"{BASE}/follow", params={"t": tok()}, json={"part": p, "id": Lp["loop"]["id"], "f": Lp["prep"]["f"], "shift": Lp["prep"]["shift"], "tempo": R["tempo"], "bars": wbars, "your": your}, timeout=600), f"{label} follow {p}")
        j = Fr.json() if Fr.headers.get("content-type", "").startswith("application/json") else {"error": Fr.text[:120]}
        if j.get("error"): out.setdefault("follow", {})[p] = {"error": j["error"]}; continue
        import base64 as _b64
        y = decode(_b64.b64decode(j["mp3"])); want = wbars * g["bar"]
        out.setdefault("follow", {})[p] = {"s": round(time.time() - t1, 1), "seconds": round(len(y) / SR, 2), "want_seconds": round(want, 2), "info": j.get("info")}
    # the listed loops' audio on demand: the second drum loop, prepared exactly as listed
    others = (loops.get("drums") or {}).get("others") or []; second_drums = None
    if others:
        o = others[0]; t1 = time.time()
        A = again(lambda: http("post", f"{BASE}/loopaudio", params={"t": tok()}, json={"part": "drums", "id": o["loop"]["id"], "f": o["prep"]["f"], "shift": o["prep"]["shift"], "tempo": R["tempo"]}, timeout=600), f"{label} loopaudio")
        j = A.json() if A.headers.get("content-type", "").startswith("application/json") else {"error": A.text[:120]}
        if j.get("error"): out["loopaudio"] = {"error": j["error"]}
        else:
            import base64 as _b64
            y = decode(_b64.b64decode(j["wav"]))
            out["loopaudio"] = {"s": round(time.time() - t1, 1), "samples": len(y), "listed_samples": o["prep"]["samples"], "same_prep": j["prep"]["samples"] == o["prep"]["samples"] and j["loop"]["id"] == o["loop"]["id"]}
            if out["loopaudio"]["same_prep"]: second_drums = dict(o)
    # nearest in brightness alone, among the 200 nearest in character: the bass
    if "bass" in loops and R.get("vectors", {}).get("bass"):
        t1 = time.time()
        Lf = again(lambda: http("post", f"{BASE}/loop", params={"t": tok()}, json={"part": "bass", "target": {"kind": "yours", "vector": R["vectors"]["bass"]}, "tempo": R["tempo"], "key": out["key"], "focus": ["brightness"]}, timeout=600), f"{label} focus loop")
        jf = Lf.json() if Lf.headers.get("content-type", "").startswith("application/json") else {"error": Lf.text[:120]}
        out["focus"] = {"error": jf["error"]} if jf.get("error") else {"loop": f"{jf['loop']['name']} by {jf['loop']['user']}", "focus": jf.get("focus"), "same_as_nearest": jf["loop"]["id"] == loops["bass"]["loop"]["id"],
                        "centroid_loop": (jf["loop"].get("plain") or {}).get("centroid_hz"), "s": round(time.time() - t1, 1)}
    # the drums' kit from two loops on the loop's own pattern, over the window's bars
    if "drums" in loops and second_drums:
        t1 = time.time(); Lp = loops["drums"]; wbars = int(R["window"]["bars"])
        Fk = again(lambda: http("post", f"{BASE}/follow", params={"t": tok()}, json={"part": "drums", "id": Lp["loop"]["id"], "f": Lp["prep"]["f"], "shift": 0, "tempo": R["tempo"], "bars": wbars, "your": {}, "pattern": "theirs",
                                                                                        "kit": {"kick": [second_drums["loop"]["id"], second_drums["prep"]["f"]]}}, timeout=600), f"{label} kit")
        jk = Fk.json() if Fk.headers.get("content-type", "").startswith("application/json") else {"error": Fk.text[:120]}
        out["kit"] = {"error": jk["error"]} if jk.get("error") else {"info": jk.get("info"), "s": round(time.time() - t1, 1), "mp3_bytes": len(jk.get("mp3") or "")}
    # your drums over the window re-timed to a scene's feel
    if jid:
        t1 = time.time(); G_ = R["grid"]; W_ = R["window"]
        first_bar = (G_["g0"] - W_["start_s"]) % G_["bar"]
        Gr = again(lambda: http("post", f"{BASE}/groove", params={"t": tok()}, json={"id": jid, "tempo": R["tempo"], "g0": first_bar, "nb": int(W_["bars"]), "scene": "uk-garage-speed-garage"}, timeout=600), f"{label} groove")
        jg = Gr.json() if Gr.headers.get("content-type", "").startswith("application/json") else {"error": Gr.text[:120]}
        out["groove"] = {"error": jg["error"]} if jg.get("error") else {"info": {k: v for k, v in (jg.get("info") or {}).items() if k not in ("target", "counts")}, "s": round(time.time() - t1, 1), "mp3_bytes": len(jg.get("mp3") or "")}
        # read back: the window's drums as they were and as re-timed, both decoded here the same way, through the same
        # hit reader: the beats should not move (the re-timed audio keeps the stem's timeline) and the offbeats should
        # move toward the scene's feel
        if not jg.get("error") and out.get("_drums_stem"):
            try:
                import base64 as _b64
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from worker import swap as SW
                xa = decode(out["_drums_stem"]); xb = decode(_b64.b64decode(jg["mp3"])); nbw = int(W_["bars"]); stp = G_["bar"] / 16
                ta, _ = SW.hit_times(xa); tb, _ = SW.hit_times(xb)
                fa, ca = SW.your_feel(ta, first_bar, G_["bar"], nbw); fb, cb = SW.your_feel(tb, first_bar, G_["bar"], nbw)
                tgt = SW.groove_feel("uk-garage-speed-garage")["rel"]
                ms = lambda v: round(v * stp * 1000, 1)
                read = [k for k in range(16) if ca[k] >= max(2, nbw // 4) and cb[k] >= max(2, nbw // 4)]
                beats = [k for k in read if k % 4 == 0]; offs = [k for k in read if k % 4 != 0]
                out["groove"]["read_back"] = {"steps_read": len(read), "beats_ms_before": [ms(fa[k]) for k in beats], "beats_ms_after": [ms(fb[k]) for k in beats],
                                              "beat_shift_ms_max": max([abs(ms(fb[k] - fa[k])) for k in beats] or [None]),
                                              "offbeats_ms_before": [ms(fa[k]) for k in offs], "offbeats_ms_after": [ms(fb[k]) for k in offs], "target_ms": [ms(tgt[k]) for k in offs],
                                              "offbeat_error_ms_before": round(float(np.mean([abs(ms(fa[k] - tgt[k])) for k in offs])), 1) if offs else None,
                                              "offbeat_error_ms_after": round(float(np.mean([abs(ms(fb[k] - tgt[k])) for k in offs])), 1) if offs else None}
            except Exception as e:
                out["groove"]["read_back"] = {"error": type(e).__name__ + ": " + str(e)[:100]}
    out.pop("_drums_stem", None)
    art = (out["like"] or [[None]])[0][0]
    if art:
        ap = {}
        for p in ("drums", "bass"):
            t1 = time.time()
            tg = {"kind": "artist", "name": art} if p == "drums" else {"kind": "artist", "name": art, "vector": (R.get("vectors") or {}).get(p), "w": 0.5}   # the bass: halfway to you
            L = again(lambda: http("post", f"{BASE}/loop", params={"t": tok()}, json={"part": p, "target": tg, "tempo": R["tempo"], "key": out["key"]}, timeout=600), f"{label} artist loop {p}")
            j = L.json() if L.headers.get("content-type", "").startswith("application/json") else {"error": L.text[:120]}
            ap[p] = {"error": j["error"]} if j.get("error") else {"loop": f"{j['loop']['name']} by {j['loop']['user']}", "artist": j.get("artist"), "closer_than": j.get("closer_than"), "others": len(j.get("others") or []), "s": round(time.time() - t1, 1)}
        out["artist"] = {"name": art, "parts": ap}
    # the whole track with the drums and bass swapped for the loops nearest your own
    sw = {p: loops[p] for p in ("drums", "bass") if p in loops}
    if sw:
        plan = plan_for(R, sw, second={"drums": second_drums} if (second_drums and "drums" in sw) else None, keep={"bass": 0.5},
                        moves={"phrase": 8, "offset": 0, "cut": list(sw), "roll": ["drums"] if "drums" in sw else []}, master=True,
                        follow=["bass"] if ("bass" in sw and (out.get("follow") or {}).get("bass", {}).get("info")) else None,
                        level={"other": 2.0}, groove={"scene": "house"}, stems=True, kit={"kick": [second_drums["loop"]["id"], second_drums["prep"]["f"]]} if second_drums else None)
        t1 = time.time()
        rr = again(lambda: http("post", f"{BASE}/render", params={"t": tok()}, files={"file": ("track", data), "plan": (None, json.dumps(plan))}, timeout=600), f"{label} render")
        if rr.status_code != 200: out["render_error"] = rr.text[:200]
        else:
            rid = rr.json()["id"]; st, took = wait(rid)
            f = again(lambda: http("get", f"{BASE}/file", params={"id": rid, "t": tok()}, timeout=300), f"{label} file")
            out["render"] = {"s": round(time.time() - t1, 1), "status": st.get("status"), "credits": (st.get("result") or {}).get("credits"),
                             "file": f.status_code, "bytes": len(f.content), "disposition": f.headers.get("content-disposition")}
            if f.status_code == 200:
                x = decode(f.content)
                out["render"]["seconds"] = round(len(x) / SR, 1)
                res_ = st.get("result") or {}; out["render"]["laid"] = res_.get("laid"); out["render"]["loudness"] = res_.get("loudness")
                try:
                    lo, lr = lufs(data), lufs(f.content); out["render"]["lufs"] = {"original": lo, "swapped": lr, "diff": round(lr - lo, 2) if lo is not None and lr is not None else None}
                except Exception as e:
                    out["render"]["lufs"] = str(e)[:60]
                out["render"]["last_beat_share"] = {"swapped": last_beat_share(x, g["g0"], g["bar"]), "original": last_beat_share(decode(data), g["g0"], g["bar"])}
                out["render"]["vs_grid_ms"] = {"beats": grid_offset_ms(x, g["first_beat"], g["beat"]), "sixteenths": grid_offset_ms(x, g["first_beat"], g["beat"], 4)}
                try: out["render"]["low_lag_vs_original_ms"] = low_lag_ms(decode(data), x)
                except Exception as e: out["render"]["low_lag_vs_original_ms"] = str(e)[:60]
                if keep_render:
                    open(os.path.join(OUT, f"{label}-swapped.mp3"), "wb").write(f.content)
                # the stems: a zip of four MP3s, four MIDI files and the recipe
                fz = again(lambda: http("get", f"{BASE}/file", params={"id": rid, "t": tok(), "kind": "stems"}, timeout=300), f"{label} stems")
                out["render"]["stems"] = {"file": fz.status_code, "bytes": len(fz.content)}
                if fz.status_code == 200:
                    import zipfile, io
                    try:
                        z = zipfile.ZipFile(io.BytesIO(fz.content)); names = z.namelist()
                        stems_sum = None
                        for nm_ in ("drums.mp3", "bass.mp3", "melody.mp3", "voice.mp3"):
                            if nm_ in names:
                                xs = decode(z.read(nm_)); stems_sum = xs if stems_sum is None else stems_sum[: len(xs)] + xs[: len(stems_sum)]
                        rec = z.read("recipe.txt").decode("utf-8", "replace") if "recipe.txt" in names else ""
                        mids = [n_ for n_ in names if n_.endswith(".mid")]
                        out["render"]["stems"].update({"names": names, "recipe_lines": rec.count("\n"), "recipe_has_loops": "freesound.org" in rec, "midi": len(mids),
                                                       "midi_ok": all(z.read(n_)[:4] == b"MThd" for n_ in mids),
                                                       "sum_vs_mix_lufs_diff": (round(lufs_x(stems_sum) - lufs_x(x), 2) if stems_sum is not None else None)})
                    except Exception as e:
                        out["render"]["stems"]["error"] = type(e).__name__ + ": " + str(e)[:80]
                res_ = st.get("result") or {}; out["render"]["level"] = res_.get("level"); out["render"]["groove"] = (res_.get("laid") or {}).get("groove")
    print(json.dumps(out, indent=1)[:4000], flush=True)
    return out


def main():
    results = {"base": BASE, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    # the door: a preflight from the site, and no token
    pf = http("options", f"{BASE}/loop", headers={"Origin": "https://www.earlysignal.live", "Access-Control-Request-Method": "POST",
                                                    "Access-Control-Request-Headers": "content-type"}, timeout=60)
    results["preflight"] = {"status": pf.status_code, "allow_origin": pf.headers.get("access-control-allow-origin")}
    results["no_token"] = http("get", f"{BASE}/artists", timeout=60).status_code
    results["example_token_cannot_split"] = http("post", f"{BASE}/start", params={"t": tok("e")}, data=b"x" * 1000, timeout=60).status_code
    a = http("get", f"{BASE}/artists", params={"t": tok()}, timeout=300)
    results["artists"] = len(a.json().get("artists", [])) if a.status_code == 200 else a.text[:200]
    note("swap door", json.dumps({k: results[k] for k in ("preflight", "no_token", "example_token_cannot_split", "artists")}))
    runs = []
    only = os.environ.get("SWAP_ONLY", "").strip()
    if only:   # a few records only, by part of their scene or name
        from concurrent.futures import ThreadPoolExecutor
        recs = [r for r in json.load(open(os.path.join(os.path.dirname(__file__), "swap-test-records.json"))) if only.lower() in (r["scene"] + " " + r["name"]).lower()]
        def rec_run1(rec):
            o = one(f"{rec['scene']}: {rec['name']}", http("get", rec["preview"], timeout=60).content); o["catalogue_tempo"] = [rec.get("tempo"), rec.get("tempo2")]; return o
        with ThreadPoolExecutor(2) as ex: runs = list(ex.map(rec_run1, recs))
        results["runs"] = runs; results["dropped"] = DROPS; json.dump(results, open(os.path.join(OUT, "swap-test.json"), "w"), indent=1)
        (warn if DROPS else note)("swap dropped calls", f"{len(DROPS)} failed and were tried again" + ((": " + " || ".join(DROPS))[:1500] if DROPS else ""))
        for o in runs: note(f"swap deeper {o['label']}"[:60], json.dumps({"focus": o.get("focus"), "kit": o.get("kit"), "groove": o.get("groove"), "stems": (o.get("render") or {}).get("stems"), "level": (o.get("render") or {}).get("level"), "render_groove": {k: v for k, v in ((o.get("render") or {}).get("groove") or {}).items() if k not in ("yours", "counts", "target", "shift")}, "render_s": (o.get("render") or {}).get("s"), "lufs": (o.get("render") or {}).get("lufs"), "error": o.get("error") or (o.get("render") or {}).get("error") or o.get("render_error")})[:1900])
        for o in runs: note(f"swap {o['label']}"[:60], json.dumps({k: o.get(k) for k in ("tempo", "catalogue_tempo", "error")} | {"grid": {k: (o.get("grid") or {}).get(k) for k in ("strength", "tracker_offset_ms", "phase_margin")}})[:900])
        return
    # a synthetic track, six minutes, 63 MB as WAV: known tempo and bars, and the size of a producer's bounce
    st, truth = track(bpm=124.0, offset=0.37, pickup_beats=1, bars=180, seed=3)
    mix = sum(st.values()); wp = os.path.join(OUT, "synthetic.wav"); wav_file(mix, wp)
    runs.append(one("synthetic", open(wp, "rb").read(), truth, keep_render=True))
    # a broken beat at drum and bass tempo: kicks on one and the and of two, so the grid cannot lean on the kick
    st, truth = track(bpm=172.0, offset=0.81, pickup_beats=2, bars=96, seed=5, style="broken")
    wp = os.path.join(OUT, "synthetic-broken.wav"); wav_file(sum(st.values()), wp)
    runs.append(one("synthetic broken beat", open(wp, "rb").read(), truth))
    from concurrent.futures import ThreadPoolExecutor
    def rec_run(rec):
        try:
            data = http("get", rec["preview"], timeout=60).content
        except Exception as e:
            return {"label": rec["name"], "error": f"preview {type(e).__name__}"}
        o = one(f"{rec['scene']}: {rec['name']}", data)
        o["catalogue_tempo"] = [rec.get("tempo"), rec.get("tempo2")]
        return o
    with ThreadPoolExecutor(4) as ex:
        runs += list(ex.map(rec_run, json.load(open(os.path.join(os.path.dirname(__file__), "swap-test-records.json")))))
    results["runs"] = runs; results["dropped"] = DROPS
    json.dump(results, open(os.path.join(OUT, "swap-test.json"), "w"), indent=1)
    (warn if DROPS else note)("swap dropped calls", f"{len(DROPS)} failed and were tried again" + ((": " + " || ".join(DROPS))[:1500] if DROPS else ""))
    for o in runs:
        if o.get("error"): continue
        msg = {k: o.get(k) for k in ("tempo", "catalogue_tempo", "tempo_error_pct", "downbeat_error_beats", "read_s", "upload_s", "probe_ms")}
        msg["phase_margin"] = (o.get("grid") or {}).get("phase_margin")
        msg["render"] = {k: (o.get("render") or {}).get(k) for k in ("s", "seconds", "vs_grid_ms")}
        msg["original_vs_grid_ms"] = o.get("original_vs_grid_ms"); msg["rough_tempo"] = (o.get("grid") or {}).get("rough_tempo")
        msg["grid_strength"] = (o.get("grid") or {}).get("strength"); msg["tracker_offset_ms"] = (o.get("grid") or {}).get("tracker_offset_ms")
        msg["low_lag_ms"] = (o.get("render") or {}).get("low_lag_vs_original_ms")
        msg["drums"] = (o.get("yours") or {}).get("drums", {}).get("loop")
        msg["others"] = {p: v.get("others") for p, v in (o.get("yours") or {}).items() if isinstance(v, dict)}
        msg["loopaudio"] = o.get("loopaudio"); msg["lufs"] = (o.get("render") or {}).get("lufs"); msg["mastering"] = (o.get("render") or {}).get("loudness")
        msg["last_beat_share"] = (o.get("render") or {}).get("last_beat_share"); msg["laid"] = (o.get("render") or {}).get("laid")
        msg["artist_bass_halfway"] = ((o.get("artist") or {}).get("parts") or {}).get("bass")
        msg["lines"] = o.get("lines"); msg["follow"] = o.get("follow"); msg["timing"] = (o.get("timing") or {}).get("lines")
        note(f"swap {o['label']}"[:60], json.dumps(msg)[:900])
        deep = {"focus": o.get("focus"), "kit": o.get("kit"), "groove": o.get("groove"), "stems": (o.get("render") or {}).get("stems"), "level": (o.get("render") or {}).get("level"), "render_groove": (o.get("render") or {}).get("groove")}
        note(f"swap deeper {o['label']}"[:60], json.dumps(deep)[:1800])


main()
