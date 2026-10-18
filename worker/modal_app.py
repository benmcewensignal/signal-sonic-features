"""The parts worker on Modal: separation and our own measures, in one container.

Two HTTP endpoints, both requiring the shared key the site holds (PARTS_KEY):
  POST /submit  body: a WAV clip          -> {"id": call id}
  GET  /result?id=...                     -> {"status": "running"} or {"status": "done", "result": {...}}
The clip lives only in memory and a temporary directory for the length of one call.
Deploy: modal deploy worker/modal_app.py (the worker-deploy workflow does this on push).
"""
import os, modal

app = modal.App("sonic-parts")
_built = (modal.Image.debian_slim(python_version="3.12")
          .apt_install("ffmpeg", "libsndfile1")
          .pip_install("numpy<2", "librosa==0.10.2", "soundfile", "demucs==4.0.1", "torch==2.3.1",
                       "torchaudio==2.3.1", "essentia-tensorflow", "fastapi[standard]", "scikit-learn==1.8.0"))
image = (_built
         .add_local_python_source("features")
         .add_local_dir("worker", "/root/worker"))   # the worker with its model file
secret = modal.Secret.from_name("sonic-parts")          # holds PARTS_KEY


@app.function(image=image, cpu=4.0, memory=6144, timeout=420, volumes={"/embed": modal.Volume.from_name("sonic-embed", create_if_missing=True)})
def read_parts(wav: bytes, with_audio: bool = False, donor: dict = None, only_leverage: bool = False, ab: dict = None) -> dict:
    import tempfile
    from worker.parts import read
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(wav); f.flush()
        return read(f.name, with_audio, donor, only_leverage, ab)



# imported plainly: the endpoint signatures need Request where modal deploy runs, not only in the image
from fastapi import Request
from fastapi.responses import JSONResponse


def _ok(request):
    return request.headers.get("authorization") == "Bearer " + os.environ.get("PARTS_KEY", "\0")


@app.function(image=image, secrets=[secret])
@modal.fastapi_endpoint(method="POST")
async def submit(request: Request):
    if not _ok(request): return JSONResponse({"error": "unauthorised"}, 401)
    body = await request.body()
    if not body or len(body) > 6_000_000: return JSONResponse({"error": "clip missing or too large"}, 400)
    # the separated parts come back as audio only when the listener has said the recording is theirs
    with_audio = request.query_params.get("parts_audio") == "1"
    # which part moves the track most toward its target: a donor record near the middle of the target, its tempo and the
    # target's centre in the learned ear (Aim a track sends them; only Beatport previews are fetched)
    qp = request.query_params; donor = None
    if qp.get("donor", "").startswith("https://geo-samples.beatport.com/"):
        try: donor = {"url": qp["donor"], "bpm": float(qp.get("dbpm") or 0), "id": qp.get("did", "")[:40], "centre": [float(x) for x in qp.get("centre", "").split(",")][:16]}
        except Exception: donor = None
    only = qp.get("lev_only") == "1" and donor is not None
    # a before-and-after: one part replaced by the licensed loop nearest the target's part (Aim a track sends the part,
    # the target's part centre in the matcher's raw space, and the track's tempo and key)
    ab = None
    if qp.get("ab") in ("drums", "bass", "other", "vocals"):
        try: ab = {"part": qp["ab"], "tc": [float(x) for x in qp.get("tc", "").split(",")][:53], "bpm": float(qp.get("tbpm") or 0), "key": (qp.get("tkey") or "")[:4] or None}
        except Exception: ab = None
    return {"id": read_parts.spawn(body, False if ab else with_audio, None if ab else donor, False if ab else only, ab).object_id}


@app.function(image=image, secrets=[secret])
@modal.fastapi_endpoint(method="POST")
async def numbers(request: Request):
    """The scene call from a device's own 75 measures: JSON {"x": [...]}, answered at once."""
    if not _ok(request): return JSONResponse({"error": "unauthorised"}, 401)
    body = await request.json()
    from worker.scene import call_from_numbers
    try:
        return call_from_numbers(body.get("x") or [], body.get("condition"))
    except Exception as e:
        return JSONResponse({"error": type(e).__name__}, 400)


@app.function(image=image, secrets=[secret], cpu=2.0, memory=3072, volumes={"/embed": modal.Volume.from_name("sonic-embed", create_if_missing=True)})
@modal.fastapi_endpoint(method="POST")
async def slices(request: Request):
    """The learned model's call from spectrogram slices a phone computed itself: JSON {"x": base64 of 8 x 96 x 188
    little-endian float16, "condition": "phone"}. The audio never leaves the phone; only these numbers do."""
    if not _ok(request): return JSONResponse({"error": "unauthorised"}, 401)
    import base64, numpy as np
    body = await request.json()
    try:
        x = np.frombuffer(base64.b64decode(body.get("x") or ""), dtype="<f2").astype(np.float32).reshape(8, 96, -1)
        from worker.embed import learned_from_slices
        return learned_from_slices(x, body.get("condition") or "phone") or JSONResponse({"error": "no model"}, 503)
    except Exception as e:
        return JSONResponse({"error": type(e).__name__ + ": " + str(e)[:80]}, 400)


@app.function(image=image, secrets=[secret], cpu=2.0, memory=4096, timeout=180, volumes={"/embed": modal.Volume.from_name("sonic-embed", create_if_missing=True)})
@modal.fastapi_endpoint(method="POST")
async def loopgaps(request: Request):
    """A label's loop against Catalogue's gaps: the audio as the body, ?family=drums|bass|melody|vocals and optional
    ?tempo= and ?key=. The loop lives only in a temporary file for the length of the call."""
    if not _ok(request): return JSONResponse({"error": "unauthorised"}, 401)
    import tempfile, os as _os
    q = request.query_params; body = await request.body()
    if len(body) > 8_000_000: return JSONResponse({"error": "loop too large"}, 413)
    d = tempfile.mkdtemp(); fp = _os.path.join(d, "loop.wav"); open(fp, "wb").write(body)
    try:
        from worker.scene import loop_gaps
        t = float(q.get("tempo")) if q.get("tempo") else None
        # a brief's gap records for this loop's part, and its defining records (What to make's "check your pack against this brief")
        clean = lambda x: [i for i in str(x or "").split(",") if i and len(i) <= 40 and all(c.isalnum() or c in ":-_" for c in i)]
        ids = clean(q.get("ids"))[:500] if q.get("ids") is not None else None
        core = clean(q.get("core"))[:12] if q.get("core") else None
        r = loop_gaps(fp, str(q.get("family") or ""), t, q.get("key") or None, ids=ids, core=core)
        return r if "error" not in r else JSONResponse(r, 400)
    except Exception as e:
        return JSONResponse({"error": type(e).__name__ + ": " + str(e)[:80]}, 400)
    finally:
        try: _os.remove(fp); _os.rmdir(d)
        except Exception: pass


@app.function(image=image, secrets=[secret], cpu=1.0, memory=2048, volumes={"/embed": modal.Volume.from_name("sonic-embed", create_if_missing=True)})
@modal.fastapi_endpoint(method="POST")
async def loops(request: Request):
    """The licensed loops closest to each part of a record Sonic has already separated, looked up by id:
    JSON {"id": track id, "scene": its scene}. No audio moves: the record's part measures are already stored."""
    if not _ok(request): return JSONResponse({"error": "unauthorised"}, 401)
    body = await request.json()
    try:
        from worker.scene import record_loops
        r = record_loops(str(body.get("id") or ""), body.get("scene"))
        return r if "error" not in r else JSONResponse(r, 404)
    except Exception as e:
        return JSONResponse({"error": type(e).__name__ + ": " + str(e)[:80]}, 400)


@app.function(image=image, secrets=[secret], cpu=1.0, memory=2048, volumes={"/embed": modal.Volume.from_name("sonic-embed", create_if_missing=True)})
@modal.fastapi_endpoint(method="POST")
async def compare(request: Request):
    """An uploaded track's parts (their measures, from its full reading) against a record Sonic has already separated:
    JSON {"id": the record, "parts": {part: 53 numbers}, "tempo": ..., "key": ...}. No audio moves."""
    if not _ok(request): return JSONResponse({"error": "unauthorised"}, 401)
    body = await request.json()
    try:
        from worker.scene import compare_parts, record_parts
        parts, tempo, key = body.get("parts") or {}, body.get("tempo"), body.get("key")
        if body.get("from_id") and not parts:   # "yours" can be a record Sonic has already separated, for a demonstration
            got = record_parts(str(body.get("from_id")))
            if "error" in got: return JSONResponse(got, 404)
            parts, tempo, key = got["parts"], got.get("tempo"), got.get("key")
        r = compare_parts(parts, str(body.get("id") or ""), tempo, key)
        return r if "error" not in r else JSONResponse(r, 404)
    except Exception as e:
        return JSONResponse({"error": type(e).__name__ + ": " + str(e)[:80]}, 400)


@app.function(image=image, secrets=[secret])
@modal.fastapi_endpoint(method="GET")
def result(request: Request, id: str):
    if not _ok(request): return JSONResponse({"error": "unauthorised"}, 401)
    fc = modal.FunctionCall.from_id(id)
    try:
        return {"status": "done", "result": fc.get(timeout=0)}
    except TimeoutError:
        return {"status": "running"}
    except Exception as e:
        return {"status": "failed", "error": type(e).__name__}

# ---- recognition in a room: the dense classics index (every fingerprint of each classic's preview, all versions) ----
# built by worker/recognise_build.py onto the sonic-recognise volume; this endpoint sleeps when idle and memory-maps the
# arrays when it wakes. The phone still sends only fingerprints. Same rule as the site's matcher: at least 8 votes at one
# offset, 1.8 times the record's own next offset, 1.4 times the next record.
light = modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "fastapi[standard]")
rvol = modal.Volume.from_name("sonic-recognise", create_if_missing=True)
_RX = {}

def _rx_load():
    # the catalogue-wide index (all/) when it exists, else the classics; the sorted fingerprints are read into memory on waking
    # (lookups touch them thousands of times per clip), the record and time arrays stay memory-mapped
    import json as _j, numpy as _np, os as _os
    d = "/rx/all" if _os.path.exists("/rx/all/ACTIVE") and _os.path.exists("/rx/all/meta.json") else "/rx/classics"   # the catalogue index goes live only once calibrated
    p = d + "/meta.json"
    if not _os.path.exists(p): return None
    st = _os.stat(p).st_mtime
    if _RX.get("at") != st:
        _RX.clear()
        _RX.update({"at": st, "which": d, "H": _np.load(d + "/H.npy"), "T": _np.load(d + "/T.npy", mmap_mode="r"),
                    "F": _np.load(d + "/F.npy", mmap_mode="r"), "meta": _j.load(open(p))})
    return _RX

def _rx_group(m):
    import re as _re, unicodedata as _ud
    def n(x): return _re.sub(r"\bu\b", "you", _re.sub(r"[^a-z0-9]+", " ", _ud.normalize("NFKD", str(x or "").replace("'", "").replace("\u2019", "")).encode("ascii", "ignore").decode().lower())).strip()
    title = n(_re.split(r"\s*[\(\[]|\s+feat\.?\s+|\s+ft\.?\s+", str(m[1] or ""), 1)[0])
    a0 = n((m[2] or [""])[0]).split(" ")[0] if m[2] else ""
    return a0 + "|" + title

def rx_match(query, R):
    """The same evidence as the site's matcher, with two changes the first live test asked for: a time bin counts with its
    neighbours (a match straddling a boundary kept splitting its votes), and a strong second offset of the same record no
    longer counts against it (dance records repeat, so a clip matches its record at several offsets). A record is named
    when its best offset has at least 10 votes and 1.4 times the best different classic; versions of one classic, grouped
    by first artist and title, never count against each other."""
    import numpy as _np
    q = _np.asarray(query, dtype=_np.int64)
    if q.ndim != 2 or q.shape[0] == 0: return {"found": False, "why": "no fingerprints"}
    qh = (q[:, 0] & 0xFFFFFFFF).astype(_np.uint32); qf = q[:, 1].astype(_np.int64)
    H = R["H"]; lo = _np.searchsorted(H, qh, "left"); hi = _np.searchsorted(H, qh, "right"); cnt = hi - lo
    keep = (cnt > 0) & (cnt <= 4000)          # a fingerprint shared by thousands of records says nothing
    if not keep.any(): return {"found": False, "hashes_in_index": 0}
    lo, cnt, qf = lo[keep], cnt[keep], qf[keep]; tot = int(cnt.sum())
    if tot > 3_000_000: return {"found": False, "why": "too many postings"}
    starts = _np.repeat(lo - _np.concatenate(([0], _np.cumsum(cnt)[:-1])), cnt) + _np.arange(tot)
    tr = _np.asarray(R["T"][starts], dtype=_np.int64); fr = _np.asarray(R["F"][starts], dtype=_np.int64)
    off = _np.rint((fr - _np.repeat(qf, cnt)) * (512 / 22050)).astype(_np.int64)
    key = tr * 100000 + (off + 50000); u, c = _np.unique(key, return_counts=True)
    # a bin and its two neighbours together
    cm = c.copy()
    for d in (-1, 1):
        k2 = u + d; ix = _np.searchsorted(u, k2); ok_ = (ix < len(u)) & (u[_np.minimum(ix, len(u) - 1)] == k2); cm[ok_] += c[ix[ok_]]
    ut = u // 100000; G = R.setdefault("groups", [_rx_group(m) for m in R["meta"]["tracks"]])
    def per_track(cc):
        order = _np.lexsort((-cc, ut)); ut_o, c_o = ut[order], cc[order]; first = _np.r_[True, ut_o[1:] != ut_o[:-1]]
        second = _np.zeros(int(first.sum()), dtype=_np.int64); idx = _np.flatnonzero(first)
        nxt_in = _np.r_[idx[1:], len(ut_o)]
        has2 = (idx + 1) < nxt_in; second[has2] = c_o[idx[has2] + 1]
        return ut_o[first], c_o[first], second
    def verdict(bt, bc, b2):
        o2 = _np.argsort(-bc); top = o2[0]; tg = G[int(bt[top])]
        nxt = next((int(bc[k]) for k in o2[1:] if G[int(bt[k])] != tg), 0)
        return int(bt[top]), int(bc[top]), int(b2[top]), nxt
    # the site's rule on single one-second bins (no wrong answer in the first live test) ...
    t1, v1, r1, n1 = verdict(*per_track(c))
    ok1 = v1 >= 10 and v1 >= 1.8 * max(r1, 1) and v1 >= 1.6 * max(n1, 1)   # an index of ~71,000 records needs a clearer margin: at 1.4 a catalogue record was named as another
    # ... or, for records whose loops match at several offsets, a bin with its neighbours, well clear of every other classic
    # (set from 243 queries of records outside the index against 180 of indexed classics)
    t2, v2, _r2, n2 = verdict(*per_track(cm))
    ok2 = v2 >= 30 and v2 >= 4.0 * max(n2, 1)   # outside records peaked at 3.6 times the next in testing; indexed classics never fell below 7.4
    tt = t1 if ok1 else t2; m = R["meta"]["tracks"][tt]
    return {"found": bool(ok1 or ok2), "track_id": m[0], "name": m[1], "artists": m[2], "scene": m[3], "votes": v1 if ok1 else v2,
            "runner_up": r1, "next_best": n1 if ok1 else n2, "path": "single" if ok1 else ("loops" if ok2 else None),
            "hashes_in_index": int(keep.sum()), "records": len(R["meta"]["tracks"])}

# 16 GB: the catalogue-wide index (all/) holds 1.3 billion sorted fingerprints, 5 GB read into memory on waking
@app.function(image=light, secrets=[secret], cpu=2.0, memory=16384, scaledown_window=300, volumes={"/rx": rvol})
@modal.fastapi_endpoint(method="POST")
async def recognise(request: Request):
    if not _ok(request): return JSONResponse({"error": "unauthorised"}, 401)
    try:
        body = await request.json(); query = (body or {}).get("hashes") or []
        if (body or {}).get("warm"): 
            if not _RX:
                try: rvol.reload()
                except Exception: pass
            _rx_load(); return {"warm": True}
        # reload the volume once, when this container wakes: the index is memory-mapped, and a reload with its files open fails
        # ("there are open files preventing the operation"); a new build is picked up when the service next sleeps and wakes
        if not _RX:
            try: rvol.reload()
            except Exception: pass
        R = _rx_load()
        if R is None: return {"found": False, "why": "no classics index yet"}
        return rx_match(query[:20000], R)
    except Exception as e:
        return JSONResponse({"error": type(e).__name__ + ": " + str(e)[:120]}, 500)



# ---- Swap: your track split into its parts, any part swapped for an openly licensed loop, the whole track to download ----
# worker/swap.py does the work. The page talks to one web app here directly (the files are larger than the site's
# functions accept), with a token the site signs with PARTS_KEY (api/swap.js in signalgood):
#   POST /start?t=&name=   body: the track's file      -> {"id"}       separation on a GPU, then the reading
#   GET  /result?id=&t=                                -> {"status", "result"} without audio
#   GET  /stem?id=&part=&t=                            -> MP3: one part over the listening window (part=probe: the click)
#   POST /loop?t=          JSON {part, target, tempo, key, exclude} -> the licensed loop, prepared, as WAV in JSON
#   POST /render?t=        form: file, plan            -> {"id"}       separates again and lays the plan's loops
#   GET  /file?id=&t=                                  -> MP3 of the whole track, swapped (an attachment)
#   GET  /artists?t=                                   -> [[artist, separated records], ...] for the search
# The track lives in memory and a temporary directory for the length of each call; nothing is written anywhere.
swap_gpu_image = (_built.apt_install("rubberband-cli")
                  .env({"TF_FORCE_GPU_ALLOW_GROWTH": "true"})
                  .add_local_python_source("features")
                  .add_local_dir("worker", "/root/worker"))
swap_light = (modal.Image.debian_slim(python_version="3.12")
              .apt_install("ffmpeg", "rubberband-cli", "libsndfile1")
              .pip_install("numpy<2", "soundfile", "fastapi[standard]")
              .add_local_dir("worker", "/root/worker"))
_embed = modal.Volume.from_name("sonic-embed", create_if_missing=True)


@app.function(image=swap_gpu_image, gpu="T4", cpu=4.0, memory=12288, timeout=900, volumes={"/embed": _embed})
def swap_read(data: bytes, name: str = "track") -> dict:
    import tempfile
    from worker.swap import read_track
    with tempfile.NamedTemporaryFile(suffix=".audio") as f:
        f.write(data); f.flush()
        try:
            return read_track(f.name, name)
        except Exception as e:
            return {"error": type(e).__name__ + ": " + str(e)[:200]}


@app.function(image=swap_gpu_image, gpu="T4", cpu=4.0, memory=12288, timeout=900, volumes={"/embed": _embed})
def swap_render(data: bytes, plan: dict) -> dict:
    import tempfile
    from worker.swap import render
    with tempfile.NamedTemporaryFile(suffix=".audio") as f:
        f.write(data); f.flush()
        try:
            return render(f.name, plan)
        except Exception as e:
            return {"error": type(e).__name__ + ": " + str(e)[:200]}


@app.function(image=swap_light, secrets=[secret], cpu=4.0, memory=6144, timeout=900, scaledown_window=600, volumes={"/embed": _embed})
@modal.concurrent(max_inputs=8)   # a loop costs downloads and stretches: past eight at once, another container takes the rest
@modal.asgi_app()
def swap():
    import asyncio, json as _json, re as _re
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response
    from fastapi.middleware.cors import CORSMiddleware
    from worker import swap as SW
    api = FastAPI()
    api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"],
                       expose_headers=["Content-Disposition"], max_age=3600)
    KEY = os.environ.get("PARTS_KEY", "")
    CACHE = {}

    def ok(req): return SW.token_scope(req.query_params.get("t"), KEY)
    def no(): return JSONResponse({"error": "unauthorised"}, 401)

    async def finished(id_):
        """The call's result once it has finished (kept here briefly so the page's next few requests do not fetch it
        again); None while it runs."""
        if id_ in CACHE: return CACHE[id_]
        fc = modal.FunctionCall.from_id(id_)
        try:
            r = await fc.get.aio(timeout=0)
        except TimeoutError:
            return None
        while len(CACHE) >= 12: CACHE.pop(next(iter(CACHE)))
        CACHE[id_] = r
        return r

    @api.post("/start")
    async def start(req: Request):
        if ok(req) != "t": return no()
        body = await req.body()
        if not body: return JSONResponse({"error": "no file"}, 400)
        if len(body) > 200_000_000: return JSONResponse({"error": "the file is larger than 200 MB"}, 413)
        fc = await swap_read.spawn.aio(body, (req.query_params.get("name") or "track")[:120])
        return {"id": fc.object_id}

    @api.get("/result")
    async def result(req: Request):
        if not ok(req): return no()
        try:
            r = await finished(req.query_params.get("id", ""))
        except Exception as e:
            return {"status": "failed", "error": type(e).__name__}
        if r is None: return {"status": "running"}
        if r.get("error"): return {"status": "failed", "error": r["error"]}
        return {"status": "done", "result": {k: v for k, v in r.items() if k not in ("audio", "mp3")}}

    @api.get("/stem")
    async def stem(req: Request):
        if not ok(req): return no()
        r = await finished(req.query_params.get("id", ""))
        a = ((r or {}).get("audio") or {}).get(req.query_params.get("part", ""))
        if not a: return JSONResponse({"error": "not ready"}, 404)
        return Response(content=a, media_type="audio/mpeg", headers={"Cache-Control": "private, max-age=3600"})

    @api.post("/loop")
    async def loop(req: Request):
        if not ok(req): return no()
        try:
            b = await req.json()
            r = await asyncio.to_thread(SW.loop_for, str(b.get("part") or ""), b.get("target") or {}, b.get("tempo"), b.get("key"),
                                        [str(x) for x in (b.get("exclude") or [])][:80], int(b.get("n") or 5) if 1 <= int(b.get("n") or 5) <= 8 else 5)
        except Exception as e:
            return JSONResponse({"error": type(e).__name__ + ": " + str(e)[:160]}, 400)
        return r if not r.get("error") else JSONResponse(r, 404)

    @api.post("/loopaudio")
    async def loopaudio(req: Request):
        """A listed loop's prepared audio, for the page's pick among the loops /loop listed."""
        if not ok(req): return no()
        try:
            b = await req.json()
            r = await asyncio.to_thread(SW.loop_audio, str(b.get("part") or ""), int(b.get("id")), b.get("f"), b.get("shift"), b.get("tempo"))
        except Exception as e:
            return JSONResponse({"error": type(e).__name__ + ": " + str(e)[:160]}, 400)
        return r if not r.get("error") else JSONResponse(r, 404)

    @api.post("/render")
    async def render_(req: Request):
        if not ok(req): return no()
        form = await req.form()
        f = form.get("file"); plan = form.get("plan")
        if plan is None: return JSONResponse({"error": "no plan"}, 400)
        if ok(req) == "e":
            # the example's key renders the example only, which the worker fetches from the site itself
            import urllib.request as _u
            data = await asyncio.to_thread(lambda: _u.urlopen(SW.EXAMPLE_URL, timeout=60).read())
        else:
            if f is None: return JSONResponse({"error": "no file"}, 400)
            data = await f.read()
        if len(data) > 200_000_000: return JSONResponse({"error": "the file is larger than 200 MB"}, 413)
        try:
            P = SW.clean_plan(_json.loads(plan))
        except Exception as e:
            return JSONResponse({"error": "plan: " + str(e)[:80]}, 400)
        fc = await swap_render.spawn.aio(data, P)
        return {"id": fc.object_id}

    @api.get("/file")
    async def file_(req: Request):
        if not ok(req): return no()
        r = await finished(req.query_params.get("id", ""))
        if r is None: return JSONResponse({"status": "running"}, 202)
        if not r.get("mp3"): return JSONResponse({"error": r.get("error") or "no file"}, 404)
        nm = _re.sub(r"[^A-Za-z0-9 ._()-]+", "", str(r.get("name") or "track"))[:80].strip() or "track"
        return Response(content=r["mp3"], media_type="audio/mpeg",
                        headers={"Content-Disposition": f'attachment; filename="{nm} (swapped).mp3"', "Cache-Control": "private, max-age=3600"})

    _ART = {}

    @api.get("/artists")
    async def artists(req: Request):
        if not ok(req): return no()
        if "list" not in _ART: _ART["list"] = await asyncio.to_thread(SW.artists_separated)
        return {"artists": _ART["list"]}

    return api
