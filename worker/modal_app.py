"""The parts worker on Modal: separation and our own measures, in one container.

Two HTTP endpoints, both requiring the shared key the site holds (PARTS_KEY):
  POST /submit  body: a WAV clip          -> {"id": call id}
  GET  /result?id=...                     -> {"status": "running"} or {"status": "done", "result": {...}}
The clip lives only in memory and a temporary directory for the length of one call.
Deploy: modal deploy worker/modal_app.py (the worker-deploy workflow does this on push).
"""
import os, modal

app = modal.App("sonic-parts")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("ffmpeg", "libsndfile1")
         .pip_install("numpy<2", "librosa==0.10.2", "soundfile", "demucs==4.0.1", "torch==2.3.1",
                      "torchaudio==2.3.1", "essentia-tensorflow", "fastapi[standard]", "scikit-learn==1.8.0")
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
    import json as _j, numpy as _np, os as _os
    p = "/rx/classics/meta.json"
    if not _os.path.exists(p): return None
    st = _os.stat(p).st_mtime
    if _RX.get("at") != st:
        _RX.update({"at": st, "H": _np.load("/rx/classics/H.npy", mmap_mode="r"), "T": _np.load("/rx/classics/T.npy", mmap_mode="r"),
                    "F": _np.load("/rx/classics/F.npy", mmap_mode="r"), "meta": _j.load(open(p))})
    return _RX

def rx_match(query, R):
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
    key = tr * 100000 + (off + 50000); u, c = _np.unique(key, return_counts=True); ut = u // 100000
    order = _np.lexsort((-c, ut)); ut_o, c_o = ut[order], c[order]
    first = _np.r_[True, ut_o[1:] != ut_o[:-1]]
    best_t, best_c = ut_o[first], c_o[first]
    second = _np.zeros_like(best_c); idx = _np.flatnonzero(first)
    for k, i in enumerate(idx):
        if i + 1 < len(ut_o) and ut_o[i + 1] == ut_o[i]: second[k] = c_o[i + 1]
    o2 = _np.argsort(-best_c); top = o2[0]; tq = R["meta"]["tracks"][int(best_t[top])][4]
    # versions of one classic (original, extended, radio edit) all match the same audio: the next best that counts against
    # the top is the best different classic, not another version of the same one
    nxt = next((best_c[k] for k in o2[1:] if R["meta"]["tracks"][int(best_t[k])][4] != tq or not tq), 0)
    v, ru = int(best_c[top]), int(second[top])
    ok = v >= 8 and v >= 1.8 * max(ru, 1) and v >= 1.4 * max(int(nxt), 1)
    m = R["meta"]["tracks"][int(best_t[top])]
    return {"found": bool(ok), "track_id": m[0], "name": m[1], "artists": m[2], "scene": m[3], "votes": v, "runner_up": ru, "next_best": int(nxt),
            "hashes_in_index": int(keep.sum()), "records": len(R["meta"]["tracks"])}

@app.function(image=light, secrets=[secret], cpu=1.0, memory=2048, scaledown_window=300, volumes={"/rx": rvol})
@modal.fastapi_endpoint(method="POST")
async def recognise(request: Request):
    if not _ok(request): return JSONResponse({"error": "unauthorised"}, 401)
    try:
        body = await request.json(); query = (body or {}).get("hashes") or []
        rvol.reload(); R = _rx_load()
        if R is None: return {"found": False, "why": "no classics index yet"}
        return rx_match(query[:20000], R)
    except Exception as e:
        return JSONResponse({"error": type(e).__name__ + ": " + str(e)[:120]}, 500)

