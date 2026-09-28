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
def read_parts(wav: bytes, with_audio: bool = False) -> dict:
    import tempfile
    from worker.parts import read
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(wav); f.flush()
        return read(f.name, with_audio)



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
    return {"id": read_parts.spawn(body, with_audio).object_id}


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
