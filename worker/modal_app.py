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


@app.function(image=image, cpu=4.0, memory=6144, timeout=420)
def read_parts(wav: bytes) -> dict:
    import tempfile
    from worker.parts import read
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(wav); f.flush()
        return read(f.name)



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
    return {"id": read_parts.spawn(body).object_id}


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
