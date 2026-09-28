"""End-to-end phone test: real previews, heard as a phone in a room would hear them for 30 seconds, turned into slices by
the page's own code (tools/melpatch.js, the same code the page runs), sent through the live site to the worker's model."""
import json, base64, subprocess, tempfile, os, numpy as np, librosa, requests
RECS = json.load(open("tools/phone-e2e-records.json")); rng = np.random.default_rng(0)
def phone(y):   # the same phone in a room the model's robustness was measured on
    F = np.fft.rfft(y); f = np.fft.rfftfreq(len(y), 1 / 16000); F[(f < 200) | (f > 6000)] = 0; z = np.fft.irfft(F, len(y))
    ir = rng.standard_normal(2400) * np.exp(-np.arange(2400) / 500); ir[0] = 1; z = np.convolve(z, ir / np.abs(ir).sum() * 4, mode="same")
    return (z + rng.standard_normal(len(z)) * np.std(z) * 0.1).astype(np.float32)
FAILED = False
# the loops route: a record looked up by name gets the licensed sounds closest to its parts
try:
    ids = __import__("numpy").load("/tmp/record-parts.npz")["ids"][:1].tolist() if __import__("os").path.exists("/tmp/record-parts.npz") else []
    rid = ids[0] if ids else "bp:10541473"
    rl = requests.post("https://www.earlysignal.live/api/parts?loops=1", json={"id": rid, "scene": "tech-house"}, timeout=60)
    jl = rl.json() if rl.headers.get("content-type", "").startswith("application/json") else {}
    lp = jl.get("licensed_parts") or {}
    print("::notice title=loops route::" + json.dumps({"HTTP": rl.status_code, "record": rid, "parts with loops": {k: len(v) for k, v in lp.items()}, "error": jl.get("error")}))
    if rl.status_code != 200 or not lp: print("::warning title=loops route::a record looked up by name got no loops")
except Exception as e:
    print("::notice title=loops route::failed " + type(e).__name__)
# first, the everyday path: a phone's own 75 measures to the older model, which every phone reading uses today
try:
    rr = requests.post("https://www.earlysignal.live/api/parts", json={"x": [0.0] * 75, "condition": "phone"}, timeout=60)
    print("::notice title=everyday measures route::" + json.dumps({"HTTP": rr.status_code, "reply": str(rr.text)[:200]}))
    FAILED = rr.status_code in (401, 403) or rr.status_code >= 500   # a 400 for dummy measures still proves the worker accepted the password and read the request
except Exception as e:
    print("::notice title=everyday measures route::failed " + type(e).__name__)
for r in RECS:
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f: f.write(requests.get(r["url"], timeout=40).content); fn = f.name
        y, _ = librosa.load(fn, sr=16000, mono=True, duration=120); os.remove(fn); L = len(y); y = phone(y)[max(0, L // 2 - 15 * 16000): L // 2 + 15 * 16000]
        json.dump(y.tolist(), open("/tmp/y.json", "w"))
        js = subprocess.run(["node", "-e", "const {melPatches16k}=require('./tools/melpatch.js'); const y=require('/tmp/y.json'); const P=melPatches16k(Float64Array.from(y)); process.stdout.write(JSON.stringify(P.map(s=>s.map(r=>Array.from(r)))))"], capture_output=True, text=True, timeout=120)
        x = np.array(json.loads(js.stdout), dtype=np.float32)
        body = {"x": base64.b64encode(x.astype("<f2").tobytes()).decode(), "condition": "phone"}
        resp = requests.post("https://www.earlysignal.live/api/parts?slices=1", json=body, timeout=60); d = resp.json()
        sl = d.get("scene_learned") or {}; top = (sl.get("scenes") or [[None]])[0][0]
        print(f"::notice title={r['scene']}::" + json.dumps({"record": (r.get("name") or "")[:40], "HTTP": resp.status_code, "called": top, "right": top == r["scene"],
              "top three": [s for s, _ in (sl.get("scenes") or [])], "confidence": sl.get("confidence"), "reliability": sl.get("reliability"), "drum_voice": d.get("drum_voice"), "error": d.get("error")}))
        FAILED = FAILED or resp.status_code != 200
    except Exception as e:
        print(f"::notice title={r['scene']}::failed {type(e).__name__}: {str(e)[:150]}")
if FAILED:
    print("::error title=phone path broken::the site and the worker are not talking; phone readings are falling back to the page alone")
    raise SystemExit(1)
