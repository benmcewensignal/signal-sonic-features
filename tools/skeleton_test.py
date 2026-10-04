"""The reader's skeleton on a loop whose answer is known: 124 BPM, a kick on every beat, a snare on 2 and 4, hats on
the offbeats 0.1 of a sixteenth late, and a bass note a sixteenth after each kick. The loop goes to the deployed
reader (sonic-parts, read_parts) exactly as an upload does, and the skeleton that comes back is printed against the truth."""
import io, json, wave, numpy as np, modal
sr = 32000; bpm = 124.0; beat = 60 / bpm; six = beat / 4; dur = 40; n = int(sr * dur); y = np.zeros(n); rng = np.random.default_rng(3)
def add(t, sig):
    i = int(t * sr); j = min(n, i + len(sig))
    if i < n: y[i:j] += sig[:j - i]
tt = np.arange(int(0.25 * sr)) / sr; kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-tt * 40)) * tt) * np.exp(-tt * 9) * 0.9
nh = int(0.05 * sr); hat = np.diff(np.concatenate([[0], rng.standard_normal(nh) * np.exp(-np.arange(nh) / sr * 90) * 0.35]))
ns = int(0.18 * sr); sn = (rng.standard_normal(ns) * 0.5 + np.sin(2 * np.pi * 200 * np.arange(ns) / sr) * 0.5) * np.exp(-np.arange(ns) / sr * 18) * 0.6
tb = np.arange(int(0.2 * sr)) / sr; bass = np.sin(2 * np.pi * 55 * tb) * np.minimum(1, tb * 200) * np.exp(-tb * 6) * 0.7
t0 = 0.5
for b in range(int((dur - 1) / (4 * beat))):
    for s in range(16):
        t = t0 + (b * 16 + s) * six
        if s % 4 == 0: add(t, kick)
        if s in (4, 12): add(t, sn)
        if s in (2, 6, 10, 14): add(t + 0.1 * six, hat)
        if s % 4 == 1: add(t, bass)
y = y / np.max(np.abs(y)) * 0.8; buf = io.BytesIO()
with wave.open(buf, "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((y * 32767).astype(np.int16).tobytes())
app = modal.App("sonic-skeleton-test")
@app.local_entrypoint()
def main():
    read = modal.Function.from_name("sonic-parts", "read_parts")
    out = read.remote(buf.getvalue())
    sk = (out or {}).get("skeleton")
    print("SKELETON", json.dumps(sk))
    if not sk or "occ" not in sk: print("no skeleton:", json.dumps(out)[:400]); return
    print(f"tempo {sk.get('tempo')} (true 124) | kick {sk.get('kick')} (true K...K...K...K...) | bars {sk.get('bars')}")
    print("hats where (share of bars):", [round(x, 2) for x in sk["occ"]], "(true 1 on steps 2, 6, 10, 14)")
    print("hats late:", [round(x, 2) for x in sk["rel"]], "(true +0.10 on steps 2, 6, 10, 14)")
    if "bass_occ" in sk: print("bass starts:", [round(x, 2) for x in sk["bass_occ"]], f"| {sk.get('bass_per_bar')} a bar | bleed {sk.get('bleed_share')} (true: one note a beat on steps 1, 5, 9, 13)")
