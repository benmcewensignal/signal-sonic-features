"""Does this Rubber Band take a frequency map (a pitch shift that changes during the file)? A two-second A2 sine, mapped
to stay for the first second and go up a fifth for the second; the output's pitch per half is measured with pyin.
  python tools/rb_freqmap_probe.py"""
import subprocess, tempfile, os, numpy as np, soundfile as sf, librosa
h = subprocess.run(["rubberband", "--help"], capture_output=True, text=True); txt = h.stdout + h.stderr
print("::notice title=rubberband version::" + (txt.splitlines()[0] if txt else "none"))
for L in txt.splitlines():
    if "freqmap" in L.lower() or "timemap" in L.lower() or "pitch" in L.lower(): print("  " + L.strip()[:200])
SR = 44100; t = np.arange(2 * SR) / SR; x = (0.5 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
w = tempfile.mkdtemp(); a, b, m = [os.path.join(w, n) for n in ("in.wav", "out.wav", "map.txt")]
sf.write(a, np.c_[x, x], SR, subtype="FLOAT")
open(m, "w").write(f"0 1.0\n{SR} {2 ** (7 / 12):.6f}\n")
for args in (["-3", "--freqmap", m], ["-3", "--freqmap", m, "-p", "0"], ["--freqmap", m]):
    p = subprocess.run(["rubberband", "-q"] + args + [a, b], capture_output=True, text=True)
    if p.returncode != 0 or not os.path.exists(b): print(f"  {' '.join(args[:2])}: failed: {(p.stderr or p.stdout)[-300:]}"); continue
    y, sr = sf.read(b, always_2d=True, dtype="float32"); y = y.mean(1)
    f0, v, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr, frame_length=2048, hop_length=256)
    n = len(f0) // 2; p1 = np.nanmedian(f0[: n - 10]); p2 = np.nanmedian(f0[n + 10:])
    print(f"::notice title=freqmap {' '.join(args[:2])}::first half {p1:.1f} Hz, second half {p2:.1f} Hz (want 110 and 164.8); length {len(y)/sr:.3f} s")
    os.remove(b)
