"""Follow-your-notes, checked on synthetic parts where the truth is known (worker/swap.py: notes_of, hits_of, loop_kit,
follow_tonal, follow_drums). A sawtooth bass loop at one note is re-pitched to a written line; the output's note per
sixteenth is read back with pyin. A drum loop's kick and snare are placed on a written pattern; the output's hits are
read back with hits_of. Needs rubberband-cli (the frequency map). Annotations; exit 1 when a check fails.
  python tools/swap_follow_test.py"""
import sys, os, json, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worker import swap as SW
SR = SW.SR; TEMPO = 125.0; BAR = 240 / TEMPO; STEP = BAR / 16
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
def nm(n): return "." if n is None else NAMES[n % 12] + str(n // 12 - 1)
fails = []
def check(ok, msg):
    print(("::notice" if ok else "::error") + " title=follow::" + msg, flush=True)
    if not ok: fails.append(msg)

def saw(midi, secs, amp=0.4, att=0.005, dec=4.0):
    f = 440 * 2 ** ((midi - 69) / 12); t = np.arange(int(secs * SR)) / SR
    y = 2 * ((t * f) % 1) - 1; env = np.minimum(1, t / att) * np.exp(-t * dec)
    return (amp * y * env).astype(np.float32)

def tonal_loop(notes_per_step, bars=1):
    """A bass loop: a sawtooth pluck on each step that has a note."""
    n = int(round(bars * BAR * SR)); x = np.zeros(n, np.float32)
    for k, m in enumerate(notes_per_step):
        if m is None: continue
        s0 = int(round(k * STEP * SR)); y = saw(m, min(0.5, STEP * 1.5)); e = min(n, s0 + len(y)); x[s0:e] += y[: e - s0]
    return np.c_[x, x]

def drum_loop(kicks, snares, bars=1, seed=0, hat_amp=0.12, hat_every=2):
    rng = np.random.default_rng(seed); n = int(round(bars * BAR * SR)); x = np.zeros(n, np.float32)
    t = np.arange(int(0.25 * SR)) / SR
    kick = (0.9 * np.sin(2 * np.pi * (55 + 60 * np.exp(-t * 40)) * t) * np.exp(-t * 9)).astype(np.float32)
    sn = (0.5 * rng.standard_normal(len(t)) * np.exp(-t * 25) + 0.3 * np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30)).astype(np.float32)
    hat = (hat_amp * rng.standard_normal(int(0.04 * SR)) * np.exp(-np.arange(int(0.04 * SR)) / SR * 90)).astype(np.float32)
    from scipy.signal import butter, sosfiltfilt
    hat = sosfiltfilt(butter(4, 5000, "high", fs=SR, output="sos"), hat).astype(np.float32)
    for k in range(bars * 16):
        s0 = int(round(k * STEP * SR))
        for flags, smp in ((kicks, kick), (snares, sn)):
            if flags[k]: e = min(n, s0 + len(smp)); x[s0:e] += smp[: e - s0]
        if k % hat_every == hat_every - 1 or hat_every == 1: e = min(n, s0 + len(hat)); x[s0:e] += hat[: e - s0]
    return np.c_[x, x]

# --- the readers on what they will read
line = [33, None, 33, None, 36, None, 36, 38, 33, None, 33, None, 31, None, 29, None]   # A1 . A1 . C2 . C2 D2 A1 . A1 . G1 . F1 .
x = tonal_loop(line, 1); notes, on = SW.notes_of(x, "bass", 0.0, BAR, 1)
held = []; last = None   # a pluck rings into the next step, so the note sounding there is the one last struck
for v in line: last = v if v is not None else last; held.append(last)
agree = sum(1 for a, b in zip(notes, held) if a == b)
check(agree >= 14, f"notes_of reads the written bass line on {agree} of 16 steps: " + " ".join(nm(v) for v in notes))
check(sum(on) >= 8, f"notes_of marks {sum(on)} note starts of 10 written (a repeated note is one run)")
K = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]; S_ = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
d = drum_loop(K, S_, 1); kk, ss = SW.hits_of(d, 0.0, BAR, 1)
check(kk == K and ss == S_, f"hits_of reads the written pattern: kicks {''.join(map(str, kk))} snares {''.join(map(str, ss))}")
kit = SW.loop_kit(d, TEMPO)
check(kit["kick"] is not None and kit["snare"] is not None and len(kit["kick"]) > 0.1 * SR, f"loop_kit finds a kick ({None if kit['kick'] is None else round(len(kit['kick']) / SR, 3)} s) and a snare ({None if kit['snare'] is None else round(len(kit['snare']) / SR, 3)} s)")

# --- following: the loop plays A1 on every other step; your line is written over four bars
loop = tonal_loop([33, None] * 8, 1); lnotes = SW.loop_notes(loop, "bass", TEMPO, 1.0)
your = []
for b in range(4):
    your += [[33, 33, 36, 36, 38, 38, 40, 40, 33, 33, 31, 31, 29, 29, 28, 28], [45, 45, 48, 48, 43, 43, 45, 45, 41, 41, 40, 40, 38, 38, 36, 36]][b % 2]
out, info = SW.follow_tonal(loop, 1.0, lnotes, your, TEMPO, 4, "bass")
got, _ = SW.notes_of(out, "bass", 0.0, BAR, 4)
# the loop sounds on even steps only; your line is moved whole octaves to the loop's register
off = info.get("octave", 0); want = [(v + off) if v is not None else None for v in your]
pairs = [(g, w) for k, (g, w) in enumerate(zip(got, want)) if k % 2 == 0 and w is not None]
hit = sum(1 for g, w in pairs if g is not None and abs(g - w) <= 1)
check(hit >= 0.85 * len(pairs), f"follow_tonal: {hit} of {len(pairs)} sounding steps within a semitone of your line (octave {off}, shifts {info.get('shift_semitones')}); read back: " + " ".join(nm(v) for v in got[:16]))
check(abs(len(out) / SR - 4 * BAR) < 0.01, f"follow_tonal keeps time: {len(out) / SR:.3f} s for four bars of {BAR:.3f}")

# --- drums: the loop's kit on your pattern
yk = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0] * 2; ys = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] * 2
dout, dinfo = SW.follow_drums(d, kit, yk, ys, TEMPO, 2)
rk, rs = SW.hits_of(dout, 0.0, BAR, 2)
check(rk == yk, f"follow_drums: kicks read back as written: {''.join(map(str, rk))} (placed {dinfo['placed']})")
check(rs == ys, f"follow_drums: snares read back as written: {''.join(map(str, rs))}")
# --- drums: a loop's own pattern read for kit mode, and a kit from two loops (their pattern)
kk2, ss2 = SW.loop_pattern(d, TEMPO, 1.0, 2)
check(kk2 == K * 2 and ss2 == S_ * 2, f"loop_pattern reads the loop's own pattern over two bars: kicks {''.join(map(str, kk2))}")

# --- their groove on your drums: a straight pattern re-timed to each measured scene feel, read back
straight = drum_loop([1, 0, 0, 0] * 32, [0, 0, 0, 0, 1, 0, 0, 0] * 16, 8, seed=3, hat_amp=0.3, hat_every=1)   # a hat on every sixteenth, loud enough to read under the kick
worst = 0.0; rms_all = []; beats_off = 0.0; scenes = ["house", "uk-garage-speed-garage", "techno-peak-time", "amapiano", "drum-and-bass"]
t_in, _ = SW.hit_times(straight); _yf, cnt_in = SW.your_feel(t_in, 0.0, BAR, 8)   # a hat right after the kick sits under its tail and is not read: those stay
read_steps = [k for k in range(16) if cnt_in[k] >= 4]
check(len(read_steps) >= 12, f"groove: the reader hears the straight loop's hats on {len(read_steps)} of 16 steps (those under the kick's tail are not read, and stay)")
for sc in scenes:
    gf = SW.groove_feel(sc); y, ginfo = SW.regroove(straight, 0.0, BAR, 8, gf["rel"])
    t2, _ = SW.hit_times(y); yf, cnt = SW.your_feel(t2, 0.0, BAR, 8)
    err = [abs(yf[k] - (0 if k % 4 == 0 else gf["rel"][k])) * STEP * 1000 for k in read_steps]
    worst = max(worst, max(err)); rms_all.append(float(np.sqrt(np.mean(np.square(err))))); beats_off = max(beats_off, max(abs(yf[k]) * STEP * 1000 for k in (0, 4, 8, 12)))
    check(max(err) <= 6.0 and len(y) == len(straight), f"groove {sc}: hits read back within {max(err):.1f} ms of the scene's feel (rms {rms_all[-1]:.1f} ms; {ginfo['moved']} moved, up to {ginfo['max_move_ms']} ms; beats within {max(abs(yf[k]) * STEP * 1000 for k in (0, 4, 8, 12)):.1f} ms)")
check(beats_off <= 3.0, f"groove: the beats stay put (within {beats_off:.1f} ms) across {len(scenes)} scenes")

print("::notice title=follow::" + json.dumps({"tonal": info, "drums": dinfo}))
sys.exit(1 if fails else 0)
