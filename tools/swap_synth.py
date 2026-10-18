"""A synthetic dance track with known beats and bars, as four parts (tools/swap_e2e.py writes its mix to test Swap)."""
import numpy as np
SR = 44100
def track(bpm=124.0, offset=0.37, pickup_beats=1, bars=48, seed=1, style="four"):
    rng = np.random.default_rng(seed)
    p = 60 / bpm; bar = 4 * p
    g0 = offset + pickup_beats * p           # the first downbeat
    n = int((g0 + bars * bar + 2) * SR)
    D, B, O, V = (np.zeros(n) for _ in range(4))
    def put(buf, t, x):
        i = int(round(t * SR)); m = min(len(x), n - i)
        if m > 0 and i >= 0: buf[i:i + m] += x[:m]
    tt = np.arange(int(0.35 * SR)) / SR
    kick = np.sin(2 * np.pi * (50 + 120 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 9)
    tc = np.arange(int(0.2 * SR)) / SR
    clap = rng.standard_normal(len(tc)) * np.exp(-tc * 25) * 0.5
    th = np.arange(int(0.05 * SR)) / SR
    hat = rng.standard_normal(len(th)) * np.exp(-th * 90) * 0.25
    notes = [45, 45, 48, 43]                  # A, A, C, G (midi) per bar, a four-bar cycle
    chords = [[57, 60, 64], [53, 57, 60], [55, 59, 62], [52, 55, 59]]
    def sawn(midi, dur, amp):
        f = 440 * 2 ** ((midi - 69) / 12); t = np.arange(int(dur * SR)) / SR
        x = 2 * ((t * f) % 1) - 1; return amp * x * np.minimum(1, t * 200) * np.exp(-t * 1.5)
    # pickup beats before the first downbeat: hats only
    for k in range(pickup_beats * 2):
        put(D, offset + k * p / 2, hat)
    for b in range(bars):
        t0 = g0 + b * bar
        drums_on = not (24 <= b < 32)         # a breakdown, bars 24-31
        bass_on = b >= 8 and not (24 <= b < 32)
        for q in range(4):
            tb = t0 + q * p
            if drums_on and style == "four":
                put(D, tb, kick)
                if q in (1, 3): put(D, tb, clap)
                put(D, tb + p / 2, hat)
            elif drums_on and style == "sixteenths":   # four on the floor under loud hats on the e and the a of each beat
                put(D, tb, kick)
                if q in (1, 3): put(D, tb, clap)
                for k16, amp in ((1, 1.6), (2, 0.5), (3, 1.6)): put(D, tb + k16 * p / 4, hat * amp)
            elif drums_on:   # a broken beat: kicks on one and the and of two, snares on two and four, hats on every eighth
                if q == 0: put(D, tb, kick)
                if q == 1: put(D, tb + p / 2, kick)
                if q in (1, 3): put(D, tb, clap)
                put(D, tb, hat * 0.6); put(D, tb + p / 2, hat)
            if bass_on:
                put(B, tb + p / 2, sawn(notes[b % 4] - 12, p / 2, 0.35))
        if b >= 4:
            for m in chords[(b // 2) % 4]:
                put(O, t0, sawn(m, bar * 0.95, 0.08))
        if b >= 16 and b % 8 in (0, 1):
            put(V, t0, sawn(69 + (b % 3), bar * 0.8, 0.05))
    st = {k: np.stack([x, x], 1).astype(np.float32) * 0.6 for k, x in (("drums", D), ("bass", B), ("other", O), ("vocals", V))}
    return st, {"bpm": bpm, "g0": g0, "bar": bar, "first_beat": offset}
