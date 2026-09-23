"""Groove: where the hits actually land against where a metronome would put them.

Our swing measure scores a coin flip between techno and house, which are the two families
the dance floor tells apart most easily. That is an instrument failure, not a fact about the
music: it clips for a sixth of all records and has eighty-one distinct values across sixteen
thousand. Pulse clarity measures how peaked the tempogram is, which is metronome-regularity,
not microtiming.

What house has and techno does not is swing: the offbeat sits consistently late. So measure
that directly. Find the beat grid, find the onsets, and ask where the eighth notes between
the beats actually fall.

  swing        how late the offbeat sits, as a share of the gap to the next beat. 0.50 is
               dead straight, 0.58 is a light shuffle, 0.66 is triplet swing.
  swing_grip   how consistently it does that. A record can average a shuffle by wobbling
               either side of straight, which is a different thing from swinging.
  push         whether hits land ahead of or behind the grid overall: drag against rush.
  grid_grip    how tightly everything sits to the grid at all: machine against hand.
"""
import numpy as np


def groove(y, sr, bpm=None):
    import librosa
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
        if bpm is None or not (60 < float(bpm) < 220):
            bpm = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=onset_env, sr=sr))[0])
        if not (60 < bpm < 220):
            return {"groove_error": "no usable tempo"}
        _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, bpm=bpm, units="time")
        if len(beats) < 8:
            return {"groove_error": "too few beats"}
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr,
                                                  backtrack=True)
        onsets = librosa.frames_to_time(onset_frames, sr=sr)
        # how strong each onset is: the anchor has to find the kick, and a kick and a hi-hat
        # are equally numerous. Counting them equally picked the hat half the time and read
        # a swung record as dragging instead of swinging.
        strength = onset_env[np.clip(onset_frames, 0, len(onset_env) - 1)]
        if len(onsets) < 16:
            return {"groove_error": "too few onsets"}

        beats = np.asarray(beats, dtype=float)
        onsets = np.asarray(onsets, dtype=float)
        gaps = np.diff(beats)
        good = (gaps > 0.15) & (gaps < 1.5)
        if good.sum() < 6:
            return {"groove_error": "beat grid unstable"}

        phases, weights = [], []
        for b0, g, ok in zip(beats[:-1], gaps, good):
            if not ok:
                continue
            sel = (onsets >= b0 - g * 0.12) & (onsets < b0 + g * 1.12)
            for o, wt in zip(onsets[sel], strength[sel]):
                p = (o - b0) / g                      # 0 is the beat, 0.5 the exact offbeat
                if -0.12 <= p <= 1.12:
                    phases.append(p % 1.0)
                    weights.append(float(wt))
        if len(phases) < 24:
            return {"groove_error": "too few placed onsets"}
        phases = np.asarray(phases)

        # beat_track finds the rate but not reliably the downbeat, so the whole grid can sit
        # off by a fraction of a beat and every reading with it. Re-anchor on the densest
        # cluster of onsets, which is the kick: that cluster becomes phase zero.
        weights = np.asarray(weights)
        hist, edges = np.histogram(phases, bins=48, range=(0.0, 1.0), weights=weights)
        # circular smoothing, so a cluster straddling the wrap is not split in two
        sm = np.convolve(np.r_[hist[-3:], hist, hist[:3]], np.ones(5) / 5.0, mode="same")[3:-3]
        shift = (np.argmax(sm) + 0.5) / 48.0
        phases = (phases - shift) % 1.0
        ontime = [min(p, 1 - p) for p in phases]

        # the offbeat population: everything in the middle half of the gap
        off = phases[(phases > 0.28) & (phases < 0.78)]
        swing = float(np.median(off)) if off.size >= 8 else 0.5
        # swing is late by definition: an offbeat sits at or after the halfway point, never
        # before it. A reading below one half means the anchor found the offbeat rather than
        # the beat, and every phase is the mirror of the truth. Flip the grid and redo.
        if off.size >= 8 and swing < 0.5 - 1e-9:
            phases = (phases - swing) % 1.0
            off = phases[(phases > 0.28) & (phases < 0.78)]
            swing = float(np.median(off)) if off.size >= 8 else 0.5
            if swing < 0.5:
                swing = 1.0 - swing
                off = 1.0 - off
        grip = float(1.0 - min(np.std(off) / 0.14, 1.0)) if off.size >= 8 else 0.0

        # push: hits near the beat, are they early or late
        onb = phases[(phases < 0.18) | (phases > 0.82)]
        onb = np.where(onb > 0.5, onb - 1.0, onb)
        push = float(np.median(onb)) if onb.size >= 8 else 0.0

        grid = float(1.0 - min(float(np.mean(ontime)) / 0.25, 1.0))
        return {"swing": round(swing, 4), "swing_grip": round(grip, 3),
                "push": round(push, 4), "grid_grip": round(grid, 3),
                "groove_n": int(len(phases))}
    except Exception as e:
        return {"groove_error": f"{type(e).__name__}: {str(e)[:50]}"}
