"""Swing: where the hi-hat lands between the beats.

Two earlier attempts failed on real records, each in an instructive way. Taking the median of
every onset in the middle of the beat returned the centre of its own search window, because a
real record carries a dozen onsets a beat and the median of a crowd is a property of the
crowd. Picking the loudest hat per beat scattered, because on most beats something else is
louder.

What works is to fold every beat onto one and look at where the energy in the hi-hat band
actually sits. One noisy beat cannot move it, and a position the record keeps returning to
shows up as a peak.

Two details matter. The band: swing is carried above about four kilohertz, and below that the
kick and the bass dominate any onset detector. And the envelope: plain spectral flux on that
band, not a decibel-scaled onset strength, which compresses the hat until it reads six
hundredths of a beat late.
"""
import numpy as np


def swing(y, sr, bpm=None):
    import librosa
    try:
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        band = S[freqs >= 4000]
        if band.size == 0:
            return {"swing_error": "no high band"}
        flux = np.maximum(0.0, np.diff(np.r_[0.0, band.sum(0)]))
        if flux.max() <= 0:
            return {"swing_error": "no energy in the hat band"}
        times = librosa.frames_to_time(np.arange(len(flux)), sr=sr, hop_length=256)

        full = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
        if bpm is None or not (60 < float(bpm) < 220):
            bpm = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=full, sr=sr))[0])
        if not (60 < bpm < 220):
            return {"swing_error": "no usable tempo"}
        _, beats = librosa.beat.beat_track(onset_envelope=full, sr=sr, bpm=bpm, units="time")
        beats = np.asarray(beats, float)
        if len(beats) < 8:
            return {"swing_error": "too few beats"}
        gaps = np.diff(beats)
        ok = (gaps > 0.15) & (gaps < 1.5)
        if ok.sum() < 6:
            return {"swing_error": "beat grid unstable"}

        # Fold: every beat laid on top of every other. Sixty slices put the whole corpus on
        # eighteen distinct values, a lattice with steps wider than the difference between a
        # swung genre and a straight one, which is the same fault the tempo estimator had.
        B = 240
        # A beat holds about forty analysis frames, so counting frames into two hundred and
        # forty bins leaves most of them empty. Resample each beat onto the grid instead:
        # every beat contributes a full curve, and the fold is smooth however fine the grid.
        prof = np.zeros(B)
        grid = (np.arange(B) + 0.5) / B
        used = 0
        for b0, g, good in zip(beats[:-1], gaps, ok):
            if not good:
                continue
            sel = (times >= b0) & (times < b0 + g)
            if sel.sum() < 8:
                continue
            ph = (times[sel] - b0) / g
            prof += np.interp(grid, ph, flux[sel], left=0.0, right=0.0)
            used += 1
        if used < 6:
            return {"swing_error": "too few usable beats"}
        prof = prof / used

        # Rotate so the loudest slice sits at zero: beat_track finds the rate reliably and
        # the phase only sometimes. But the loudest slice is not always the kick, and on a
        # record with a heavy hat it is the hat, which puts the kick at one minus the swing
        # and reads every shuffle as its own mirror. Swing is late by definition, so a
        # reading before the halfway point means we anchored on the offbeat: take the
        # complement, which is where the beat actually was.
        prof = np.roll(prof, -int(np.argmax(prof)))
        lo, hi = int(B * 0.22), int(B * 0.80)
        seg = prof[lo:hi]
        if seg.size < 4:
            return {"swing_error": "window too small"}
        j = int(np.argmax(seg))
        # Interpolate between bins. A peak sits somewhere inside its slice, and taking the
        # slice centre throws that away: fit a parabola through the peak and its neighbours
        # and take its vertex, which recovers a position finer than the grid.
        if 0 < j < len(seg) - 1:
            yl, y0, yr = float(seg[j - 1]), float(seg[j]), float(seg[j + 1])
            denom = yl - 2.0 * y0 + yr
            off = 0.5 * (yl - yr) / denom if abs(denom) > 1e-12 else 0.0
            off = max(-0.5, min(0.5, off))
        else:
            off = 0.0
        sw = (lo + j + 0.5 + off) / B
        if sw < 0.5:
            sw = 1.0 - sw
        # How much the peak stands above the rest of the offbeat window, not above the whole
        # beat: the beat includes near-silent stretches, so a median taken across it is tiny
        # and every record looked like it had a towering hat, including records with none.
        others = np.delete(seg, j)
        base = float(np.median(others)) if others.size else float(np.median(prof))
        spread = float(np.std(others)) if others.size > 2 else 0.0
        lift = float((seg[j] - base) / (spread + 1e-9))
        # a straight record has no offbeat peak worth the name; calling its noise a shuffle
        # is the failure this measure exists to avoid
        if lift < 1.2:
            return {"swing": 0.5, "swing_lift": round(lift, 4), "swing_n": int(used)}
        # swing_grip saturated at one for every scene in the corpus, so it carried nothing.
        # Report how much the offbeat peak stands above the rest of the beat directly, on a
        # log scale so a record with ten times the lift is not ten times the number.
        # No confidence number is reported. Two attempts at one read the same for a record
        # with a hat and a record without, and a field that cannot tell those apart is worse
        # than no field: it invites a reader to trust a reading we cannot vouch for. The raw
        # lift is here for anyone checking the instrument, and is not shown on the site.
        return {"swing": round(float(sw), 4), "swing_lift": round(lift, 4),
                "swing_n": int(used)}
    except Exception as e:
        return {"swing_error": f"{type(e).__name__}: {str(e)[:50]}"}
