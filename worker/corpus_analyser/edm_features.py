"""Production features an EDM analyser should have and ours does not.

Three measures from Xu, Dai, Goudet and Wang, "Acoustic Overspecification in Electronic
Dance Music Taxonomy" (arXiv 2509.11474), which found seventeen to twenty acoustic families
behind Beatport's thirty-five genres. Their feature design is better than ours in three
specific places and this module implements those, ready to fold into the analyser as 2.4
once the 2.3 re-measure has finished. Adding them now would leave the corpus measured at two
versions again, which is the fault that voided every comparison across the shard boundary.

  sidechain     the defining production technique of modern dance music, and we do not
                measure it at all
  sub_bass      power below 60 Hz. Our bass_weight saturates at 1.000 for almost every
                record, so it distinguishes nothing
  tempo_cyclic  octave ambiguity removed by construction rather than by the heuristic
                written yesterday, which still reads drum and bass at the wrong rate

Nothing here is called by the pipeline yet. Run the self-test before wiring it in.
"""
import numpy as np


def sidechain_pumping(y: np.ndarray, sr: int) -> float:
    """How hard the kick ducks everything above it.

    A compressor keyed to the kick pulls the synths down on every beat, so the low-band
    envelope and the high-band envelope move in opposition. The strength of that opposition
    is the pumping: near -1 is heavy sidechain, near 0 is none.
    """
    import librosa
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = S[freqs <= 120, :].sum(0)
    high = S[(freqs >= 2000) & (freqs <= 10000), :].sum(0)
    if low.size < 16 or low.std() == 0 or high.std() == 0:
        return 0.0
    low = (low - low.mean()) / low.std()
    high = (high - high.mean()) / high.std()
    # the duck lags the kick slightly: take the strongest opposition within a short window
    best = 0.0
    for lag in range(0, 6):
        a, b = (low[:-lag], high[lag:]) if lag else (low, high)
        if a.size < 16:
            continue
        r = float(np.dot(a - a.mean(), b - b.mean()) / ((np.linalg.norm(a - a.mean()) * np.linalg.norm(b - b.mean())) or 1))
        if r < best:
            best = r
    return round(best, 4)


def sub_bass_ratio(y: np.ndarray, sr: int) -> float:
    """Share of the record's power below 60 Hz: the part you feel rather than hear."""
    import librosa
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    total = S.sum()
    if total <= 0:
        return 0.0
    return round(float(S[freqs < 60, :].sum() / total), 4)


def tempo_cyclic(y: np.ndarray, sr: int, ref: float = 60.0) -> dict:
    """Tempo with the octave folded out, rather than guessed at.

    Every rate related by a power of two is pooled into one octave, so a record at 174 and
    its half-time feel at 87 land in the same bin. The reported tempo is then that bin
    mapped back into the range dance records are counted in.
    """
    import librosa
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    if onset.size < 32:
        return {"tempo_cyclic": 0.0, "pulse_clarity": 0.0}
    tg = librosa.feature.tempogram(onset_envelope=onset, sr=sr, hop_length=512)
    bpms = librosa.tempo_frequencies(tg.shape[0], sr=sr, hop_length=512)
    ok = np.isfinite(bpms) & (bpms > 20) & (bpms < 400)
    tg, bpms = tg[ok], bpms[ok]
    strength = tg.mean(axis=1)
    # fold onto one octave above the reference, then read the peak
    cyc = np.log2(np.maximum(bpms, 1e-9) / ref) % 1.0
    nbins = 120
    hist = np.zeros(nbins)
    idx = np.clip((cyc * nbins).astype(int), 0, nbins - 1)
    for i, s in zip(idx, strength):
        hist[i] += max(float(s), 0.0)
    if hist.sum() <= 0:
        return {"tempo_cyclic": 0.0, "pulse_clarity": 0.0}
    peak = int(np.argmax(hist))
    folded = ref * (2 ** (peak / nbins))
    while folded < 90:
        folded *= 2
    while folded > 190:
        folded /= 2
    clarity = float(hist.max() / (hist.mean() or 1))
    return {"tempo_cyclic": round(float(folded), 2), "pulse_clarity": round(clarity, 3)}


def measure(y: np.ndarray, sr: int) -> dict:
    out = {"sidechain": sidechain_pumping(y, sr), "sub_bass": sub_bass_ratio(y, sr)}
    out.update(tempo_cyclic(y, sr))
    return out
