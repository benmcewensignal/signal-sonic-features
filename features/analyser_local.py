"""LocalAnalyser — implementation B behind the same seam.

Runs entirely on-machine from an audio file (Beatport preview clips):
  - tempo, key estimate, 8-segment energy curve      (librosa DSP)
  - scalar proxies: drum_density (onset rate), drum_swing (onset-interval
    asymmetry), bass_weight (sub-band energy share), vocal_presence
    (mid-band spectral flatness heuristic)
  - embedding: statistics vector over MFCC / chroma / spectral contrast /
    mel-band energies, L2-normalised, EMBED_DIM dims

Deterministic, dependency-light, zero network. A CLAP-class model can
replace the embedding later behind the same interface; that is a new
analyser_id and a re-baselined history, per the vintage rule.

Honest scope note: the scalar proxies are heuristics, not ground truth.
The pilot's silent quarter exists to measure how they behave on real
material before any threshold trusts them.
"""
from __future__ import annotations
import math
import numpy as np
import librosa
import librosa.feature.rhythm
from .analyser import Analyser, FeatureVector, EMBED_DIM

_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Schmuckler key profiles
_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _decoder_fingerprint() -> str:
    """8-char hash of the audio decode stack. A silent decoder upgrade on the
    runner shifts MFCCs subtly; stamping it makes environment drift visible
    in provenance instead of appearing as fake sonic drift (the suspected
    mechanism behind the 2024-M12 chunk step)."""
    import hashlib
    import subprocess
    parts = []
    try:
        parts.append(subprocess.run(["ffmpeg", "-version"], capture_output=True,
                                    text=True, timeout=10).stdout.splitlines()[0])
    except Exception:
        parts.append("no-ffmpeg")
    try:
        import soundfile
        parts.append(soundfile.__libsndfile_version__)
    except Exception:
        parts.append("no-sndfile")
    parts.append(librosa.__version__); parts.append("emb45")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:8]


class LocalAnalyser(Analyser):
    analyser_id = "local"
    version = "2"          # 2: full 45-dim embedding, spectral contrast no longer truncated away

    def __init__(self, sr: int = 22050, max_seconds: float = 120.0):
        self.sr = sr
        self.max_seconds = max_seconds
        self.version = f"2+{_decoder_fingerprint()}"

    def analyse(self, audio_ref: str) -> FeatureVector:
        y, sr = librosa.load(audio_ref, sr=self.sr, mono=True,
                             duration=self.max_seconds)
        if y.size < sr:  # under a second of audio: refuse rather than guess
            raise ValueError(f"audio too short to analyse: {audio_ref}")
        y = y / (np.max(np.abs(y)) or 1.0)

        tempo = float(np.atleast_1d(
            librosa.feature.rhythm.tempo(y=y, sr=sr, aggregate=np.median))[0])
        key = self._key(y, sr)
        energy = self._energy_curve(y)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        drum_density = self._drum_density(onset_env, sr)
        drum_swing = self._swing(onset_env, sr)
        bass_weight = self._bass_weight(y, sr)
        vocal = self._vocal_presence(y, sr)
        emb = self._embedding(y, sr)
        rms = librosa.feature.rms(y=y)[0]
        loud = float(20 * np.log10(max(float(rms.mean()), 1e-6)))

        return FeatureVector(
            tempo=tempo, key=key, energy_curve=energy,
            drum_palette=[], drum_density=drum_density, drum_swing=drum_swing,
            bass_character=[], bass_weight=bass_weight,
            vocal_treatment=[], vocal_presence=vocal,
            mood=[], embedding=emb, loudness=loud,
            analyser_id=self.analyser_id, analyser_version=self.version)

    # -- features ----------------------------------------------------------
    def _key(self, y, sr) -> str:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
        chroma = chroma / (chroma.sum() or 1.0)
        best, best_r = "", -2.0
        for i in range(12):
            rolled = np.roll(chroma, -i)
            for prof, suffix in ((_MAJ, ""), (_MIN, "m")):
                r = float(np.corrcoef(rolled, prof / prof.sum())[0, 1])
                if r > best_r:
                    best_r, best = r, _KEYS[i] + suffix
        return best

    def _energy_curve(self, y) -> list[float]:
        rms = librosa.feature.rms(y=y)[0]
        segs = np.array_split(rms, 8)
        vals = np.array([float(s.mean()) for s in segs])
        top = vals.max() or 1.0
        return [float(v / top) for v in vals]

    def _drum_density(self, onset_env, sr) -> float:
        hop = 512
        peaks = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        secs = len(onset_env) * hop / sr
        rate = len(peaks) / (secs or 1.0)          # onsets per second
        return float(min(1.0, rate / 8.0))          # 8/s ≈ saturated

    def _swing(self, onset_env, sr) -> float:
        """Asymmetry of inter-onset intervals: straight grids score low,
        swung/shuffled patterns score higher."""
        peaks = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr,
                                           units="time")
        if len(peaks) < 8:
            return 0.0
        iois = np.diff(peaks)
        iois = iois[(iois > 0.05) & (iois < 1.0)]
        if len(iois) < 6:
            return 0.0
        med = np.median(iois)
        dev = np.abs(iois - med) / (med or 1.0)
        return float(min(1.0, np.median(dev) * 2.5))

    def _bass_weight(self, y, sr) -> float:
        S = np.abs(librosa.stft(y, n_fft=2048)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        total = S.sum() or 1.0
        sub = S[freqs < 120].sum()
        return float(min(1.0, (sub / total) * 4.0))  # 25% share ≈ saturated

    def _vocal_presence(self, y, sr) -> float:
        """Heuristic: harmonic energy share in the 300-3400 Hz band with low
        spectral flatness reads as voice-like. Crude; measured, not trusted."""
        y_h = librosa.effects.harmonic(y, margin=3.0)
        S = np.abs(librosa.stft(y_h, n_fft=2048)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        band = S[(freqs >= 300) & (freqs <= 3400)]
        share = band.sum() / (S.sum() or 1.0)
        flat = float(librosa.feature.spectral_flatness(y=y_h).mean())
        return float(min(1.0, share * (1.0 - min(1.0, flat * 8.0)) * 2.2))

    def _embedding(self, y, sr) -> list[float]:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        parts = [
            mfcc.mean(axis=1), mfcc.std(axis=1),          # 26
            chroma.mean(axis=1),                          # 12
            contrast.mean(axis=1),                        # 7
        ]
        v = np.concatenate(parts).astype(float)   # 13+13+12+7 = 45
        # per-dim robust scaling so no single family dominates
        v = (v - np.median(v)) / (np.percentile(np.abs(v - np.median(v)), 75) or 1.0)
        # v2: the old 32-dim cut kept the MFCCs, half the chroma and none of the spectral
        # contrast — the descriptor most tied to how produced a record sounds. Keep all 45.
        if v.size >= EMBED_DIM:
            v = v[:EMBED_DIM]
        else:
            v = np.pad(v, (0, EMBED_DIM - v.size))
        n = float(np.linalg.norm(v)) or 1.0
        return [float(x) for x in v / n]
