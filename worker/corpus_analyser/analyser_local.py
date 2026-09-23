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
    parts.append(librosa.__version__); parts.append("emb45"); parts.append("tempo-octave"); parts.append("rhythm16"); parts.append("perfamily"); parts.append("edm")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:8]


class LocalAnalyser(Analyser):
    analyser_id = "local"
    version = "2.9"        # 2: full 45-dim embedding. 2.1: tempo resolves the octave
                           # error. 2.2: rhythm vector. 2.3: each feature family scaled
                           # against itself, which brings the twelve chroma dimensions back

    def __init__(self, sr: int = 22050, max_seconds: float = 120.0):
        self.sr = sr
        self.max_seconds = max_seconds
        # the class version, not a literal. This line read f"2+..." and overwrote every
        # version bump made above it: 2.1 for the tempo octave fix, 2.2 for the rhythm
        # vector, 2.3 for per-family scaling. All three were committed, correct, and inert,
        # because the re-analysis compares a record's stored version against this one and
        # saw no difference. Fifty thousand records sat unmeasured behind one literal.
        self.version = f"{type(self).version}+{_decoder_fingerprint()}"

    def analyse(self, audio_ref: str) -> FeatureVector:
        y, sr = librosa.load(audio_ref, sr=self.sr, mono=True,
                             duration=self.max_seconds)
        if y.size < sr:  # under a second of audio: refuse rather than guess
            raise ValueError(f"audio too short to analyse: {audio_ref}")
        y = y / (np.max(np.abs(y)) or 1.0)

        tempo = self._tempo(y, sr)
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
            edm=self._edm(y, sr),
            tempo2=self._tempo2(y, sr),
            groove=self._groove(y, sr),
            rhythm_vector=self._rhythm_vector(y, sr),
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

    def _groove(self, y, sr) -> dict:
        """Where the hits land against the grid: swing, and how tightly it is held.

        Our drum_swing tells techno from house at fifty-one per cent, which is a coin flip
        between the two families a dance floor separates most easily. That is the instrument
        failing, not the music: it clips for a sixth of records and has eighty-one distinct
        values across sixteen thousand. This measures the offbeat directly.
        """
        try:
            from .groove import groove
            from .swing import swing as _swing
            g = groove(y, sr)
            # the groove module reads swing from every onset in the beat, which on a real
            # record is the centre of its own window. The swing module reads the hi-hat band
            # folded across all beats, and that one can tell garage from techno.
            s2 = _swing(y, sr)
            if isinstance(g, dict) and isinstance(s2, dict):
                g.pop("swing", None); g.pop("swing_grip", None)
                g.update(s2)
            return g
        except Exception as e:
            return {"groove_error": f"{type(e).__name__}: {str(e)[:50]}"}

    def _tempo2(self, y, sr) -> dict:
        """Tempo on a fine grid with the octave family scored rather than assumed.

        The field this replaces returns 29 distinct values across fifteen thousand records,
        so four different genres report a median of 129 and trance, which is reliably 138 to
        140, has no lattice point to land on. The tree's thresholds were midpoints between
        grid positions rather than musical boundaries.
        """
        try:
            from .tempo_resolved import tempo_resolved
            return tempo_resolved(y, sr)
        except Exception as e:
            return {"tempo2_error": f"{type(e).__name__}: {str(e)[:50]}"}

    def _edm(self, y, sr) -> dict:
        """Sidechain, sub-bass and a tempo with the octave folded out.

        From Xu et al., arXiv 2509.11474. Sidechain is the defining production technique of
        modern dance music and we did not measure it; sub-bass replaces a bass_weight that
        saturates at 1.000 for almost every record; the cyclic tempogram removes the octave
        ambiguity by construction rather than by the heuristic below it.

        Wrapped: a failure here costs one record, not a hundred-minute pass.
        """
        try:
            from .edm_features import measure
        except ModuleNotFoundError as e:
            # not a per-record failure. Sixteen thousand records were analysed with this
            # raising every time because the module was written and never committed, and
            # the per-record catch turned a broken build into a quiet field in the output.
            raise RuntimeError(
                "sonic/edm_features.py is missing from the deployment: "
                "every record would be measured without the production features") from e
        try:
            return measure(y, sr)
        except Exception as e:
            return {"edm_error": f"{type(e).__name__}: {str(e)[:60]}"}

    def _tempo(self, y, sr) -> float:
        """Beat rate, resolved against the octave error.

        A pulse is ambiguous by factors of two: a drum and bass record at 174 has a real,
        strongly autocorrelating half-time pulse at 87, and the detector was choosing it for
        the whole scene. Measured at 117 against a true 174, which made an entire genre
        look mid-tempo and fed a wrong number to the classifier.

        Autocorrelation cannot settle it, because both rates are genuinely present. What
        settles it is convention: dance music is counted at the faster pulse, and no scene
        we measure is counted below about ninety. So when doubling lands inside the range
        dance records actually occupy, and the onsets support it nearly as well, take it.
        """
        cands = np.atleast_1d(librosa.feature.rhythm.tempo(y=y, sr=sr, aggregate=None))
        base = float(np.median(cands)) if cands.size else 120.0
        onset = librosa.onset.onset_strength(y=y, sr=sr)

        def support(bpm):
            if bpm <= 0: return 0.0
            lag = int(round(60.0 / bpm * sr / 512))
            if lag < 2 or lag >= len(onset) // 2: return 0.0
            a = onset[:-lag] - onset[:-lag].mean()
            b = onset[lag:] - onset[lag:].mean()
            den = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
            return float(np.dot(a, b) / den)

        LO, HI = 90.0, 190.0          # the range the scenes we measure are counted in
        best, s_best = base, support(base)
        for mult in (2.0, 1.5, 3.0):
            alt = base * mult
            if not (LO <= alt <= HI):
                continue
            # it need not beat the slower pulse, only nearly match it: the slower one is
            # real, it is simply not how the record is counted.
            if support(alt) >= s_best * 0.75:
                best = alt
                break
        if best < LO and base * 2 <= HI:
            best = base * 2
        return float(best)

    def _rhythm_vector(self, y, sr) -> list[float]:
        """How the record is counted and how it moves: 16 numbers, scaled on their own.

        The 45-dim embedding describes timbre and harmony. Rhythm was represented by three
        hand-built numbers, so any contest between the two layers compared three features
        against thirty-three. Worse, taking a slice of the jointly-normalised embedding and
        re-scaling it made a change in production show up as a change in harmony: the same
        pair moved +63% or -48% depending only on how it was sliced. Two vectors, each
        normalised alone, removes that entirely.
        """
        # onsets from the low and low-mid only: a brighter hi-hat sharpens a full-spectrum
        # onset envelope and leaks a production change into the rhythm reading. Kick, snare
        # and bass carry the count; cymbals carry the sheen.
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        band = S[(freqs >= 30) & (freqs <= 1200), :]
        onset = librosa.onset.onset_strength(S=librosa.amplitude_to_db(band, ref=np.max), sr=sr)
        if onset.size < 16:
            return [0.0] * 16
        o = onset - onset.mean()
        n = float(np.linalg.norm(o)) or 1.0
        # the beat histogram: how strongly the record repeats at each of 12 lags, from a
        # sixteenth note at 200 bpm out to two bars at 90. This is the shape of the groove.
        fps = sr / 512.0
        lags = [max(2, int(round(60.0 / bpm * fps))) for bpm in
                (200, 175, 155, 140, 128, 120, 112, 100, 90, 80, 70, 60)]
        hist = []
        for lag in lags:
            if lag >= len(o) // 2:
                hist.append(0.0); continue
            a, b = o[:-lag], o[lag:]
            hist.append(float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0)))
        # how even the onsets are, how sharp, and how much of the energy falls off the grid
        gaps = np.diff(librosa.onset.onset_detect(onset_envelope=onset, sr=sr, units="frames"))
        evenness = float(1.0 / (1.0 + (np.std(gaps) / (np.mean(gaps) or 1.0)))) if gaps.size > 2 else 0.0
        attack = float(np.mean(np.abs(np.diff(o))) / (np.std(o) or 1.0))
        density = float((onset > onset.mean() + onset.std()).mean())
        pulse = float(np.max(hist)) if hist else 0.0
        v = np.array(hist + [evenness, attack, density, pulse], dtype=float)
        v = (v - np.median(v)) / (np.percentile(np.abs(v - np.median(v)), 75) or 1.0)
        nv = float(np.linalg.norm(v)) or 1.0
        return [float(x) for x in (v / nv)]

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
        # Scale each family against itself, not the whole vector. The line here computed one
        # median and one spread across all forty-five numbers and applied them to everything.
        # MFCCs run in the tens and chroma in nought to one, so after scaling all twelve
        # chroma dimensions collapsed into an identical near-zero range: an audit of 38,365
        # records found every one of them carrying the same spread, which is the signature of
        # a dimension that measures nothing. A quarter of the vector was dead weight in every
        # distance we computed.
        out = []
        for p in parts:
            p = np.asarray(p, dtype=float)
            med = np.median(p)
            scale = np.percentile(np.abs(p - med), 75) or (np.std(p) or 1.0)
            out.append((p - med) / scale)
        v = np.concatenate(out)
        # v2: the old 32-dim cut kept the MFCCs, half the chroma and none of the spectral
        # contrast — the descriptor most tied to how produced a record sounds. Keep all 45.
        if v.size >= EMBED_DIM:
            v = v[:EMBED_DIM]
        else:
            v = np.pad(v, (0, EMBED_DIM - v.size))
        n = float(np.linalg.norm(v)) or 1.0
        return [float(x) for x in v / n]
