"""The analyser seam. Everything downstream touches audio analysis
through Analyser.analyse() and nothing else.

Two implementations:
  - CyaniteAnalyser: real, needs CYANITE_TOKEN, network. Impl A.
  - StubAnalyser: deterministic from track id, for tests and dry runs.

Every FeatureVector carries analyser_id + version. Aggregates MUST
filter on analyser_id — mixing vintages silently corrupts fingerprints
(the DfE multi-year-file lesson: always filter the vintage).
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import urllib.request
from dataclasses import dataclass, field, asdict

EMBED_DIM = 45  # 13 MFCC means + 13 MFCC deviations + 12 chroma + 7 spectral contrast.
                # The old 32 truncated spectral contrast away entirely, which is the
                # descriptor most tied to how produced a record sounds.


@dataclass
class FeatureVector:
    tempo: float
    key: str
    energy_curve: list[float]          # 8 segments, 0..1
    drum_palette: list[str]
    drum_density: float                # 0..1
    drum_swing: float                  # 0..1
    bass_character: list[str]
    bass_weight: float                 # 0..1
    vocal_treatment: list[str]
    vocal_presence: float              # 0..1
    mood: list[str]
    embedding: list[float]             # EMBED_DIM, L2-normalised
    analyser_id: str = ""
    analyser_version: str = ""
    loudness: float = 0.0              # mean RMS dBFS; mastering confound, measured not hidden

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "FeatureVector":
        return FeatureVector(**json.loads(s))


class Analyser:
    analyser_id = "base"
    version = "0"

    def analyse(self, audio_ref: str) -> FeatureVector:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Impl B-shaped stub: deterministic pseudo-analysis from the audio_ref.
# Exists so the entire pipeline, harness and derivative maths run with no
# network and no keys. Feature values are stable across runs (hash-seeded).
# ---------------------------------------------------------------------------

_DRUMS = ["breaks", "four-four", "shuffled", "log-drum", "gqom-stomp", "808", "garage-swing"]
_BASS = ["sub", "reese", "warm", "distorted", "rolling", "sparse"]
_VOX = ["chopped", "full-song", "pitched", "spoken", "none", "diva"]
_MOODS = ["dark", "euphoric", "hypnotic", "aggressive", "warm", "melancholic"]
_KEYS = ["Am", "Cm", "Dm", "Em", "Fm", "Gm", "A#m", "F#m"]


def _rng_stream(seed: str):
    """Deterministic float stream in [0,1) from a string seed."""
    counter = 0
    while True:
        h = hashlib.sha256(f"{seed}:{counter}".encode()).digest()
        for i in range(0, 32, 4):
            yield int.from_bytes(h[i:i + 4], "big") / 2**32
        counter += 1


class StubAnalyser(Analyser):
    """Deterministic features from audio_ref. Supports an offset= param in the
    ref (e.g. 'track42?offset=0.3') which shifts the feature distribution —
    the test harness uses this to simulate drift and convergence."""
    analyser_id = "stub"
    version = "1"

    def analyse(self, audio_ref: str) -> FeatureVector:
        base_ref, _, off = audio_ref.partition("?offset=")
        offset = float(off) if off else 0.0
        r = _rng_stream(base_ref)

        def u() -> float:
            return next(r)

        def shifted(x: float) -> float:
            return min(1.0, max(0.0, x * (1 - abs(offset)) + (offset if offset > 0 else 0.0)))

        # Tracks within a scene are correlated: the ref's first token is the
        # style seed contributing a strong shared component, individual
        # variation rides on top. This models reality; without it a scene
        # centroid is pure noise and nothing is detectable (correctly).
        style = base_ref.split("-")[0]
        sr = _rng_stream("style:" + style)
        style_vec = [next(sr) * 2 - 1 for _ in range(EMBED_DIM)]
        emb = [sv * 2.5 + (u() * 2 - 1) for sv in style_vec]
        # offset pushes the embedding along a fixed direction — simulated drift
        for i in range(0, EMBED_DIM, 2):
            emb[i] += offset * 3.0
        norm = math.sqrt(sum(x * x for x in emb)) or 1.0
        emb = [x / norm for x in emb]

        return FeatureVector(
            tempo=118 + u() * 60,
            key=_KEYS[int(u() * len(_KEYS))],
            energy_curve=[shifted(u()) for _ in range(8)],
            drum_palette=[_DRUMS[int(u() * len(_DRUMS))]],
            drum_density=shifted(u()),
            drum_swing=shifted(u()),
            bass_character=[_BASS[int(u() * len(_BASS))]],
            bass_weight=shifted(u()),
            vocal_treatment=[_VOX[int(u() * len(_VOX))]],
            vocal_presence=shifted(u()),
            mood=[_MOODS[int(u() * len(_MOODS))]],
            embedding=emb,
            analyser_id=self.analyser_id,
            analyser_version=self.version,
        )


# ---------------------------------------------------------------------------
# Impl A: Cyanite. Library Track flow via GraphQL. Needs CYANITE_TOKEN.
# Kept thin: auth, one query, mapping into FeatureVector. Untested against
# the live API from this environment (no network) — treat the query shape
# as a starting point to verify against current Cyanite docs, not gospel.
# ---------------------------------------------------------------------------

_CYANITE_URL = "https://api.cyanite.ai/graphql"

_ANALYSIS_QUERY = """
query Track($id: ID!) {
  libraryTrack(id: $id) {
    ... on LibraryTrack {
      audioAnalysisV6 {
        ... on AudioAnalysisV6Finished {
          result {
            bpmPrediction { value }
            keyPrediction { value }
            energyLevel
            moodTags
            genreTags
            voiceTags
            segments { representativeSegmentIndex }
          }
        }
      }
    }
  }
}
"""


class CyaniteAnalyser(Analyser):
    analyser_id = "cyanite"
    version = "v6"

    def __init__(self, token: str | None = None):
        self.token = token or os.environ.get("CYANITE_TOKEN", "")
        if not self.token:
            raise RuntimeError("CYANITE_TOKEN not set")

    def _gql(self, query: str, variables: dict) -> dict:
        req = urllib.request.Request(
            _CYANITE_URL,
            data=json.dumps({"query": query, "variables": variables}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def analyse(self, audio_ref: str) -> FeatureVector:
        """audio_ref here is a Cyanite library track id (upload happens
        upstream in the ingest step via Cyanite's file upload mutation)."""
        data = self._gql(_ANALYSIS_QUERY, {"id": audio_ref})
        res = (
            data.get("data", {})
            .get("libraryTrack", {})
            .get("audioAnalysisV6", {})
            .get("result")
        )
        if not res:
            raise RuntimeError(f"analysis not ready or failed for {audio_ref}")
        # Map Cyanite's vocabulary into ours. Embedding: Cyanite exposes
        # similarity vectors on some plans; absent that, we derive a crude
        # embedding from tag one-hots so downstream maths still works.
        tags = (res.get("moodTags") or []) + (res.get("genreTags") or []) + (res.get("voiceTags") or [])
        emb = _tags_to_embedding(tags)
        return FeatureVector(
            tempo=float(res.get("bpmPrediction", {}).get("value") or 0.0),
            key=str(res.get("keyPrediction", {}).get("value") or ""),
            energy_curve=[float(res.get("energyLevel") or 0.5)] * 8,
            drum_palette=[],
            drum_density=0.5,
            drum_swing=0.5,
            bass_character=[],
            bass_weight=0.5,
            vocal_treatment=res.get("voiceTags") or [],
            vocal_presence=1.0 if res.get("voiceTags") else 0.0,
            mood=res.get("moodTags") or [],
            embedding=emb,
            analyser_id=self.analyser_id,
            analyser_version=self.version,
        )


def _tags_to_embedding(tags: list[str]) -> list[float]:
    emb = [0.0] * EMBED_DIM
    for t in tags:
        h = int(hashlib.sha256(t.lower().encode()).hexdigest(), 16)
        emb[h % EMBED_DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in emb)) or 1.0
    return [x / norm for x in emb]


class UrlAwareAnalyser(Analyser):
    """Wraps an analyser so http(s) audio_refs are downloaded, analysed,
    and deleted. analyser_id passes through — the wrapper is not a vintage."""

    def __init__(self, inner: Analyser):
        self.inner = inner
        self.analyser_id = inner.analyser_id
        self.version = inner.version

    def analyse(self, audio_ref: str) -> FeatureVector:
        if audio_ref.startswith(("http://", "https://")):
            from .beatport import download_preview
            import os as _os
            path = download_preview(audio_ref)
            try:
                return self.inner.analyse(path)
            finally:
                try:
                    _os.unlink(path)
                except OSError:
                    pass
        return self.inner.analyse(audio_ref)


def get_analyser(name: str) -> Analyser:
    if name == "stub":
        return StubAnalyser()
    if name == "cyanite":
        return CyaniteAnalyser()
    if name == "local":
        from .analyser_local import LocalAnalyser
        return UrlAwareAnalyser(LocalAnalyser())
    raise ValueError(f"unknown analyser: {name}")
