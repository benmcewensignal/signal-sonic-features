"""The triangulated model's inputs, in one place, so training and the worker build them identically.

The whole mix's 75 (as worker/scene.py), then for each part (drums, bass, other, vocals) its 45
numbers and six measures, then the drums' breakdown share, hits per beat and swing version 2,
then the kick pattern as six flags.
"""
import math
PARTS = ("drums", "bass", "other", "vocals")
PART_MEASURES = ("level", "centroid_hz", "onsets_per_s", "flatness", "dynamic_span", "share_of_energy")
KICKS = ("K...K...K...K...", "K.K.K.K.K.K.K.K.", "................", "K.......K.......", "K.........K.....", "K......K..K.....")
NAN = float("nan")


def num(x):
    return float(x) if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)) else NAN


def parts_vector(stems, kick_fallback=None, swing_fallback=None):
    """219 numbers from a record's measured parts, or None if a part lacks its 45 numbers."""
    v = []
    for p in PARTS:
        s = stems.get(p) or {}
        e = s.get("embedding")
        if not (isinstance(e, list) and len(e) == 45): return None
        v += [float(x) for x in e] + [num(s.get(m)) for m in PART_MEASURES]
    d = stems.get("drums") or {}
    sw = d.get("swing16") if isinstance(d.get("swing16"), (int, float)) else swing_fallback
    v += [num(d.get("breakdown_share")), num(d.get("hits_per_beat")), num(sw)]
    kp = d.get("kick_pattern") or kick_fallback or ""
    return v + [1.0 if kp == k else 0.0 for k in KICKS]
