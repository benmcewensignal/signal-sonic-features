"""The scene call: the corpus analyser's own measures of a file, through the trained model.

Inputs are exactly what the model was trained on, in order: the 45 numbers, log2 tempo (doubled
below 100 BPM), the 8-step energy curve, loudness, bass weight, drum density, vocal presence,
swing, and the 16-number rhythm vector. Returns the ranked scenes with probabilities, and the
reliability measured for that confidence with whole artists held out.
"""
import os, pickle, numpy as np
_M = None


def _model():
    global _M
    if _M is None:
        with open(os.path.join(os.path.dirname(__file__), "scene_model.pkl"), "rb") as f:
            _M = pickle.load(f)
    return _M


def features_of(path):
    from worker.corpus_analyser.analyser_local import LocalAnalyser
    fv = LocalAnalyser().analyse(path)
    d = fv.__dict__ if hasattr(fv, "__dict__") else dict(fv)
    M = _model()
    tp = float(d["tempo"]); tp = tp * 2 if tp < 100 else tp
    x = list(d["embedding"]) + [float(np.log2(tp))] + list(d["energy_curve"]) + [float(d[k]) for k in M["keys"]] + list(d["rhythm_vector"])
    return np.array(x, float), d



def _calibrated(M, p):
    """The model's probabilities made honest (temperature fitted on held-out records), and the smallest
    set of scenes that contains the true one nine times in ten (conformal threshold, also held out)."""
    T = M.get("temperature") or 1.0
    L = np.log(np.clip(p, 1e-9, 1)) / T; L -= L.max(); q = np.exp(L); q /= q.sum()
    order = np.argsort(-q); cum = np.cumsum(q[order]); thr = M.get("conformal_q90")
    k = int((cum < thr).sum() + 1) if thr else 1
    return q, [str(M["classes"][i]) for i in order[:k]]


def _reliability(M, scene, conf):
    """How often a call of this scene at this confidence was right with whole artists held out;
    the overall tier where the scene had fewer than 15 such calls."""
    bins = (M.get("scene_tiers") or {}).get(scene, [])
    for lo, hi, r, n in bins:
        if lo <= conf < hi + 1e-9 and r is not None and n >= 15: return r, "scene"
    # too few calls at this confidence: the scene's accuracy over all its calls, before the overall tier
    tot = sum(n for *_, n in bins); right = sum((r or 0) * n for _, _, r, n in bins)
    if tot >= 15: return round(right / tot, 3), "scene"
    t = next((t for t in M["tiers"] if t[0] <= conf < t[1]), M["tiers"][-1])
    return t[2], "all"


def call_from_numbers(x, condition=None):
    """The scene call from the 75 numbers a device measured itself; no audio involved."""
    M = _model(); x = np.array(x, float)
    if x.shape != (len(M["mu"]),): raise ValueError("wrong number of measures")
    z = (x - np.array(M["mu"])) / np.array(M["sd"])
    p = M["model"].predict_proba(z[None, :])[0]; order = np.argsort(-p); conf = float(p[order[0]])
    rel, basis = _reliability(M, str(M["classes"][order[0]]), conf)
    CT = M.get("condition_tiers") or {}
    if condition in CT:   # the reliability measured for this condition (a phone, an excerpt), on held-out records
        for row in CT[condition]:
            if row[0] != "overall" and row[0] <= conf < row[1] + 1e-9 and row[2] is not None: rel, basis = row[2], "condition"; break
    pc, cset = _calibrated(M, p)
    return {"scenes": [[str(M["classes"][i]), round(float(pc[i]), 3)] for i in order[:5]], "confidence": round(float(pc[order[0]]), 3), "set": cset, "set_coverage": 0.9 if M.get("conformal_q90") else None,
            "right_at_this_confidence": rel, "reliability_basis": basis, "condition": condition, "analyser": M.get("analyser", "2.9"),
            "model": {"trained_on": M["trained_on"], "built": M["built"], "held_out_accuracy": 0.472}}


def call(path):
    M = _model(); x, d = features_of(path)
    z = (x - np.array(M["mu"])) / np.array(M["sd"])
    p = M["model"].predict_proba(z[None, :])[0]
    order = np.argsort(-p)
    conf = float(p[order[0]])
    rel, basis = _reliability(M, str(M["classes"][order[0]]), conf); tier = [0, 0, rel]
    return {"scenes": [[str(M["classes"][i]), round(float(p[i]), 3)] for i in order[:5]], "confidence": round(conf, 3),
            "right_at_this_confidence": tier[2], "tempo": round(float(d["tempo"]), 1), "loudness": round(float(d["loudness"]), 2),
            "model": {"trained_on": M["trained_on"], "built": M["built"], "held_out_accuracy": 0.472}}


_MP = None


def call_parts(mix_x, stems):
    """The triangulated call, from the whole mix's 75 and the measured parts; None if no parts model."""
    global _MP
    import pickle
    if _MP is None:
        fp = os.path.join(os.path.dirname(__file__), "scene_model_parts.pkl")
        if not os.path.exists(fp): return None
        with open(fp, "rb") as f: _MP = pickle.load(f)
    from worker.parts_features import parts_vector
    pv = parts_vector(stems)
    if pv is None: return None
    x = np.array(list(mix_x) + pv, float); z = (x - np.array(_MP["mu"])) / np.array(_MP["sd"])
    p = _MP["model"].predict_proba(z[None, :])[0]; order = np.argsort(-p); conf = float(p[order[0]])
    rel, basis = _reliability(_MP, str(_MP["classes"][order[0]]), conf)
    pc, cset = _calibrated(_MP, p)
    return {"scenes": [[str(_MP["classes"][i]), round(float(pc[i]), 3)] for i in order[:5]], "confidence": round(float(pc[order[0]]), 3), "set": cset, "set_coverage": 0.9 if _MP.get("conformal_q90") else None,
            "right_at_this_confidence": rel, "reliability_basis": basis, "analyser": "3.0", "inputs": "mix and four parts",
            "model": {"trained_on": _MP["trained_on"], "built": _MP["built"], "held_out_accuracy": _MP["held_out_accuracy"]}}


_AR = None


def sounds_like(mix_x, stems, k=5):
    """The artists whose average sound is nearest, from the whole mix and its four parts."""
    global _AR
    import json as _j
    if _AR is None:
        base = os.path.dirname(__file__)
        if not os.path.exists(os.path.join(base, "artists_parts.npz")): return None
        _AR = (np.load(os.path.join(base, "artists_parts.npz")), _j.load(open(os.path.join(base, "artists_parts.json"))))
    from worker.parts_features import parts_vector
    pv = parts_vector(stems)
    if pv is None: return None
    z, meta = _AR
    def nz(x, mu, sd):
        v = np.nan_to_num((np.array(x, float) - mu) / sd); return v / (np.linalg.norm(v) + 1e-9)
    q = np.hstack([nz(mix_x, z["muX"], z["sdX"]), nz(pv, z["muP"], z["sdP"])]) / np.sqrt(2)
    sc = z["c"] @ q; order = np.argsort(-sc)[:k]
    return {"artists": [[meta["artists"][i], round(float(sc[i]), 3)] for i in order], "top5": meta["top5"], "of": len(meta["artists"])}
