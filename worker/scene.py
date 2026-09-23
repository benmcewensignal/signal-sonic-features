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


def call(path):
    M = _model(); x, d = features_of(path)
    z = (x - np.array(M["mu"])) / np.array(M["sd"])
    p = M["model"].predict_proba(z[None, :])[0]
    order = np.argsort(-p)
    conf = float(p[order[0]])
    tier = next((t for t in M["tiers"] if t[0] <= conf < t[1]), M["tiers"][-1])
    return {"scenes": [[str(M["classes"][i]), round(float(p[i]), 3)] for i in order[:5]], "confidence": round(conf, 3),
            "right_at_this_confidence": tier[2], "tempo": round(float(d["tempo"]), 1), "loudness": round(float(d["loudness"]), 2),
            "model": {"trained_on": M["trained_on"], "built": M["built"], "held_out_accuracy": 0.472}}
