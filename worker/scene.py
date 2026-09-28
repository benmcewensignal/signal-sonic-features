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


_PM = None


def part_calls(stems):
    """Each part heard alone: what scene it points to, with how often that part alone is right."""
    global _PM
    import pickle
    if _PM is None:
        fp = os.path.join(os.path.dirname(__file__), "part_models.pkl")
        if not os.path.exists(fp): return None
        with open(fp, "rb") as f: _PM = pickle.load(f)
    SC = ("level", "crest", "dynamic_span", "centroid_hz", "rolloff_hz", "flatness", "onsets_per_s", "share_of_energy")
    out = {}
    for p in ("drums", "bass", "other", "vocals"):
        s = (stems or {}).get(p) or {}; e = s.get("embedding"); M = _PM.get(p)
        if not M or not (isinstance(e, list) and len(e) == 45): continue
        v = np.array([float(x) for x in e] + [float(s.get(k)) if isinstance(s.get(k), (int, float)) else 0.0 for k in SC], float)
        z = (v - M["mu"]) / M["sd"]; pr = M["model"].predict_proba(z[None, :])[0]; order = np.argsort(-pr)
        out[p] = {"scenes": [[str(M["classes"][i]), round(float(pr[i]), 3)] for i in order[:3]], "alone_right": M["held_out_accuracy"]}
    return out or None


_AP = None


def part_sounds_like(stems, k=3):
    """For each separated part, the artists whose records' same part sounds most like it."""
    global _AP
    import pickle
    if _AP is None:
        fp = os.path.join(os.path.dirname(__file__), "artist_parts.pkl")
        if not os.path.exists(fp): return None
        with open(fp, "rb") as f: _AP = pickle.load(f)
    SC = ("level", "crest", "dynamic_span", "centroid_hz", "rolloff_hz", "flatness", "onsets_per_s", "share_of_energy")
    out = {}; total = None; used = 0
    for p in ("drums", "bass", "other", "vocals"):
        s = (stems or {}).get(p) or {}; e = s.get("embedding"); P = _AP["parts"].get(p)
        if not P or not (isinstance(e, list) and len(e) == 45): continue
        v = np.array([float(x) for x in e] + [float(s.get(c)) if isinstance(s.get(c), (int, float)) else 0.0 for c in SC], float)
        z = (v - P["mu"]) / P["sd"]; z = z / (np.linalg.norm(z) + 1e-9); sim = P["M"].astype(np.float32) @ z.astype(np.float32)
        top = np.argsort(-sim)[:k]; out[p] = [[_AP["artists"][i], round(float(sim[i]), 3)] for i in top]
        total = sim if total is None else total + sim; used += 1
    # all four parts together: in a held-out test, a record's own artist is among these five 22 times in 100 (drums alone, 16)
    if total is not None and used >= 3:
        top = np.argsort(-total)[:5]; out["all"] = [[_AP["artists"][i], round(float(total[i] / used), 3)] for i in top]
    return out or None


# Key, from the separated parts: melody, with bass and voice, matched to dance-music key templates (EDMA).
# Against Beatport's key on 3,996 records: exact 60% overall; 77% when the detection is clearest (margin >= 0.108),
# 61% in the middle, 43% when least clear (margin < 0.048).
_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_EDMA_MAJ = np.array([.1652, .0475, .0829, .0669, .0999, .0927, .0529, .1316, .0522, .0744, .0694, .0643])
_EDMA_MIN = np.array([.1724, .0400, .0761, .1253, .0567, .0822, .0626, .1435, .0810, .0571, .0836, .0551])
_CAMELOT = {"G#m": "1A", "D#m": "2A", "A#m": "3A", "Fm": "4A", "Cm": "5A", "Gm": "6A", "Dm": "7A", "Am": "8A", "Em": "9A", "Bm": "10A", "F#m": "11A", "C#m": "12A",
            "B": "1B", "F#": "2B", "C#": "3B", "G#": "4B", "D#": "5B", "A#": "6B", "F": "7B", "C": "8B", "G": "9B", "D": "10B", "A": "11B", "E": "12B"}


def key_from_parts(stems):
    def ch(p):
        e = ((stems or {}).get(p) or {}).get("embedding")
        if not (isinstance(e, list) and len(e) == 45): return None
        v = np.array(e[26:38], float); return (v - v.mean()) / (v.std() + 1e-9)
    o = ch("other")
    if o is None: return None
    v = o + sum(0.5 * x for x in (ch("bass"), ch("vocals")) if x is not None)
    sc = sorted(((float(np.corrcoef(np.roll(v, -i), prof)[0, 1]), _KEYS[i] + suf) for i in range(12) for prof, suf in ((_EDMA_MAJ, ""), (_EDMA_MIN, "m"))), reverse=True)
    k, margin = sc[0][1], sc[0][0] - sc[1][0]
    rel = 0.77 if margin >= 0.108 else (0.61 if margin >= 0.048 else 0.43)
    return {"key": k, "camelot": _CAMELOT.get(k), "margin": round(margin, 3), "matches_beatport": rel}


_PC = None


def part_residuals(stems, scene):
    """How far each separated part sits from its scene's typical part, as a percentile of how far the scene's own
    records stray: 90 means its drums are further from the scene's typical drums than 90 in 100 of its records."""
    global _PC
    import pickle
    if _PC is None:
        fp = os.path.join(os.path.dirname(__file__), "part_centroids.pkl")
        if not os.path.exists(fp): return None
        with open(fp, "rb") as f: _PC = pickle.load(f)
    SC = ("level", "crest", "dynamic_span", "centroid_hz", "rolloff_hz", "flatness", "onsets_per_s", "share_of_energy")
    out = {}
    for p in ("drums", "bass", "other", "vocals"):
        s = (stems or {}).get(p) or {}; e = s.get("embedding"); P = _PC.get(p); S = (P or {}).get("scenes", {}).get(scene)
        if not S or not (isinstance(e, list) and len(e) == 45): continue
        v = np.array([float(x) for x in e] + [float(s.get(c)) if isinstance(s.get(c), (int, float)) else 0.0 for c in SC], float)
        z = (v - P["mu"]) / P["sd"]; z = z / (np.linalg.norm(z) + 1e-9); d = 1 - float(z @ S["c"])
        q = S["q"]; pct = float(np.interp(d, q, np.linspace(0, 100, len(q))))
        out[p] = {"pct": round(pct), "scene": scene}
    return out or None


_LI = None
_CAM = {"G#m": "1A", "D#m": "2A", "A#m": "3A", "Fm": "4A", "Cm": "5A", "Gm": "6A", "Dm": "7A", "Am": "8A", "Em": "9A", "Bm": "10A", "F#m": "11A", "C#m": "12A",
        "B": "1B", "F#": "2B", "C#": "3B", "G#": "4B", "D#": "5B", "A#": "6B", "F": "7B", "C": "8B", "G": "9B", "D": "10B", "A": "11B", "E": "12B"}


def _mixes(a, b):   # keys that mix harmonically: the same Camelot code, one step either way, or the relative key
    ca, cb = _CAM.get(a), _CAM.get(b)
    if not ca or not cb: return None
    na, la, nb, lb = int(ca[:-1]), ca[-1], int(cb[:-1]), cb[-1]
    return na == nb or (la == lb and (na - nb) % 12 in (1, 11))


def licensed_parts(stems, tempo=None, key=None, k=3):
    """For each separated part, the openly licensed loops of the same family whose sound is closest, among loops
    at a workable tempo (within 8%, or double or half) and, for bass and melody, in a key that mixes with the record's."""
    global _LI
    import pickle
    if _LI is None:
        fp = os.path.join(os.path.dirname(__file__), "loop_index.pkl")
        if not os.path.exists(fp): return None
        with open(fp, "rb") as f: _LI = pickle.load(f)
    SC = ("level", "crest", "dynamic_span", "centroid_hz", "rolloff_hz", "flatness", "onsets_per_s", "share_of_energy")
    FAM = {"drums": "drums", "bass": "bass", "other": "melody", "vocals": "vocals"}; out = {}
    for p, fam in FAM.items():
        s = (stems or {}).get(p) or {}; e = s.get("embedding"); L = _LI.get(fam)
        if not L or not (isinstance(e, list) and len(e) == 45): continue
        v = np.array([float(x) for x in e] + [float(s.get(c)) if isinstance(s.get(c), (int, float)) else 0.0 for c in SC], float)[L["keep"]]
        z = (v - L["mu"]) / L["sd"]; z = z / (np.linalg.norm(z) + 1e-9); sim = L["Z"].astype(np.float32) @ z.astype(np.float32)
        order, users = [], set()
        for i in np.argsort(-sim):
            m_ = L["meta"][i]; lt = m_.get("tempo")
            if tempo and lt and not any(abs(lt * f - tempo) / tempo <= 0.08 for f in (1, 2, 0.5)): continue
            if fam in ("bass", "melody") and key and m_.get("key") and _mixes(key, m_["key"]) is False: continue
            if m_.get("user") in users: continue
            users.add(m_.get("user")); order.append(i)
            if len(order) >= k: break
        rec_plain = {n: float(s.get(n)) for n in ("crest", "centroid_hz", "flatness", "onsets_per_s") if isinstance(s.get(n), (int, float))}
        out[p] = [dict(L["meta"][i], sim=round(float(sim[i]), 3), why=_why(rec_plain, L["meta"][i].get("plain") or {})) for i in order]
    return out or None


def _why(a, b):
    """Why a loop matched a part, in a producer's words: which plain measures are alike, and which differ most."""
    words = {"centroid_hz": ("brightness", "brighter", "darker"), "onsets_per_s": ("hit density", "busier", "sparser"),
             "crest": ("punch", "punchier", "softer"), "flatness": ("noisiness", "noisier", "more tonal")}
    same, diff = [], []
    for k, (name, up, dn) in words.items():
        if k not in a or k not in b or not a[k]: continue
        r = b[k] / a[k] if a[k] else 1.0
        if 0.8 <= r <= 1.25: same.append(name)
        else: diff.append((abs(np.log(max(r, 1e-6))), up if r > 1 else dn))
    diff.sort(reverse=True)
    parts = (["similar " + " and ".join(same[:2])] if same else []) + ([diff[0][1]] if diff else [])
    return ", ".join(parts)


_RP = None


def record_loops(tid, scene=None, path="/embed/record-parts.npz"):
    """For a record Sonic has already separated, looked up by id: the licensed loops closest to each of its parts,
    and how far each part sits from its scene."""
    global _RP
    if _RP is None:
        if not os.path.exists(path): return {"error": "record parts not available"}
        d = np.load(path, allow_pickle=False)
        _RP = {"at": {t: i for i, t in enumerate(d["ids"].tolist())}, "V": d["V"], "tempo": d["tempo"], "key": d["key"]}
    i = _RP["at"].get(tid)
    if i is None: return {"error": "not separated"}
    SC = ("level", "crest", "dynamic_span", "centroid_hz", "rolloff_hz", "flatness", "onsets_per_s", "share_of_energy")
    stems = {}
    for j, p in enumerate(("drums", "bass", "other", "vocals")):
        v = _RP["V"][i, j].astype(float)
        stems[p] = dict({"embedding": [float(x) for x in v[:45]]}, **{c: float(v[45 + k]) for k, c in enumerate(SC)})
    tempo = float(_RP["tempo"][i]) or None; key = str(_RP["key"][i]) or None
    out = {"id": tid, "tempo": tempo, "key": key, "licensed_parts": licensed_parts(stems, tempo, key)}
    if scene: out["part_residuals"] = part_residuals(stems, scene)
    return out



def compare_parts(upload, ref_id, upload_tempo=None, upload_key=None, path="/embed/record-parts.npz"):
    """An uploaded track against a record Sonic has already separated, part by part: how close each part is (the same
    closeness as the loop matching, within each part's own family), and in plain words how the upload's part differs."""
    r = record_loops(ref_id, None, path)   # loads the record parts once; tells us whether the reference is separated
    if "error" in r: return r
    import pickle
    stats_path = os.path.join(os.path.dirname(__file__), "loop_index.pkl")
    global _LI
    if _LI is None:
        with open(stats_path, "rb") as f: _LI = pickle.load(f)
    SC = ("level", "crest", "dynamic_span", "centroid_hz", "rolloff_hz", "flatness", "onsets_per_s", "share_of_energy")
    FAM = {"drums": "drums", "bass": "bass", "other": "melody", "vocals": "vocals"}
    words = {"centroid_hz": ("brighter", "darker"), "onsets_per_s": ("busier", "sparser"), "crest": ("punchier", "softer"), "flatness": ("noisier", "more tonal"),
             "share_of_energy": ("louder in the mix", "quieter in the mix")}
    i = _RP["at"][ref_id]; out = {"id": ref_id, "tempo": r.get("tempo"), "key": r.get("key"), "parts": {}}
    for j, (p, fam) in enumerate(FAM.items()):
        u = (upload or {}).get(p)
        if not (isinstance(u, list) and len(u) == 53): continue
        L = _LI.get(fam)
        if not L: continue
        keep = L["keep"]; mu, sd = L["mu"], L["sd"]
        a = (np.array(u, float)[keep] - mu) / sd; b = (_RP["V"][i, j].astype(float)[keep] - mu) / sd
        sim = float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1))
        diffs = []
        for k, (up, dn) in words.items():
            x, y = float(u[45 + SC.index(k)]), float(_RP["V"][i, j][45 + SC.index(k)])
            if x > 0 and y > 0:
                q = x / y
                if q >= 1.25: diffs.append((np.log(q), up))
                elif q <= 0.8: diffs.append((-np.log(q), dn))
        diffs.sort(reverse=True)
        out["parts"][p] = {"sim": round(sim, 3), "words": [w for _, w in diffs[:2]]}
    if out["parts"]:
        far = min(out["parts"], key=lambda p: out["parts"][p]["sim"]); out["furthest"] = far
    if upload_tempo and out["tempo"]:
        t, T = float(upload_tempo), float(out["tempo"]); f = min((1, 2, 0.5), key=lambda m: abs(T * m - t)); out["tempo_gap"] = round(t - T * f, 1)
    if upload_key and out["key"]: out["key_mixes"] = _mixes(upload_key, out["key"])
    return out
