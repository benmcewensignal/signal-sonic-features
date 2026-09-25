"""Calibrate both scene models on the artist-held-out split their figures come from.

Temperature scaling: one number that makes the stated confidence match how often calls are right.
Conformal threshold: the cumulative calibrated probability at which the top scenes contain the true
one 90% of the time on held-out records, so each call can return the smallest set that does.
Both are stored beside each model; the models themselves are unchanged.
  python tools/calibrate_models.py sonic.db
"""
import sys, json, glob, pickle, sqlite3, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.optimize import minimize_scalar
sys.path.insert(0, "."); from worker.parts_features import parts_vector
c = sqlite3.connect(sys.argv[1]); sc = {}
for t, s in c.execute("select track_id, scene from track_scenes where week like '____-M__' order by week"): sc.setdefault(t, s)
H = set(json.load(open("hidden.json"))) if __import__("os").path.exists("hidden.json") else set()
art = {}
for t, a in c.execute("select track_id, artists from track_meta"):
    try: v = (json.loads(a) if a and a.startswith("[") else [a])[0]
    except Exception: v = None
    art[t] = v if isinstance(v, str) and v else ""
KEYS = ("loudness", "bass_weight", "drum_density", "vocal_presence", "drum_swing"); mix = {}
for t, f in c.execute("select track_id, features from tracks where analyser_id='local' and analyser_ver like '3.0%'"):
    if t not in sc or sc[t] in H: continue
    j = json.loads(f); e = j.get("embedding"); tp = j.get("tempo"); ec = j.get("energy_curve"); rv = j.get("rhythm_vector"); sca = [j.get(k) for k in KEYS]
    if e and len(e) == 45 and tp and ec and len(ec) == 8 and isinstance(rv, list) and len(rv) == 16 and all(isinstance(x, (int, float)) for x in sca):
        mix[t] = e + [float(np.log2(tp * 2 if tp < 100 else tp))] + ec + sca + rv
kick = {json.loads(l)["track_id"]: json.loads(l).get("kick_pattern") for l in open("kick/kick-v3.jsonl")}
sw = {json.loads(l)["track_id"]: json.loads(l).get("swing16") for l in open("kick/swing-v2.jsonl")}
parts = {}
for f in glob.glob("out/stems-*.jsonl"):
    for ln in open(f):
        try: r = json.loads(ln)
        except Exception: continue
        t = r.get("track_id")
        if t in mix:
            v = parts_vector(r.get("stems") or {}, kick.get(t), sw.get(t))
            if v: parts[t] = v
P = dict(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, early_stopping=False, random_state=0)
def calibrate(name, path, ids, X):
    M = pickle.load(open(path, "rb")); S = np.array([sc[t] for t in ids]); A = np.array([art.get(t, "") for t in ids])
    rng = np.random.default_rng(0); arts = np.array(sorted(set(A[A != ""]))); rng.shuffle(arts); hold = set(arts[:len(arts) // 5])
    te = np.array([i for i in range(len(ids)) if A[i] in hold]); tr = np.array([i for i in range(len(ids)) if A[i] not in hold])
    Z = ((X - np.array(M["mu"])) / np.array(M["sd"])).astype(np.float32)
    m = HistGradientBoostingClassifier(**P).fit(Z[tr], S[tr]); Pp = np.clip(m.predict_proba(Z[te]), 1e-9, 1); cls = list(m.classes_)
    y = np.array([cls.index(s) for s in S[te]]); half = np.arange(len(te)) % 2 == 0   # fit on one half, check on the other
    def nll(T, idx):
        L = np.log(Pp[idx]) / T; L -= L.max(1, keepdims=True); Q = np.exp(L); Q /= Q.sum(1, keepdims=True); return -np.mean(np.log(Q[np.arange(idx.sum()), y[idx]]))
    T = float(minimize_scalar(lambda T: nll(T, half), bounds=(0.3, 5), method="bounded").x)
    L = np.log(Pp) / T; L -= L.max(1, keepdims=True); Q = np.exp(L); Q /= Q.sum(1, keepdims=True)
    # conformal: cumulative probability needed to include the true scene; the 90% quantile on the fitting half
    order = np.argsort(-Q, 1); cum = np.cumsum(np.take_along_axis(Q, order, 1), 1); pos = np.argmax(order == y[:, None], 1); need = cum[np.arange(len(y)), pos]
    n1 = half.sum(); q = float(np.quantile(need[half], min(1, np.ceil((n1 + 1) * 0.9) / n1)))
    setsize = (cum < q).sum(1) + 1; cover = float(np.mean(pos[~half] < setsize[~half]))
    conf = Q.max(1); pred = Q.argmax(1); bins = []
    for lo, hi in ((0.8, 1.01), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (0, 0.2)):
        k = (~half) & (conf >= lo) & (conf < hi)
        if k.sum(): bins.append([lo, hi, round(float(conf[k].mean()), 3), round(float(np.mean(pred[k] == y[k])), 3), int(k.sum())])
    M["temperature"] = round(T, 4); M["conformal_q90"] = round(q, 4); M["calibration"] = {"coverage_checked": round(cover, 3), "mean_set_size": round(float(setsize[~half].mean()), 2),
        "single_scene_share": round(float(np.mean(setsize[~half] == 1)), 3), "bins_stated_vs_right": bins, "held_out_records": int((~half).sum())}
    pickle.dump(M, open(path, "wb"))
    msg = f"{name}: temperature {T:.2f}; 90% sets cover {cover*100:.0f}% on unseen records, mean size {setsize[~half].mean():.2f}, a single scene {np.mean(setsize[~half]==1)*100:.0f}% of the time; stated vs right: " + ", ".join(f"{b[2]*100:.0f}%->{b[3]*100:.0f}%" for b in bins)
    print(msg); print(f"::notice title=calibration::{msg}")
ids = sorted(mix); calibrate("mix model", "worker/scene_model.pkl", ids, np.array([mix[t] for t in ids], np.float32))
ids2 = sorted(t for t in mix if t in parts); calibrate("mix and parts model", "worker/scene_model_parts.pkl", ids2, np.hstack([np.array([mix[t] for t in ids2], np.float32), np.array([parts[t] for t in ids2], np.float32)]))
