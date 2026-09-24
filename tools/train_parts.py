"""Train the triangulated scene model (the whole mix and its four parts) and calibrate it with whole
artists held out. Writes worker/scene_model_parts.pkl and its metrics beside it."""
import sqlite3, json, glob, pickle, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, "."); from worker.parts_features import parts_vector
db = sys.argv[1]; c = sqlite3.connect(db); sc = {}
for t, s in c.execute("select track_id, scene from track_scenes where week like '____-M__' order by week"): sc.setdefault(t, s)
art = {}
for t, a in c.execute("select track_id, artists from track_meta"):
    try: v = (json.loads(a) if a and a.startswith("[") else [a])[0]
    except Exception: v = None
    art[t] = v if isinstance(v, str) and v else ""
KEYS = ("loudness", "bass_weight", "drum_density", "vocal_presence", "drum_swing"); mix = {}
for t, f in c.execute("select track_id, features from tracks where analyser_id='local' and analyser_ver like '3.0%'"):
    if t not in sc: continue
    j = json.loads(f); e = j.get("embedding"); tp = j.get("tempo"); ec = j.get("energy_curve"); rv = j.get("rhythm_vector"); sca = [j.get(k) for k in KEYS]
    if e and len(e) == 45 and tp and ec and len(ec) == 8 and isinstance(rv, list) and len(rv) == 16 and all(isinstance(x, (int, float)) for x in sca):
        mix[t] = e + [float(np.log2(tp * 2 if tp < 100 else tp))] + ec + sca + rv
kick = {json.loads(l)["track_id"]: json.loads(l).get("kick_pattern") for l in open("kick/kick-v3.jsonl")}
sw = {json.loads(l)["track_id"]: json.loads(l).get("swing16") for l in open("kick/swing-v2.jsonl")}
stem = {}
for f in glob.glob("out/stems-*.jsonl"):
    for ln in open(f):
        try: r = json.loads(ln)
        except Exception: continue
        t = r.get("track_id")
        if t in mix:
            v = parts_vector(r.get("stems") or {}, kick.get(t), sw.get(t))
            if v: stem[t] = v
ids = [t for t in mix if t in stem]; S = np.array([sc[t] for t in ids]); A = np.array([art.get(t, "") for t in ids])
X = np.hstack([np.array([mix[t] for t in ids], np.float32), np.array([stem[t] for t in ids], np.float32)]); del mix, stem
rng = np.random.default_rng(0); arts = np.array(sorted(set(A[A != ""]))); rng.shuffle(arts); hold = set(arts[:len(arts) // 5])
te = np.array([i for i in range(len(ids)) if A[i] in hold]); tr = np.array([i for i in range(len(ids)) if A[i] not in hold])
mu = np.nanmean(X, 0); sd = np.nanstd(X, 0) + 1e-9; Z = ((X - mu) / sd).astype(np.float32)
P = dict(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, early_stopping=False, random_state=0)
m = HistGradientBoostingClassifier(**P).fit(Z[tr], S[tr]); Pp = m.predict_proba(Z[te]); pred = m.classes_[Pp.argmax(1)]; conf = Pp.max(1)
first = float(np.mean(pred == S[te])); top3 = float(np.mean([S[te][k] in m.classes_[np.argsort(-Pp[k])[:3]] for k in range(len(te))]))
tiers = []
for lo, hi in ((0.6, 1.01), (0.4, 0.6), (0.25, 0.4), (0, 0.25)):
    k = (conf >= lo) & (conf < hi); tiers.append([lo, min(hi, 1.0), round(float(np.mean(pred[k] == S[te][k])), 3) if k.sum() else None, round(float(k.mean()), 3)])
ST = {}
for s in m.classes_:
    ST[str(s)] = []
    for lo, hi in ((0.6, 1.01), (0.0, 0.6)):
        k = (pred == s) & (conf >= lo) & (conf < hi); ST[str(s)].append([lo, min(hi, 1.0), round(float(np.mean(S[te][k] == s)), 3) if k.sum() else None, int(k.sum())])
per = {str(s): round(float(np.mean(pred[S[te] == s] == s)), 3) for s in sorted(set(S))}
final = HistGradientBoostingClassifier(**P).fit(Z, S)
pickle.dump({"model": final, "mu": mu.tolist(), "sd": sd.tolist(), "classes": list(final.classes_), "tiers": tiers, "scene_tiers": ST, "trained_on": len(ids),
             "analyser": "3.0", "kind": "parts", "held_out_accuracy": round(first, 3), "held_out_top3": round(top3, 3), "built": "2026-09-24",
             "note": "inputs: the whole mix's 75, then worker/parts_features.parts_vector (219); standardised by mu/sd with NaN kept for the model"},
            open("worker/scene_model_parts.pkl", "wb"))
metrics = {"records": len(ids), "artists_held_out": len(hold), "held_out_records": len(te), "first": first, "top3": top3, "tiers": tiers, "per_scene": per}
json.dump(metrics, open("worker/scene_model_parts.json", "w"), indent=1)
msg = f"triangulated model: {len(ids)} records; with {len(hold)} artists held out, right first time {first*100:.1f}%, top three {top3*100:.1f}%; when 0.6+ sure ({tiers[0][3]*100:.0f}% of records) right {tiers[0][2]*100:.0f}%"
print(msg); print(f"::notice title=triangulated model::{msg}")
