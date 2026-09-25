"""Retrain the device's scene model on the conditions users create, and keep it only if it is better.

Training data: every clean corpus preview, plus the robustness variants (a phone-like recording, an
unmastered version, a 64 kbps file, and the first, middle and last thirty seconds) of the records
in the robustness runs. Whole artists are held out, so no version of a test record is trained on.
Reported per condition against the clean-only baseline on the same split; the new model replaces
worker/scene_model.pkl only if no condition gets worse by more than a point, clean included.
Each call can then carry the reliability measured for its condition.
  python tools/train_augmented.py sonic.db robustness/
"""
import sys, json, glob, pickle, sqlite3, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.optimize import minimize_scalar
db, rdir = sys.argv[1], sys.argv[2]; c = sqlite3.connect(db)
H = set(json.load(open("hidden.json"))); sc = {}
for t, s in c.execute("select track_id, scene from track_scenes where week like '____-M__' order by week"): sc.setdefault(t, s)
art = {}
for t, a in c.execute("select track_id, artists from track_meta"):
    try: v = (json.loads(a) if a and a.startswith("[") else [a])[0]
    except Exception: v = None
    art[t] = v if isinstance(v, str) and v else t
KEYS = ("loudness", "bass_weight", "drum_density", "vocal_presence", "drum_swing"); clean = {}
for t, f in c.execute("select track_id, features from tracks where analyser_id='local' and analyser_ver like '3.0%'"):
    if t not in sc or sc[t] in H: continue
    j = json.loads(f); e = j.get("embedding"); tp = j.get("tempo"); ec = j.get("energy_curve"); rv = j.get("rhythm_vector"); sca = [j.get(k) for k in KEYS]
    if e and len(e) == 45 and tp and ec and len(ec) == 8 and isinstance(rv, list) and len(rv) == 16 and all(isinstance(x, (int, float)) for x in sca):
        clean[t] = e + [float(np.log2(tp * 2 if tp < 100 else tp))] + ec + sca + rv
R = {}
for f in glob.glob(f"{rdir}/vectors*.jsonl"):
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        if r.get("tag") not in H: R[r["track_id"]] = r
CONDS = ["less compressed", "phone", "64 kbps", "first 30 s", "middle 30 s", "last 30 s"]
print(f"{len(clean):,} clean records, {len(R):,} with condition variants", flush=True)
rng = np.random.default_rng(0); arts = sorted({art.get(t, t) for t in clean}); rng.shuffle(arts); hold = set(arts[:len(arts) // 5])
tr_ids = [t for t in clean if art.get(t, t) not in hold]; te_ids = [t for t in clean if art.get(t, t) in hold]
rv_tr = [t for t in R if art.get(t, t) not in hold]; rv_te = [t for t in R if art.get(t, t) in hold]
def arr(v): return np.array(v, np.float32)
Xc = arr([clean[t] for t in tr_ids]); yc = np.array([sc[t] for t in tr_ids])
Xa = arr([R[t]["vectors"][k] for t in rv_tr for k in CONDS]); ya = np.array([R[t]["tag"] for t in rv_tr for k in CONDS])
mu, sd = Xc.mean(0), Xc.std(0) + 1e-9
P = dict(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, early_stopping=False, random_state=0)
def fit(X, y): return HistGradientBoostingClassifier(**P).fit((X - mu) / sd, y)
def score(m):
    out = {"clean (held-out artists)": float(np.mean(m.predict((arr([clean[t] for t in te_ids]) - mu) / sd) == np.array([sc[t] for t in te_ids])))}
    tags = np.array([R[t]["tag"] for t in rv_te])
    for k in ["clean"] + CONDS: out[k] = float(np.mean(m.predict((arr([R[t]["vectors"][k] for t in rv_te]) - mu) / sd) == tags))
    return out
base = score(fit(Xc, yc)); aug_m = fit(np.vstack([Xc, Xa]), np.concatenate([yc, ya])); aug = score(aug_m)
print("condition                  baseline  retrained")
for k in base: print(f"  {k:24} {base[k]*100:6.0f}%  {aug[k]*100:6.0f}%", flush=True)
ok = all(aug[k] >= base[k] - 0.01 for k in base)
# reliability per condition, by confidence, from the held-out records
Q = {}
for k in CONDS + ["clean"]:
    X = (arr([R[t]["vectors"][k] for t in rv_te]) - mu) / sd; p = aug_m.predict_proba(X); pred = aug_m.classes_[p.argmax(1)]; conf = p.max(1); tags = np.array([R[t]["tag"] for t in rv_te])
    Q[k] = [[lo, hi, round(float(np.mean(pred[(conf >= lo) & (conf < hi)] == tags[(conf >= lo) & (conf < hi)])), 3) if ((conf >= lo) & (conf < hi)).sum() >= 20 else None, int(((conf >= lo) & (conf < hi)).sum())] for lo, hi in ((0.6, 1.01), (0.4, 0.6), (0.0, 0.4))]
    Q[k].append(["overall", round(float(np.mean(pred == tags)), 3), len(tags)])
msg = ("accepted: " if ok else "not accepted: ") + "; ".join(f"{k} {base[k]*100:.0f}% -> {aug[k]*100:.0f}%" for k in base)
print(msg); print(f"::notice title=augmented model::{msg}")
json.dump({"baseline": base, "retrained": aug, "accepted": ok, "condition_tiers": Q, "held_out_variant_records": len(rv_te)}, open("worker/scene_model_augmented.json", "w"), indent=1)
if ok:
    final = HistGradientBoostingClassifier(**P).fit((np.vstack([arr(list(clean.values())), arr([R[t]["vectors"][k] for t in R for k in CONDS])]) - mu) / sd,
                                                  np.concatenate([np.array([sc[t] for t in clean]), np.array([R[t]["tag"] for t in R for k in CONDS])]))
    old = pickle.load(open("worker/scene_model.pkl", "rb"))
    new = dict(old); new.update({"model": final, "mu": mu.tolist(), "sd": sd.tolist(), "classes": list(final.classes_), "condition_tiers": Q, "augmented": True, "built": "2026-09-25",
                                "trained_on": int(len(clean) + len(R) * len(CONDS)), "held_out_accuracy": round(base["clean (held-out artists)"] if not ok else aug["clean (held-out artists)"], 3)})
    new.pop("temperature", None); new.pop("conformal_q90", None)   # refitted by the calibrate job on the new model
    pickle.dump(new, open("worker/scene_model.pkl", "wb")); print("worker/scene_model.pkl replaced")
