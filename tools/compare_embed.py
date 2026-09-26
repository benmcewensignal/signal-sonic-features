"""The hand-built model scored on the learned model's exact held-out records.
  python tools/compare_embed.py sonic.db split.json
"""
import sys, json, sqlite3, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
db, sp = sys.argv[1], json.load(open(sys.argv[2])); c = sqlite3.connect(db)
KEYS = ("loudness", "bass_weight", "drum_density", "vocal_presence", "drum_swing"); V = {}
for t, f in c.execute("select track_id, features from tracks where analyser_id='local' and analyser_ver like '3.0%'"):
    j = json.loads(f); e = j.get("embedding"); tp = j.get("tempo"); ec = j.get("energy_curve"); rv = j.get("rhythm_vector"); s5 = [j.get(k) for k in KEYS]
    if e and len(e) == 45 and tp and ec and len(ec) == 8 and isinstance(rv, list) and len(rv) == 16 and all(isinstance(x, (int, float)) for x in s5):
        V[t] = e + [float(np.log2(tp * 2 if tp < 100 else tp))] + ec + s5 + rv
art = {}
for t, a in c.execute("select track_id, artists from track_meta"):
    try: v = (json.loads(a) if a and a.startswith("[") else [a])[0]
    except Exception: v = None
    art[t] = v if isinstance(v, str) and v else t
hold = set(sp["held_artists"]); H = set(json.load(open("hidden.json"))); sc = {}
for t, s in c.execute("select track_id, scene from track_scenes where week like \'____-M__\' order by week"): sc.setdefault(t, s)
test = [(t, s) for t, s in sp["test"] if t in V]; P = dict(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, early_stopping=False, random_state=0)
def score(tr):
    X = np.array([V[t] for t, _ in tr], np.float32); y = np.array([s for _, s in tr]); mu, sd = X.mean(0), X.std(0) + 1e-9
    m = HistGradientBoostingClassifier(**P).fit((X - mu) / sd, y); pr = m.predict_proba((np.array([V[t] for t, _ in test], np.float32) - mu) / sd)
    tags = np.array([s for _, s in test]); first = float(np.mean(m.classes_[pr.argmax(1)] == tags)); top3 = float(np.mean([tags[i] in m.classes_[np.argsort(-pr[i])[:3]] for i in range(len(test))]))
    return first, top3, len(tr)
same = [(t, s) for t, s in sp["train"] if t in V]
wide = [(t, sc[t]) for t in V if t in sc and sc[t] not in H and art.get(t, t) not in hold]
r1, r2 = score(same), score(wide)
msg = f"on the learned model's {len(test):,} held-out records: hand-built on the same {r1[2]:,} records {r1[0]*100:.1f}% first, {r1[1]*100:.1f}% top three; hand-built on all {r2[2]:,} other records {r2[0]*100:.1f}% first, {r2[1]*100:.1f}% top three; learned 60.7% first, 82.0% top three"
print(msg); print(f"::notice title=same-split comparison::{msg}")
