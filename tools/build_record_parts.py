"""Every separated record's four parts (the 45 sound measures and 8 plain part measures each), its tempo and key, in one
compact file the worker reads to find licensed loops for a record looked up by name.
  python tools/build_record_parts.py --out /tmp/record-parts.npz"""
import sys, json, glob, argparse, numpy as np
sys.path.insert(0, ".")
from worker.scene import key_from_parts
SC = ("level", "crest", "dynamic_span", "centroid_hz", "rolloff_hz", "flatness", "onsets_per_s", "share_of_energy")
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/tmp/record-parts.npz"); a = ap.parse_args()
rows = {}
for f in sorted(glob.glob("out/stems-*.jsonl")):   # later files re-measure earlier ones, so the last reading wins
    for ln in open(f):
        if '"embedding"' not in ln: continue
        try: r = json.loads(ln)
        except Exception: continue
        st = r.get("stems") or {}
        if all(isinstance((st.get(p) or {}).get("embedding"), list) and len(st[p]["embedding"]) == 45 for p in ("drums", "bass", "other", "vocals")):
            rows[r["track_id"]] = st
ids = sorted(rows); V = np.zeros((len(ids), 4, 53), np.float16); T = np.zeros(len(ids), np.float32); K = []
for i, t in enumerate(ids):
    st = rows[t]
    for j, p in enumerate(("drums", "bass", "other", "vocals")):
        s = st[p]; V[i, j] = [float(x) for x in s["embedding"]] + [float(s.get(c)) if isinstance(s.get(c), (int, float)) else 0.0 for c in SC]
    T[i] = float((st.get("drums") or {}).get("beats_per_minute") or 0)
    try: K.append((key_from_parts(st) or {}).get("key") or "")
    except Exception: K.append("")
np.savez_compressed(a.out, ids=np.array(ids), V=V, tempo=T, key=np.array(K))
import os; print(f"record parts: {len(ids):,} records, {os.path.getsize(a.out)/1e6:.1f} MB, keys for {sum(1 for k in K if k):,}")
