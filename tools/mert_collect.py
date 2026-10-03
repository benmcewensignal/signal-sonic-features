"""Gathers the records' MERT fingerprints from the mert-records release into one file for the worker's volume:
ids (sorted) and F, records x four parts (drums, bass, other, vocals) x 768, float16.
  python tools/mert_collect.py <dir of mert-*.jsonl.gz> mert-records.npz"""
import base64, glob, gzip, json, os, sys, numpy as np
src, dst = sys.argv[1], sys.argv[2]; parts = ("drums", "bass", "other", "vocals"); rows = {}
for f in sorted(glob.glob(os.path.join(src, "mert-*.jsonl.gz"))):
    for line in gzip.open(f, "rt"):
        r = json.loads(line)
        if all(r.get(p) for p in parts): rows[r["track_id"]] = [np.frombuffer(base64.b64decode(r[p]), np.float16) for p in parts]
ids = sorted(rows); F = np.zeros((len(ids), 4, 768), np.float16)
for k, t in enumerate(ids): F[k] = np.stack(rows[t])
np.savez(dst, ids=np.array(ids), F=F); print(f"records: {len(ids)} | {os.path.getsize(dst) / 1e6:.0f} MB")
