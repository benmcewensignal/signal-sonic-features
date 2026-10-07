"""Gathers the records' MERT fingerprints from the mert-records release into one file for the worker's volume:
ids (sorted) and F, records x four parts (drums, bass, other, vocals) x 768, float16.
  python tools/mert_collect.py <dir of mert-*.jsonl.gz> mert-records.npz"""
import base64, glob, gzip, json, os, sys, zlib, numpy as np
src, dst = sys.argv[1], sys.argv[2]; parts = ("drums", "bass", "other", "vocals"); rows = {}; cut = 0
def read_part(f):
    """The whole records of one file, and whether it was cut off: a cancelled wave can file a part mid-write (no
    end-of-stream marker). The lines before the cut are whole, and later waves redid those records."""
    out = []
    try:
        with gzip.open(f, "rt") as g:
            for line in g: out.append(json.loads(line))
        return out, False
    except (EOFError, OSError, zlib.error, ValueError): return out, True
files = sorted(glob.glob(os.path.join(src, "mert-*.jsonl.gz")))
for f in files:
    recs, c = read_part(f); cut += c
    for r in recs:
        if all(r.get(p) for p in parts): rows[r["track_id"]] = [np.frombuffer(base64.b64decode(r[p]), np.float16) for p in parts]
print(f"files: {len(files)}, cut off mid-write: {cut} (their whole records kept)")
ids = sorted(rows); F = np.zeros((len(ids), 4, 768), np.float16)
for k, t in enumerate(ids): F[k] = np.stack(rows[t])
np.savez(dst, ids=np.array(ids), F=F); print(f"records: {len(ids)} | {os.path.getsize(dst) / 1e6:.0f} MB")
