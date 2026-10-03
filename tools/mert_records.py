"""MERT fingerprints of records' parts, on GitHub's runners: one shard of the matching index per job.

The method tested in worker/gap_tests.py (window check): the preview is cut to the four ten-second windows MERT listens to
(centred at a fifth, two, three and four fifths of the record, two seconds of padding each side), separated with the
records' own separator (features/stems.py: separate), and each part fingerprinted with MERT (m-a-p/MERT-v1-95M: hidden
states averaged over layers and time, then over the four windows). Window-only and full separation agreed at 0.98 to 0.99
on 150 records. Records already in the mert-records release are skipped, so a wave can stop and the next carries on.
  python tools/mert_records.py <shard> <shards> <seconds budget>"""
import base64, gzip, json, os, subprocess, sys, tempfile, time, urllib.request
import numpy as np, soundfile as sf, librosa, torch
sys.path.insert(0, os.getcwd())
from features import stems as S

SR = 44100; shard, shards, budget = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]); t_start = time.time()
torch.set_num_threads(os.cpu_count() or 4)
UA = {"User-Agent": "sonic-mert"}
def get(url): return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180).read())
previews = get("https://www.earlysignal.live/data/previews.json")["previews"]
index = get("https://raw.githubusercontent.com/benmcewensignal/signal-sonic-audio/main/out/index.json")
ids = sorted(t["track_id"] for t in index["tracks"] if t["track_id"] in previews)
done = set()
for line in open("done-ids.txt") if os.path.exists("done-ids.txt") else []:
    done.add(line.strip())
# the order: records in published DJ sets first (so the "played next" check can run early), then charting records, then the rest
try:
    D = get("https://www.earlysignal.live/data/dj-sets.json"); dj = set(D.get("next", {}).keys())
    for k_, v_ in D.get("next", {}).items(): dj |= {x[0] for x in v_}
    for st_ in D.get("sets", []): dj |= set(st_[2])
except Exception:
    dj = set()
charted = {t["track_id"] for t in index["tracks"] if t.get("chart_best") is not None}
ids = sorted(ids, key=lambda t: (0 if t in dj else 1 if t in charted else 2, t))
todo = [t for t in ids if t not in done][shard::shards]
print(f"matching index with previews: {len(ids)} | done before this wave: {len(done)} | this shard: {len(todo)}", flush=True)

from transformers import AutoModel, Wav2Vec2FeatureExtractor
model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).eval()
proc = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)

def mert(y24):
    with torch.no_grad():
        hs = model(**proc(y24, sampling_rate=24000, return_tensors="pt"), output_hidden_states=True).hidden_states
    return torch.stack(hs).mean(dim=(0, 2))[0].numpy()

def one(t):
    w = tempfile.mkdtemp(); src = os.path.join(w, "p.mp3"); urllib.request.urlretrieve(previews[t], src)
    wav = os.path.join(w, "p.wav"); subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-i", src, "-ac", "2", "-ar", str(SR), wav], check=True, timeout=120)
    x, _ = sf.read(wav, always_2d=True); L = len(x); win, pad = 10 * SR, 2 * SR; segs, keep = [], []
    for c in (0.2, 0.4, 0.6, 0.8):
        a = max(0, min(L - win, int(L * c) - win // 2)); lo, hi = max(0, a - pad), min(L, a + win + pad)
        keep.append((sum(len(s_) for s_ in segs) + (a - lo), win)); segs.append(x[lo:hi])
    joined = os.path.join(w, "w.wav"); sf.write(joined, np.concatenate(segs), SR)
    st = S.separate(joined, os.path.join(w, "sep")); fps = {}
    for k in ("drums", "bass", "other", "vocals"):
        y, sr = sf.read(st[k], always_2d=True); y = y.mean(1); vs = []
        for off, n in keep:
            seg = librosa.resample(y[off:off + n].astype(np.float32), orig_sr=sr, target_sr=24000)
            if len(seg) >= 5 * 24000: vs.append(mert(seg))
        fps[k] = base64.b64encode(np.mean(vs, axis=0).astype(np.float16).tobytes()).decode() if vs else None
    subprocess.run(["rm", "-rf", w])
    return fps

# filed as it goes: every 100 records the part so far is closed and uploaded, so a runner stopped part-way loses minutes,
# not hours. Names end in the run id, as the chain's count expects: mert-<shard>-p<k>-<run>.jsonl.gz and mert-ids-...
os.makedirs("mert-out", exist_ok=True); run = os.environ.get("GITHUB_RUN_ID", "local"); n_ok = n_err = 0; t0 = time.time(); part = 0
def open_part(k):
    o = f"mert-out/mert-{shard:02d}-p{k:03d}-{run}.jsonl.gz"; i = f"mert-out/mert-ids-{shard:02d}-p{k:03d}-{run}.txt"
    return o, i, gzip.open(o, "wt"), open(i, "w")
def file_part(o, i):
    if run != "local" and os.path.getsize(i) > 0:
        r = subprocess.run(["gh", "release", "upload", "mert-records", o, i, "--clobber"], capture_output=True, text=True)
        if r.returncode: print("  upload failed:", (r.stderr or "")[-160:], flush=True)
outp, idsp, out, idf = open_part(part); in_part = 0
for t in todo:
    if time.time() - t_start > budget: break
    try:
        fp = one(t); out.write(json.dumps({"track_id": t, "v": 1, **fp}) + "\n"); idf.write(t + "\n"); n_ok += 1; in_part += 1
    except Exception as ex:
        n_err += 1; print(f"  {t}: {type(ex).__name__}: {str(ex)[:120]}", flush=True)
    if in_part >= 100:
        out.close(); idf.close(); file_part(outp, idsp); part += 1; outp, idsp, out, idf = open_part(part); in_part = 0
    if (n_ok + n_err) % 25 == 0: print(f"  {n_ok} done, {n_err} failed, {(time.time() - t0) / max(1, n_ok + n_err):.1f} s a record", flush=True)
out.close(); idf.close(); file_part(outp, idsp)
print(f"shard {shard}: {n_ok} fingerprinted, {n_err} failed, {(time.time() - t0) / max(1, n_ok + n_err):.1f} s a record", flush=True)
