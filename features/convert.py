"""Feature conversion worker.

The main pipeline runs one job at a time and two of its jobs requeue themselves for
days. Re-analysing the corpus to the current analyser version is the longest of them
and needs nothing from the database except a list of track ids, so it does not belong
there. This does that work on its own runner and publishes jsonl the pipeline imports.

Rules this worker follows, each of them learned from a failure:

  it never writes sonic.db          two writers to one file is how a night gets lost
  it samples across scene-months    processing in rowid order silently samples whatever
                                    was ingested last, which has now happened three times
  it archives nothing               the pipeline keeps the old vector when it imports
  it writes as it goes              a run killed at the cap still leaves what it did

  python -m features.convert --db sonic.db --limit 400 --budget-minutes 75
"""
import argparse, collections, json, os, random, sqlite3, tempfile, time, urllib.request
from .analyser_local import LocalAnalyser
from .beatport import get_token, _get


def preview_url(track_id, token):
    if not str(track_id).startswith("bp:"): return None
    d = _get(f"/catalog/tracks/{str(track_id).split(':')[-1]}/", token)
    return (d.get("sample_url") or (d.get("preview") or {}).get("mp3", {}).get("url") or "") or None


def local_copy(url):
    req = urllib.request.Request(url, headers={"User-Agent": "signal-sonic-features"})
    fd, path = tempfile.mkstemp(suffix=".mp3")
    with urllib.request.urlopen(req, timeout=25) as r, os.fdopen(fd, "wb") as f:
        f.write(r.read())
    return path


def already_done(out_dir):
    have = set()
    if not os.path.isdir(out_dir): return have
    for f in sorted(os.listdir(out_dir)):
        if f.startswith("features-") and f.endswith(".jsonl"):
            for line in open(os.path.join(out_dir, f)):
                try: have.add(json.loads(line)["track_id"])
                except Exception: pass
    return have


def to_do(db, have, want_version, limit):
    """One record per scene-month, round robin, so the sample is never a slice of one wave."""
    c = sqlite3.connect(db); c.row_factory = sqlite3.Row
    pool = collections.defaultdict(list)
    for r in c.execute("""select ts.scene, ts.week, ts.track_id, t.analyser_ver from track_scenes ts
                          join tracks t on t.track_id=ts.track_id and t.analyser_id='local'
                          where ts.week like '____-M__' and ts.track_id like 'bp:%'"""):
        if r["track_id"] in have: continue
        if str(r["analyser_ver"] or "1").startswith(want_version): continue   # already current
        pool[(r["scene"], r["week"])].append(r["track_id"])
    keys = sorted(pool); rng = random.Random(4)
    for k in keys: rng.shuffle(pool[k])
    todo = []
    while len(todo) < limit and any(pool[k] for k in keys):
        for k in keys:
            if pool[k] and len(todo) < limit: todo.append(pool[k].pop())
    remaining = sum(len(v) for v in pool.values())
    return todo, remaining


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="sonic.db"); ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--budget-minutes", type=int, default=75); ap.add_argument("--out-dir", default="out")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    an = LocalAnalyser()
    want = an.version.split("+")[0]
    have = already_done(a.out_dir)
    todo, remaining = to_do(a.db, have, want, a.limit)
    print(f"convert: target version {an.version} | {len(have)} already published | {len(todo)} this run | {remaining} left after", flush=True)
    if not todo:
        print("nothing to convert"); return
    token = get_token()
    n = len([f for f in os.listdir(a.out_dir) if f.startswith("features-")])
    path = os.path.join(a.out_dir, f"features-{n:03d}.jsonl")
    t0 = time.time(); done = err = 0
    with open(path, "w") as out:
        for tid in todo:
            if (time.time() - t0) / 60 > a.budget_minutes:
                print("budget reached", flush=True); break
            p = None
            try:
                url = preview_url(tid, token)
                if not url: raise ValueError("no preview url")
                p = local_copy(url)
                fv = an.analyse(p)
                d = fv.__dict__ if hasattr(fv, "__dict__") else dict(fv)
                out.write(json.dumps({"track_id": tid, "analyser_version": an.version, "features": d}) + "\n")
                done += 1
                if done % 50 == 0:
                    out.flush(); print(f"  {done}/{len(todo)}, {err} failed", flush=True)
            except Exception as e:
                err += 1
                if err <= 3: print(f"  {tid}: {type(e).__name__}: {str(e)[:70]}", flush=True)
            finally:
                if p:
                    try: os.unlink(p)
                    except OSError: pass
    print(f"convert: {done} written to {path}, {err} failed, {remaining + (len(todo) - done)} still to convert", flush=True)


if __name__ == "__main__":
    main()
