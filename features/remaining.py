"""How many records still need converting, or separating. Prints one number."""
import glob, json, os, sqlite3, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def stems_left(db="sonic.db", out_dir="out"):
    """Records the stem separator has not covered yet.

    Eighteen shards finishing used to leave the runner idle until someone noticed. The
    collect job asks this before booking the next eighteen, so the count has to be right:
    if it cannot answer it says so rather than returning zero, because zero means stop.
    """
    done = set()
    for f in glob.glob(os.path.join(out_dir, "stems-*.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                done.add(json.loads(line)["track_id"])
            except Exception:
                pass
    c = sqlite3.connect(db)
    have = {r[0] for r in c.execute("select track_id from tracks where analyser_id='local'")}
    return len(have - done)


if "--stems" in sys.argv:
    try:
        print(stems_left())
    except Exception as e:
        print(f"unknown: {type(e).__name__}", file=sys.stderr)
        print(-1)                      # -1 means "could not tell", which is not "stop"
    sys.exit(0)

from features.convert import already_done, to_do
try:
    from features.analyser_local import LocalAnalyser
    want = LocalAnalyser().version.split("+")[0]
except Exception:
    want = "2"
try:
    _, remaining = to_do("sonic.db", already_done("out"), want, 1)
    print(remaining)
except Exception:
    print(0)
