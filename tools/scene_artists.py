"""Each scene's artists ranked by how many of their records were in its Beatport top 100, across the weekly charts the
pipeline has kept (track_scenes rows from beatport:genreNN:top100), for Swap's Scenes: when a scene is chosen as a
target, its most-charted artists that Swap can point at (three or more separated records, counted as the worker
counts them: worker/artist_ids.json against the separated parts) are offered as artist targets. Built daily by
tools/site_data.py into the site's data/scene-artists.json.
  python tools/scene_artists.py --db sonic.db --parts record-parts.npz --artists worker/artist_ids.json --out scene-artists.json"""
import argparse, collections, datetime, json, sqlite3
import numpy as np


def monday(week):
    """'2026-W36' -> the date its chart week starts (ISO: a Monday), as '2026-08-31'."""
    y, w = week.split("-W")
    return datetime.date.fromisocalendar(int(y), int(w), 1).isoformat()


def main():
    ap = argparse.ArgumentParser()
    for k in ("db", "parts", "artists", "out"): ap.add_argument("--" + k, required=True)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    sep = set(np.load(a.parts)["ids"].tolist())
    can = {}   # lower-case name -> (name as the worker writes it, separated records)
    for nm, ids in json.load(open(a.artists))["artists"].items():
        n = sum(1 for t in ids if t in sep)
        if n >= 3: can[nm.lower()] = (nm, n)
    rows = c.execute("select track_id, scene, chart_rank, week from track_scenes where source like 'beatport:genre%:top100' and week like '____-W__'").fetchall()
    meta = {}
    for t, ar in c.execute("select track_id, artists from track_meta where artists is not null"):
        try: L = json.loads(ar) if str(ar).startswith("[") else [ar]
        except Exception: L = [ar]
        meta[t] = [str(x if isinstance(x, str) else (x or {}).get("name", "")).strip() for x in (L if isinstance(L, list) else [L]) if x]
    recs = collections.defaultdict(lambda: collections.defaultdict(set)); weeks = collections.defaultdict(set)
    place = collections.defaultdict(collections.Counter); best = collections.defaultdict(dict)
    for t, s, r, w in rows:
        weeks[s].add(w)
        for nm in dict.fromkeys(meta.get(t, [])):
            k = nm.lower(); recs[s][k].add(t); place[s][k] += 1
            if r: best[s][k] = min(best[s].get(k, 999), int(r))
    out = {}
    for s in sorted(recs):
        order = sorted(recs[s], key=lambda k: (-len(recs[s][k]), -place[s][k], best[s].get(k, 999), k))
        top = [[can[k][0], len(recs[s][k]), place[s][k], best[s].get(k)] for k in order if k in can][: a.top]
        W = sorted(weeks[s])
        out[s] = {"charts": len(W), "from": monday(W[0]), "to": monday(W[-1]), "charting": len(order),
                  "records": len({t for k in recs[s] for t in recs[s][k]}), "artists": top}
    json.dump({"note": "per scene: its artists Swap can point at (three or more separated records), by how many of their records were in "
                       "the scene's Beatport top 100 across the weekly charts kept (charts, from and to: the weeks' Mondays); each row "
                       "[artist, records in the top 100, placings (a record a week), best position]; charting: how many artists charted",
               "built": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "scenes": out},
              open(a.out, "w"), separators=(",", ":"), ensure_ascii=False)
    print(f"scene artists: {len(out)} scenes, {sum(len(v['artists']) for v in out.values())} artists, "
          f"{len(can):,} artists with three or more separated records, charts {min(v['charts'] for v in out.values()) if out else 0} to {max(v['charts'] for v in out.values()) if out else 0} a scene")


if __name__ == "__main__":
    main()
