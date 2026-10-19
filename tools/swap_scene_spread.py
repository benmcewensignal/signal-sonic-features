"""How often the nearest licensed loop lands inside a scene's spread: for every scene the Listen map draws and every
part, the loops nearest the scene's average (the target as Swap points at it, unsteered), and for each the share of
the scene's own separated records that sit further from that average than the loop does (closer_than, the number
the page shows). A loop with closer_than 0 is further from the scene's average than every record in the scene: the
nearest the open library gets, not that sound. Runs against the deployed worker as the page does.
  PARTS_KEY=... python tools/swap_scene_spread.py out/   -> out/scene-spread.json and a summary on stdout"""
import os, sys, json, time, hmac, hashlib, secrets, statistics
import requests

BASE = os.environ.get("SWAP_URL", "https://benmcewensignal--sonic-parts-swap.modal.run").rstrip("/")
KEY = os.environ.get("PARTS_KEY", "")
OUT = sys.argv[1] if len(sys.argv) > 1 else "swap-out"; os.makedirs(OUT, exist_ok=True)
PARTS = ("drums", "bass", "other", "vocals")
TEMPO = {"140-deep-dubstep-grime": 140, "afro-house": 122, "amapiano": 113, "bass-house": 126, "breaks-breakbeat-uk-bass": 130, "deep-house": 122,
         "drum-and-bass": 174, "hard-techno": 150, "house": 125, "indie-dance": 118, "melodic-house-techno": 124, "minimal-deep-tech": 126,
         "organic-house": 120, "progressive-house": 126, "tech-house": 127, "techno-peak-time": 134, "techno-raw-deep": 132, "trance": 138,
         "uk-garage-bassline": 134, "dubstep": 140, "electro": 130, "jackin-house": 125, "nu-disco": 118, "psy-trance": 142}


def tok(scope="e"):
    exp = int(time.time()) + 3600; nonce = secrets.token_hex(6)
    return f"{exp}.{nonce}.{scope}." + hmac.new(KEY.encode(), f"swap|{exp}|{nonce}|{scope}".encode(), hashlib.sha256).hexdigest()[:32]


def note(title, msg): print(f"::notice title={title}::{msg}", flush=True)


def main():
    scenes = requests.get(f"{BASE}/scenes", params={"t": tok()}, timeout=300).json().get("scenes", [])
    rows = []; t_all = time.time()
    for slug, name, n in scenes:
        if n < 3: rows.append({"scene": slug, "name": name, "records": n, "skipped": "fewer than three separated records"}); continue
        tempo = TEMPO.get(slug, 126)
        for p in PARTS:
            t0 = time.time(); r = None
            for attempt in range(2):
                try:
                    r = requests.post(f"{BASE}/loop", params={"t": tok()}, json={"part": p, "target": {"kind": "scene", "name": slug}, "tempo": tempo, "n": 5}, timeout=900)
                    break
                except requests.RequestException as e:
                    err = str(e)[:120]; time.sleep(3)
            row = {"scene": slug, "name": name, "records": n, "part": p, "tempo": tempo, "seconds": round(time.time() - t0, 1)}
            if r is None: row["error"] = err; rows.append(row); print(json.dumps(row), flush=True); continue
            j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"error": f"http {r.status_code}"}
            if "error" in j: row["error"] = j["error"]; rows.append(row); print(json.dumps(row), flush=True); continue
            ents = [j] + (j.get("others") or [])
            row.update({"separated": (j.get("artist") or {}).get("records"), "own_median": (j.get("artist") or {}).get("own_median"),
                        "loops": [{"id": e["loop"]["id"], "closeness_to_scene": e.get("closeness_to_artist"), "closer_than": e.get("closer_than")} for e in ents],
                        "skipped": j.get("skipped")})
            ct = [e.get("closer_than") for e in ents if e.get("closer_than") is not None]
            row["first_closer_than"] = ct[0] if ct else None; row["best_closer_than"] = max(ct) if ct else None
            rows.append(row); print(json.dumps({k: row[k] for k in row if k != "loops"}), flush=True)
    json.dump({"base": BASE, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t_all), "rows": rows}, open(os.path.join(OUT, "scene-spread.json"), "w"), indent=1)
    # the summary: per part, how many scenes' nearest loop sits inside the spread at all, and the typical closer_than
    for p in PARTS:
        R = [r for r in rows if r.get("part") == p and r.get("first_closer_than") is not None]
        if not R: note(f"scene spread: {p}", "no rows"); continue
        inside = sum(1 for r in R if r["first_closer_than"] > 0); inside5 = sum(1 for r in R if (r["best_closer_than"] or 0) > 0)
        med = statistics.median(r["first_closer_than"] for r in R); med5 = statistics.median(r["best_closer_than"] for r in R)
        top = sorted(R, key=lambda r: -r["first_closer_than"])[:3]; bottom = sorted(R, key=lambda r: r["first_closer_than"])[:3]
        note(f"scene spread: {p}", f"{inside} of {len(R)} scenes: the nearest loop sits closer to the scene's average than at least one of its records "
             f"(median closer than {med:.0%} of them); with the best of five, {inside5} of {len(R)} (median {med5:.0%}). "
             f"Highest: " + ", ".join(f"{r['name']} {r['first_closer_than']:.0%}" for r in top) + ". Lowest: " + ", ".join(f"{r['name']} {r['first_closer_than']:.0%}" for r in bottom))
    errs = [r for r in rows if r.get("error")]
    if errs: note("scene spread: errors", "; ".join(f"{r['scene']} {r.get('part')}: {r['error']}" for r in errs)[:900])


if __name__ == "__main__":
    main()
