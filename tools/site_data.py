"""Rebuild the site's derived data, in dependency order, from the four repositories and the pipeline database.
Run daily by signalgood's site-data workflow, so nothing built from the pipelines goes stale.

  python tools/site_data.py --site ../signalgood --features . --sonic ../signal-sonic --audio ../signal-sonic-audio \
      --db sonic.db --parts record-parts.npz

  1 dj-index.json          the DJ tool's index (tools/dj_index.py), with Beatport's keys where the keys job has them
  2 from it                record-parts-lite.json, dj-names.json, scene-examples.json, ear-centres.json
  3 scene-patterns.json    kick patterns, steadiness, swing and breakdowns per scene, from the separated parts
  4 catalogue-gaps.json    the gap report (tools/catalogue_gaps.py)
  5 demand.json            the demand panel (tools/demand_panel.py), with DJ charts
  6 site-data.json         when each was built and from what
"""
import argparse, base64, collections, datetime, glob, json, os, re, sqlite3, statistics as st, subprocess, sys, unicodedata
import numpy as np

def run(args, env=None):
    r = subprocess.run([sys.executable] + args, capture_output=True, text=True, env={**os.environ, **(env or {})})
    print((r.stdout or "").strip()[-400:], flush=True)
    if r.returncode: print("FAILED:", " ".join(args[:2]), (r.stderr or "")[-600:], flush=True)
    return r.returncode == 0

def dec(s, T): return np.frombuffer(base64.b64decode(s), dtype=T)

def main():
    ap = argparse.ArgumentParser()
    for k in ("site", "features", "sonic", "audio", "db", "parts"): ap.add_argument("--" + k, required=True)
    a = ap.parse_args(); D = os.path.join(a.site, "data"); F = a.features; man = {"built": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "files": {}}
    ok = lambda name, good: man["files"].__setitem__(name, "rebuilt" if good else "kept (rebuild failed)")
    # 0 the records DJs play (every matched tracklist line), with a scene where one is known, for the DJ index
    db = sqlite3.connect(a.db); EX = {}
    try:
        scn_db = {t: s_ for t, s_, _ in db.execute("select track_id, scene, max(weight) from track_scenes group by track_id")}
        SL = json.load(open(os.path.join(F, "data", "genre-slugs.json")))["slugs"]
        for f in glob.glob(os.path.join(a.sonic, "data", "tracklists", "*.json")):
            if f.endswith("curves.json"): continue
            for sset in json.load(open(f)).get("sets", []):
                for r in sset.get("records", []):
                    b = r.get("bp")
                    if b and b not in EX: EX[b] = scn_db.get(b) or SL.get(r.get("genre") or "", "")
    except Exception as e: print("played records:", e)
    json.dump(EX, open(os.path.join(os.environ.get("RUNNER_TEMP", "/tmp"), "played.json"), "w"))
    # 1 the DJ index
    ok("dj-index.json", run([os.path.join(F, "tools", "dj_index.py"), "--extra", os.path.join(os.environ.get("RUNNER_TEMP", "/tmp"), "played.json"), "--index", os.path.join(a.audio, "out", "index.json"), "--detail", os.path.join(a.audio, "out", "detail"),
                             "--parts", a.parts, "--ear", os.path.join(F, "data", "walk_ear.json"), "--keys", os.path.join(a.sonic, "data", "track-keys.json"), "--out", os.path.join(D, "dj-index.json")]))
    X = json.load(open(os.path.join(D, "dj-index.json"))); n = X["n"]; ids = X["ids"]; at = {t: i for i, t in enumerate(ids)}
    ok("dj-sets.json", run([os.path.join(F, "tools", "dj_sets.py"), "--tracklists", os.path.join(a.sonic, "data", "tracklists"), "--djindex", os.path.join(D, "dj-index.json"),
                            "--prev", os.path.join(D, "dj-sets.json"), "--out", os.path.join(D, "dj-sets.json")]))
    # 2 derived from the DJ index
    try:
        RP = np.load(a.parts); rat = {t: i for i, t in enumerate(RP["ids"].tolist())}; V = RP["V"].astype(np.float32)
        sh = np.zeros((n, 4), np.uint8); cen = np.zeros((n, 4), np.uint16)
        for k, t in enumerate(ids):
            i = rat.get(t)
            if i is None: continue
            s_ = np.clip(V[i, :, 52], 0, None); tot = s_.sum() or 1
            sh[k] = np.round(s_ / tot * 100).astype(np.uint8); cen[k] = np.clip(np.nan_to_num(V[i, :, 48]), 0, 65535).astype(np.uint16)
        json.dump({"note": "per record, aligned with dj-index ids: each part's share of the mix's level in per cent (drums, bass, melody, voice) and its brightness (spectral centroid, Hz)", "n": n,
                   "share": base64.b64encode(sh.tobytes()).decode(), "centroid": base64.b64encode(cen.tobytes()).decode()}, open(os.path.join(D, "record-parts-lite.json"), "w"), separators=(",", ":")); ok("record-parts-lite.json", True)
    except Exception as e: print("record-parts-lite:", e); ok("record-parts-lite.json", False)
    try:
        I = json.load(open(os.path.join(a.audio, "out", "index.json"))); T = {t["track_id"]: t for t in I["tracks"]}
        def norm(s):
            s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
            s = re.sub(r"\((original|extended|radio|club|dub|instrumental|vip|edit)[^)]*\)", "", s); s = re.sub(r"\b(feat|ft|featuring)\b.*$", "", s)
            return re.sub(r"[^a-z0-9]+", " ", s).strip()
        MD = {}
        for t_, nm_, ar_ in db.execute("select track_id, name, artists from track_meta"):
            try: aa = json.loads(ar_) if ar_ and str(ar_).startswith("[") else [ar_]
            except Exception: aa = [ar_]
            MD[t_] = {"name": nm_, "artists": [x if isinstance(x, str) else (x or {}).get("name", "") for x in aa if x]}
        tl, al = [], []
        for t in ids:
            r = T.get(t) or MD.get(t) or {}; ar = r.get("artists") or []; ar = ar if isinstance(ar, list) else re.split(r",|&| x | and ", str(ar))
            tl.append(norm(r.get("name"))); al.append("|".join(x for x in (norm(y) for y in ar[:4]) if x))
        LB = {t: (l or "").strip() for t, l in db.execute("select track_id, label from track_meta")}
        json.dump({"note": "for matching a DJ's library on their own device: each record in dj-index (same order), title and artists normalised; l: the record's label as Beatport gives it", "t": tl, "a": al, "l": [LB.get(t, "") for t in ids]}, open(os.path.join(D, "dj-names.json"), "w"), separators=(",", ":")); ok("dj-names.json", True)
    except Exception as e: print("dj-names:", e); ok("dj-names.json", False)
    try:
        d_ = X["eardims"]; q = dec(X["ear"], np.int8).astype(np.float32).reshape(n, d_) * np.array(X["earscale"], np.float32) / 127
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9); has = dec(X["earhas"], np.uint8)[:n].astype(bool); sc = dec(X["scene"], np.uint8)[:n]; ex = {}
        for s_, name in enumerate(X["scenes"]):
            idx = np.where((sc == s_) & has)[0]
            if len(idx) < 20: continue
            c = q[idx].mean(0); c /= np.linalg.norm(c) or 1; ex[name] = [ids[i] for i in idx[np.argsort(-(q[idx] @ c))][:8]]
        json.dump({"note": "per scene, the records closest to the centre of its records in the learned ear", "scenes": ex}, open(os.path.join(D, "scene-examples.json"), "w"), separators=(",", ":")); ok("scene-examples.json", True)
        art = {}; lab = {}
        for t, ar, l in db.execute("select track_id, artists, label from track_meta"):
            try: L = json.loads(ar) if ar and str(ar).startswith("[") else [ar]
            except Exception: L = [ar]
            if L: art[t] = [re.sub(r"\s+", " ", str(x).lower()).strip() for x in L[:3] if x]
            if l: lab[t] = str(l).strip()
        byA, byL = collections.defaultdict(list), collections.defaultdict(list)
        for i, t in enumerate(ids):
            if not has[i]: continue
            for x in art.get(t, []): byA[x].append(i)
            if t in lab: byL[lab[t]].append(i)
        AL = json.load(open(os.path.join(a.sonic, "data", "artist-lookup.json")))["artists"]
        cent = lambda I_: (lambda v: v / (np.linalg.norm(v) or 1))(q[I_].mean(0))
        arts = [[AL[k].get("n") or k, len(byA[k]), [round(float(x), 3) for x in cent(byA[k])]] for k in AL if len(byA.get(k, [])) >= 3]
        labs = []
        for l, I_ in byL.items():
            if len(I_) < 5: continue
            cv = cent(I_); dist = 1 - q[I_] @ cv; labs.append([l, len(I_), [round(float(x), 3) for x in cv], round(float(np.percentile(dist, 50)), 4), round(float(np.percentile(dist, 75)), 4)])
        json.dump({"note": "centres in the learned ear's 16 directions: artists with 3+ records, labels with 5+ (with the median and 75th-percentile distance of their own records)", "dims": d_, "artists": arts, "labels": labs},
                  open(os.path.join(D, "ear-centres.json"), "w"), separators=(",", ":")); ok("ear-centres.json", True)
    except Exception as e: print("examples/centres:", e); ok("scene-examples.json", False); ok("ear-centres.json", False)
    ok("dj-ranker.json", run([os.path.join(F, "tools", "dj_ranker.py"), "--tracklists", os.path.join(a.sonic, "data", "tracklists"), "--djindex", os.path.join(D, "dj-index.json"),
                              "--djnames", os.path.join(D, "dj-names.json"), "--out", os.path.join(D, "dj-ranker.json")]))
    ok("dj-identity.json", run([os.path.join(F, "tools", "dj_identity.py"), "--sonic", a.sonic, "--out", os.path.join(D, "dj-identity.json")]))
    # 3 rhythm patterns per scene
    try:
        best = {}
        for f in glob.glob(os.path.join(F, "out", "stems-*.jsonl")):
            for l in open(f):
                if '"kick_pattern"' not in l: continue
                try: d = json.loads(l)
                except Exception: continue
                t = d.get("track_id"); dr = (d.get("stems") or {}).get("drums") or {}; v = (dr.get("kick_version") or 0, dr.get("swing_version") or 0)
                if t and (t not in best or v >= best[t][0]): best[t] = (v, dr)
        scene = {t: s_ for t, s_, _ in db.execute("select track_id, scene, max(weight) from track_scenes group by track_id")}
        SCN = I.get("scenes") or []
        for tr in I["tracks"]:
            s_ = tr.get("scene")
            if isinstance(s_, int) and s_ < len(SCN): scene.setdefault(tr["track_id"], SCN[s_])
        agg = collections.defaultdict(lambda: {"n": 0, "kp": collections.Counter(), "kpb": [], "steady": [], "swing": [], "brk_any": 0})
        for t, (v, dr) in best.items():
            s_ = scene.get(t)
            if not s_: continue
            A = agg[s_]; A["n"] += 1
            if dr.get("kick_pattern") and (dr.get("kick_version") or 0) >= 3: A["kp"][dr["kick_pattern"]] += 1
            for k, dst in (("kicks_per_bar", "kpb"), ("kick_steadiness", "steady"), ("swing16", "swing")):
                if isinstance(dr.get(k), (int, float)): A[dst].append(float(dr[k]))
            if isinstance(dr.get("breakdowns"), (int, float)) and dr["breakdowns"] > 0: A["brk_any"] += 1
        med = lambda x: round(st.median(x), 3) if x else None; out = {}
        for s_, A in agg.items():
            if A["n"] < 30: continue
            kp = sum(A["kp"].values())
            out[s_] = {"records": A["n"], "kick_patterns": [[p, round(k / kp * 100)] for p, k in A["kp"].most_common(3)] if kp >= 20 else [], "kicks_per_bar": med(A["kpb"]),
                       "kick_steadiness": med(A["steady"]), "swing16": med(A["swing"]), "with_breakdowns_pc": round(A["brk_any"] / A["n"] * 100)}
        json.dump({"note": "per scene, from the separated parts' rhythm detail: commonest kick patterns with their share, kicks per bar, steadiness, swing16 (0.5 straight) and the share of records with a breakdown", "scenes": out},
                  open(os.path.join(D, "scene-patterns.json"), "w"), separators=(",", ":")); ok("scene-patterns.json", True)
    except Exception as e: print("patterns:", e); ok("scene-patterns.json", False)
    # 3b ranges per scene (each part's middle half), and preview links (the classics and the listening sample), both in the shapes the page reads
    try:
        RN = {"drums": "drums", "bass": "bass", "other": "melody", "vocals": "voice"}; SR = {}
        for s_, name in enumerate(X["scenes"]):
            idx = [rat[t] for k, t in enumerate(ids) if sc[k] == s_ and t in rat]
            if len(idx) < 60: continue
            shp = np.clip(V[idx][:, :, 52], 0, None); shp = shp / np.maximum(shp.sum(1, keepdims=True), 1e-9) * 100
            o = {p: [int(round(x)) for x in np.percentile(shp[:, j], [25, 50, 75])] for j, p in enumerate(("drums", "bass", "other", "vocals"))}
            o["share"] = {RN[k]: o[k] for k in ("drums", "bass", "other", "vocals")}; o["records"] = len(idx); SR[name] = o
        json.dump({"note": "per scene, each part's share of the mix's level: 25th, 50th and 75th percentiles across its separated records", "scenes": SR}, open(os.path.join(D, "scene-ranges.json"), "w"), separators=(",", ":")); ok("scene-ranges.json", True)
        PV = {}
        cp = os.path.join(D, "canon-previews.json")
        if os.path.exists(cp): PV.update(json.load(open(cp)).get("previews") or {})
        lp = os.path.join(a.sonic, "data", "listening-previews.json")
        if os.path.exists(lp): PV.update(json.load(open(lp)))
        json.dump({"note": "Beatport preview links known to Sonic, for hearing a record in the page", "u": PV, "previews": PV}, open(os.path.join(D, "previews.json"), "w"), separators=(",", ":")); ok("previews.json", True)
    except Exception as e: print("ranges/previews:", e); ok("scene-ranges.json", False); ok("previews.json", False)
    # 4 the gap report, 5 the demand panel
    ok("catalogue-gaps.json", run([os.path.join(F, "tools", "catalogue_gaps.py")], {"SD_FEATURES": F, "SD_DB": a.db, "SD_PARTS": a.parts,
        "SD_GAPS_OUT": os.path.join(D, "catalogue-gaps.json"), "SD_FAR_OUT": os.path.join(os.environ.get("RUNNER_TEMP", "/tmp"), "catalogue_far.json")}))
    ok("demand.json", run([os.path.join(F, "tools", "demand_panel.py"), "--supply", os.path.join(a.sonic, "data", "supply.json"), "--sets", os.path.join(D, "dj-sets.json"),
        "--djindex", os.path.join(D, "dj-index.json"), "--charts", os.path.join(a.sonic, "data", "djcharts", "charts.jsonl"), "--slugs", os.path.join(F, "data", "genre-slugs.json"), "--out", os.path.join(D, "demand.json")]))
    json.dump(man, open(os.path.join(D, "site-data.json"), "w"), indent=1); print(json.dumps(man))

if __name__ == "__main__":
    main()
