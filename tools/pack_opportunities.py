"""Pack opportunities for What to make (site: data/pack-opportunities.json).

A sample label commissions packs: a coherent style, named by the artists and labels it sounds like. This finds them.
  reference   per scene, the records producers aspire to: records DJs chart (Beatport DJ charts), records DJs play
              (published tracklists since 2024), and records in Beatport's own weekly genre charts this year
  sub-styles  clusters of the reference records in the learned ear, the number chosen by how cleanly they separate
              (silhouette), each at least 15 records and a tenth of the scene, and kept only if the split survives
              reseeding (adjusted Rand index 0.5 or more); otherwise the scene is one style
  market      every record of the scene Sonic has placed (the long tail included: the label's customers) that falls
              inside a sub-style's range (its own reference records' 90th-percentile distance from its centre, so a tenth of
              the references themselves fall outside), times
              the scene's Beatport releases a month
  DJ share    the sub-style's share of the scene's chart picks and set plays; against its release share it says where
              DJs want more of a sound than producers deliver (a description of now, not a forecast)
  coverage    for each part that carries 8% or more of a reference record's mix, the best openly licensed loop at a workable tempo (and for bass
              and melody a key that mixes): close at 0.7 or more, far under 0.45 (as the gap report measures it)
  brief       tempo, keys, part balance, kick pattern, swing, the records nearest the centre, artists and labels, and
              for each part the records whose part no free loop comes close to (what a label's pack is checked against)
  target      for Aim a track: each style's and genre's centre in the ear, its part-balance ranges, and the DJs who
              chart it most
  map         styles laid out so alike ones sit close (classical scaling of the distances between their centres in the
              ear, then nudged apart so no two dots overlap), each linked to its two nearest styles
  python tools/pack_opportunities.py --site ../signalgood --features . --sonic ../signal-sonic --db sonic.db --parts record-parts.npz
"""
import argparse, base64, collections, glob, json, os, pickle, sqlite3, sys, numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

def main():
    ap = argparse.ArgumentParser()
    for k in ("site", "features", "sonic", "db", "parts"): ap.add_argument("--" + k, required=True)
    ap.add_argument("--out", default=None); a = ap.parse_args(); D = os.path.join(a.site, "data"); sys.path.insert(0, a.features)
    from worker.scene import _mixes
    X = json.load(open(os.path.join(D, "dj-index.json"))); n = X["n"]; ids = X["ids"]; at = {t: i for i, t in enumerate(ids)}; SCN = X["scenes"]; CAM = X["camelot"]
    d = X["eardims"]; E = np.frombuffer(base64.b64decode(X["ear"]), np.int8).astype(np.float32).reshape(n, d) * np.array(X["earscale"], np.float32) / 127
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9); has = np.frombuffer(base64.b64decode(X["earhas"]), np.uint8)[:n].astype(bool)
    sc = np.frombuffer(base64.b64decode(X["scene"]), np.uint8)[:n]; tm = np.frombuffer(base64.b64decode(X["tempo"]), np.uint16)[:n].astype(np.float32) / 10
    key = np.frombuffer(base64.b64decode(X["key"]), np.uint8)[:n]
    NM = json.load(open(os.path.join(D, "dj-names.json"))); LB = NM.get("l") or [""] * n
    db = sqlite3.connect(a.db); meta_art = {}
    for t, ar in db.execute("select track_id, artists from track_meta"):
        try: L = json.loads(ar) if ar and str(ar).startswith("[") else [ar]
        except Exception: L = [ar]
        meta_art[t] = [x if isinstance(x, str) else (x or {}).get("name", "") for x in L if x]
    # reference weights: chart picks, set plays, Beatport chart weeks
    W = collections.Counter(); src = collections.Counter(); WHO = collections.defaultdict(collections.Counter)
    cp = os.path.join(a.sonic, "data", "djcharts", "charts.jsonl")
    if os.path.exists(cp):
        for line in open(cp):
            try: r = json.loads(line)
            except Exception: continue
            for t in r.get("tracks") or []:
                i = at.get(t)
                if i is not None and has[i]:
                    W[i] += 1; src["chart picks"] += 1
                    if r.get("dj") and not str(r["dj"]).lower().startswith("beatport"): WHO[i][str(r["dj"])] += 1   # Beatport's own editorial charts are not a DJ to send to
    for f in glob.glob(os.path.join(a.sonic, "data", "tracklists", "*.json")):
        if f.endswith("curves.json"): continue
        for st in json.load(open(f)).get("sets", []):
            if (st.get("title") or "")[:4] < "2024": continue
            for r in st.get("records", []):
                i = at.get(r.get("bp"))
                if i is not None and has[i]: W[i] += 1; src["set plays"] += 1
    for t, s_, rk in db.execute("select track_id, scene, chart_rank from track_scenes where chart_rank is not null and week like '2026-%'"):
        i = at.get(t)
        if i is not None and has[i]: W[i] += 1; src["Beatport chart weeks"] += 1
    PER = {}
    dm = os.path.join(D, "demand.json")
    if os.path.exists(dm):
        for s_, v in (json.load(open(dm)).get("scenes") or {}).items():
            pm = ((v or {}).get("releases") or {}).get("per_month")
            if pm: PER[s_] = float(pm)
    # parts and the loop library
    RP = np.load(a.parts); rat = {t: i for i, t in enumerate(RP["ids"].tolist())}; V = RP["V"].astype(np.float32)
    LI = pickle.load(open(os.path.join(a.features, "worker", "loop_index.pkl"), "rb"))
    FAM = {"drums": ("drums", 0), "bass": ("bass", 1), "melody": ("melody", 2), "voice": ("vocals", 3)}
    LIB = {}
    for part, (fam, pi) in FAM.items():
        L = LI.get(fam)
        if not L: continue
        LIB[part] = (L["Z"].astype(np.float32), L["keep"], L["mu"].astype(np.float32), L["sd"].astype(np.float32),
                     np.array([m.get("tempo") or 0 for m in L["meta"]], np.float32), [m.get("key") for m in L["meta"]], pi, {})
    def best_loop(i, part):
        Z, keep, mu, sd, lt, lk, pi, KM = LIB[part]; r = rat.get(ids[i])
        if r is None: return None
        v = V[r, pi][keep]; z = (v - mu) / sd; z /= np.linalg.norm(z) + 1e-9; sim = Z @ z; T = float(tm[i]) or 0
        ok = np.ones(len(lt), bool)
        if T:
            ok = np.zeros(len(lt), bool)
            for f in (1, 2, 0.5): ok |= (lt > 0) & (np.abs(lt * f - T) / T <= 0.08)
        if part in ("bass", "melody") and key[i] < 24:
            kk = CAM[key[i]]
            if kk not in KM: KM[kk] = np.array([(_mixes(kk, x) is not False) if x else True for x in lk])
            ok &= KM[kk]
        return float(sim[ok].max()) if ok.any() else -1.0
    # rhythm detail per record (kick pattern, swing) from the separated drums
    RH = {}
    want = {ids[i] for i in W}
    for f in glob.glob(os.path.join(a.features, "out", "stems-*.jsonl")):
        for line in open(f):
            if '"kick_pattern"' not in line: continue
            try: dd = json.loads(line)
            except Exception: continue
            t = dd.get("track_id")
            if t not in want: continue
            dr = (dd.get("stems") or {}).get("drums") or {}
            if (dr.get("kick_version") or 0) >= 3: RH[t] = (dr.get("kick_pattern"), dr.get("swing16"), dr.get("kicks_per_bar"))
    out = []; report = {}; CEN = []; GEN = []; GCEN = []
    for s_i, name in enumerate(SCN):
        refs = np.array([i for i in W if sc[i] == s_i])
        if len(refs) >= 15 and name != "unknown":
            gc = E[refs].mean(0); gc /= np.linalg.norm(gc) or 1; GCEN.append(gc)
            order_g = np.argsort(1 - E[refs] @ gc)
            covg = {}; gap_g = {}
            for part in LIB:
                pi = FAM[part][1]; pres = []; bs = []; far_g = []
                for i in refs:
                    r = rat.get(ids[i])
                    if r is None: continue
                    sh_ = np.clip(V[r, :, 52], 0, None); w8 = sh_[pi] / max(float(sh_.sum()), 1e-9) >= 0.08; pres.append(w8)
                    if w8:
                        b = best_loop(i, part)
                        if b is not None:
                            bs.append(b)
                            if b < 0.45: far_g.append(ids[i])
                if far_g: gap_g[part] = far_g
                bs = np.array(bs); present = float(np.mean(pres)) if pres else 0.0
                if len(bs) >= 10: covg[part] = {"present": round(present, 3), "records": int(len(bs)), "close": round(float((bs >= 0.7).mean()), 3), "some": round(float(((bs >= 0.45) & (bs < 0.7)).mean()), 3), "far": round(float((bs < 0.45).mean()), 3)}
            rs_g = [rat.get(ids[i]) for i in refs]; rs_g = [r for r in rs_g if r is not None]
            shg = np.clip(V[rs_g][:, :, 52], 0, None); shg = shg / np.maximum(shg.sum(1, keepdims=True), 1e-9) * 100
            ksg = collections.Counter(CAM[key[i]] for i in refs if key[i] < 24); minor_g = np.mean([key[i] < 12 for i in refs if key[i] < 24]) if ksg else None
            rhg = [RH[ids[i]] for i in refs if ids[i] in RH]; kpg = collections.Counter(x[0] for x in rhg if x[0]); swg = [x[1] for x in rhg if isinstance(x[1], (int, float))]
            tqg = [float(np.percentile(tm[refs], q)) for q in (25, 50, 75)]
            if tqg[1] < 95: tqg = [x * 2 for x in tqg]
            leastg = max(covg, key=lambda p: covg[p]["far"] * covg[p]["present"]) if covg else None; pmg = PER.get(name)
            cag = collections.Counter(); clg = collections.Counter()
            for i in refs:
                for ar in meta_art.get(ids[i], [])[:2]: cag[ar] += W[i]
                if LB[i]: clg[LB[i]] += W[i]
            GEN.append({"id": name, "scene": name, "references": int(len(refs)), "releases_month": round(pmg) if pmg else None, "coverage": covg, "least_covered": leastg,
                        "uncovered_month": round(pmg * covg[leastg]["far"] * covg[leastg]["present"]) if pmg and leastg else None,
                        "tempo": [round(x, 1) for x in tqg], "keys": [k for k, _ in ksg.most_common(3)], "minor": None if minor_g is None else round(float(minor_g), 2),
                        "balance": {p: round(float(np.median(shg[:, j])), 1) for j, p in enumerate(("drums", "bass", "melody", "voice"))},
                        "kick": [[p, round(k_ / sum(kpg.values()) * 100)] for p, k_ in kpg.most_common(2)] if sum(kpg.values()) >= 12 else [],
                        "swing": round(float(np.median(swg)), 3) if len(swg) >= 12 else None, "examples": [ids[i] for i in refs[order_g][:6]],
                        "artists": [x for x, _ in cag.most_common(6)], "labels": [x for x, _ in clg.most_common(5)], "rough": bool(len(refs) < 40), "styles": [], "_w": float(sum(W[i] for i in refs)), "gap_ids": gap_g,
                        "c": [round(float(x), 4) for x in gc], "ranges": {p: [round(float(np.percentile(shg[:, j], q)), 1) for q in (25, 50, 75)] for j, p in enumerate(("drums", "bass", "melody", "voice"))},
                        "djs": [[n_, k_] for n_, k_ in sum((WHO[i] for i in refs), collections.Counter()).most_common(8)]})
        if len(refs) < 40 or name == "unknown": continue
        Er = E[refs]; best_k, best_lab, best_sil = 1, np.zeros(len(refs), int), None
        for k in (2, 3, 4, 5):
            if len(refs) < k * 15: break
            labs = [KMeans(n_clusters=k, n_init=5, random_state=s).fit(Er).labels_ for s in range(4)]
            sizes = np.bincount(labs[0], minlength=k)
            if sizes.min() < max(15, 0.1 * len(refs)): continue
            ari = np.mean([adjusted_rand_score(labs[0], l2) for l2 in labs[1:]])
            if ari < 0.5: continue
            sil = silhouette_score(Er, labs[0])
            if best_sil is None or sil > best_sil: best_k, best_lab, best_sil = k, labs[0], sil
        cen = np.stack([Er[best_lab == c].mean(0) for c in range(best_k)]); cen /= np.linalg.norm(cen, axis=1, keepdims=True)
        dist_ref = 1 - (Er @ cen.T); rad = np.array([np.percentile(dist_ref[best_lab == c, c], 90) for c in range(best_k)])
        allr = np.where((sc == s_i) & has)[0]; dall = 1 - (E[allr] @ cen.T); near = dall.argmin(1)
        inside = dall[np.arange(len(allr)), near] <= rad[near]
        rel_share = np.array([np.mean(inside & (near == c)) for c in range(best_k)])
        wts = np.array([W[i] for i in refs], np.float32); dj_share = np.array([wts[best_lab == c].sum() for c in range(best_k)]) / wts.sum()
        report[name] = {"references": int(len(refs)), "styles": int(best_k), "silhouette": None if best_sil is None else round(float(best_sil), 3), "outside_every_range": round(float(1 - inside.mean()), 3)}
        for c in range(best_k):
            m = refs[best_lab == c]; mw = wts[best_lab == c]
            order = np.argsort(dist_ref[best_lab == c, c]); examples = [ids[i] for i in m[order][:6]]
            ca = collections.Counter(); cl = collections.Counter()
            for i, w_ in zip(m, mw):
                for ar in meta_art.get(ids[i], [])[:2]: ca[ar] += float(w_)
                if LB[i]: cl[LB[i]] += float(w_)
            cov = {}; gap_ids = {}
            for part in LIB:
                pi = FAM[part][1]; pres = []; bs = []; far_ids = []
                for i in m:
                    r = rat.get(ids[i])
                    if r is None: continue
                    sh_ = np.clip(V[r, :, 52], 0, None); w8 = sh_[pi] / max(float(sh_.sum()), 1e-9) >= 0.08; pres.append(w8)
                    if w8:
                        b = best_loop(i, part)
                        if b is not None:
                            bs.append(b)
                            if b < 0.45: far_ids.append(ids[i])
                if far_ids: gap_ids[part] = far_ids
                bs = np.array(bs); present = float(np.mean(pres)) if pres else 0.0
                if len(bs) >= 10: cov[part] = {"present": round(present, 3), "records": int(len(bs)), "close": round(float((bs >= 0.7).mean()), 3), "some": round(float(((bs >= 0.45) & (bs < 0.7)).mean()), 3), "far": round(float((bs < 0.45).mean()), 3)}
            rs = [rat.get(ids[i]) for i in m]; rs = [r for r in rs if r is not None]
            shp = np.clip(V[rs][:, :, 52], 0, None); shp = shp / np.maximum(shp.sum(1, keepdims=True), 1e-9) * 100
            ks = collections.Counter(CAM[key[i]] for i in m if key[i] < 24); minor = np.mean([key[i] < 12 for i in m if key[i] < 24]) if ks else None
            rh = [RH[ids[i]] for i in m if ids[i] in RH]; kp = collections.Counter(x[0] for x in rh if x[0])
            sw = [x[1] for x in rh if isinstance(x[1], (int, float))]
            least = max(cov, key=lambda p: cov[p]["far"] * cov[p]["present"]) if cov else None
            pm = PER.get(name)
            tq = [float(np.percentile(tm[m], q)) for q in (25, 50, 75)]
            if tq[1] < 95: tq = [x * 2 for x in tq]   # Beatport lists much drum and bass and dubstep at half tempo
            CEN.append(cen[c])
            out.append({"id": f"{name}:{c}", "scene": name, "references": int(len(m)), "artists": [x for x, _ in ca.most_common(6)], "labels": [x for x, _ in cl.most_common(5)],
                        "release_share": round(float(rel_share[c]), 3), "releases_month": round(pm * float(rel_share[c])) if pm else None, "dj_share": round(float(dj_share[c]), 3),
                        "dj_vs_releases": round(float(dj_share[c] / max(rel_share[c], 1e-3)), 2), "coverage": cov, "least_covered": least,
                        "uncovered_month": round(pm * float(rel_share[c]) * cov[least]["far"] * cov[least]["present"]) if pm and least else None,
                        "tempo": [round(x, 1) for x in tq], "keys": [k for k, _ in ks.most_common(3)], "minor": None if minor is None else round(float(minor), 2),
                        "balance": {p: round(float(np.median(shp[:, j])), 1) for j, p in enumerate(("drums", "bass", "melody", "voice"))},
                        "kick": [[p, round(k_ / sum(kp.values()) * 100)] for p, k_ in kp.most_common(2)] if sum(kp.values()) >= 12 else [],
                        "swing": round(float(np.median(sw)), 3) if len(sw) >= 12 else None, "examples": examples, "rough": bool(len(m) < 40), "gap_ids": gap_ids,
                        "c": [round(float(x), 4) for x in cen[c]], "ranges": {p: [round(float(np.percentile(shp[:, j], q)), 1) for q in (25, 50, 75)] for j, p in enumerate(("drums", "bass", "melody", "voice"))},
                        "djs": [[n_, k_] for n_, k_ in sum((WHO[i] for i in m), collections.Counter()).most_common(8)]})
    # the map: classical scaling of the cosine distances between style centres, then relaxed so dots do not overlap
    ns_ = len(CEN); C = np.stack(CEN + GCEN); Dm = np.clip(1 - C @ C.T, 0, 2); nn = len(C); J = np.eye(nn) - 1 / nn; Bm = -0.5 * J @ (Dm ** 2) @ J
    w_, v_ = np.linalg.eigh(Bm); o_ = np.argsort(w_)[::-1][:2]; XY = v_[:, o_] * np.sqrt(np.maximum(w_[o_], 1e-9))
    XY = (XY - XY.min(0)) / np.maximum(XY.max(0) - XY.min(0), 1e-9)
    Wd, Hd, pad = 350.0, 330.0, 26.0; P_ = np.stack([pad + XY[:, 0] * (Wd - 2 * pad), pad + XY[:, 1] * (Hd - 2 * pad)], 1)
    def relax(idx, gap):
      for _ in range(200):
        moved = False
        for a_ in range(len(idx)):
            for b_ in range(a_ + 1, len(idx)):
                i, j = idx[a_], idx[b_]
                dv = P_[j] - P_[i]; dd = float(np.hypot(*dv))
                if dd < gap:
                    moved = True; u = dv / (dd or 1e-3) if dd else np.array([1.0, 0.0]); P_[i] -= u * (gap - dd) / 2; P_[j] += u * (gap - dd) / 2
        P_[:, 0] = np.clip(P_[:, 0], pad, Wd - pad); P_[:, 1] = np.clip(P_[:, 1], pad, Hd - pad)
        if not moved: break
    relax(list(range(ns_)), 17); relax(list(range(ns_, nn)), 30)
    S2 = C[:ns_] @ C[:ns_].T; pairs = set()
    for i in range(ns_):
        for j in np.argsort(-S2[i])[1:3]: pairs.add((min(i, int(j)), max(i, int(j))))
    for i, o in enumerate(out): o["xy"] = [round(float(P_[i, 0]), 1), round(float(P_[i, 1]), 1)]
    for gi, g in enumerate(GEN): g["xy"] = [round(float(P_[ns_ + gi, 0]), 1), round(float(P_[ns_ + gi, 1]), 1)]
    # shares across genres: only genres Beatport's release count covers take part; a count of 10,000 is Beatport's cap
    cnt = [g for g in GEN if g["releases_month"]]; tw = sum(g["_w"] for g in cnt) or 1; tp = sum(g["releases_month"] for g in cnt) or 1
    for g in GEN:
        w_ = g.pop("_w"); g["capped"] = bool(g["releases_month"] and g["releases_month"] >= 10000)
        if g["releases_month"]:
            g["dj_share"] = round(w_ / tw, 3); g["release_share"] = round(g["releases_month"] / tp, 3); g["dj_vs_releases"] = round(g["dj_share"] / max(g["release_share"], 1e-3), 2)
        else:
            g["dj_share"] = g["release_share"] = g["dj_vs_releases"] = None
    capped = {g["scene"] for g in GEN if g["capped"]}
    for o in out: o["capped"] = o["scene"] in capped
    for o in out:
        for g in GEN:
            if g["scene"] == o["scene"]: g["styles"].append(o["id"])
    PAIRS = [[i, j, round(float(S2[i, j]), 3)] for i, j in sorted(pairs)]
    order = sorted(range(ns_), key=lambda i: -(out[i]["uncovered_month"] or 0)); remap = {old: new for new, old in enumerate(order)}
    out = [out[i] for i in order]; PAIRS = [[remap[i], remap[j], s_] for i, j, s_ in PAIRS]
    res = {"note": __doc__.split("\n  python")[0].strip(), "sources": dict(src), "scenes": report, "opportunities": out, "genres": sorted(GEN, key=lambda g: -(g["uncovered_month"] or 0)), "pairs": PAIRS, "map": {"w": 350, "h": 330}}
    json.dump(res, open(a.out or os.path.join(D, "pack-opportunities.json"), "w"), separators=(",", ":"))
    print(json.dumps({"opportunities": len(out), "scenes": len(report), "sources": dict(src)}))

if __name__ == "__main__":
    main()
