"""How many of the licensed library's loops can Swap lay on a track's bars, family by family, under each rule: whether a
loop's length is a whole 1, 2, 4, 8 or 16 bars at its stated tempo (or half or double), within 4%, or within 4% short
and 20% long (a tail after the last bar); and how squarely its hits sit on a straight sixteenth grid at its own tempo,
read from the rise in its level (worker/swap.py, grid_fit) and from the rise in its spectrum, which sees a legato
bassline's note changes that its level does not. Annotations only.
  modal run tools/swap_diag.py"""
import json, modal

app = modal.App("sonic-swap-diag")
img = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg", "rubberband-cli", "libsndfile1")
       .pip_install("numpy<2", "soundfile")
       .add_local_dir("worker", "/root/worker"))


@app.function(image=img, cpu=1.0, memory=2048, timeout=300, max_containers=24)
def one(meta):
    import sys, urllib.request, math, numpy as np
    sys.path.insert(0, "/root")
    from worker import swap as SW
    try:
        x = SW.decode_bytes(urllib.request.urlopen(meta["preview"], timeout=30).read())
    except Exception as e:
        return {"id": meta.get("id"), "error": type(e).__name__}
    d = len(x) / SW.SR; lt = float(meta.get("tempo") or 0)
    if not lt: return {"id": meta.get("id"), "error": "no tempo"}
    best = None
    for q in (1.0, 2.0, 0.5):
        bf = d * lt * q / 240
        for N in (1, 2, 4, 8, 16):
            r = bf / N
            if best is None or abs(math.log(r)) < abs(math.log(best[0])): best = (r, N, q)
    r, N, q = best
    out = {"id": meta.get("id"), "d": round(d, 2), "lt": lt, "ratio_to_bars": round(r, 3), "N": N,
           "old_rule": bool(0.96 <= r <= 1.20), "new_rule": SW.loop_bars(d, lt) is not None}   # whole bars within 4% short and 20% long, 1/2/4/8/16 bars; the worker's opened rule (19 Oct: 60% long, 3/6/12 too)
    def flux_fit(y, tempo):
        m = y.mean(1); n = 1024; hop = 256
        if len(m) < n * 2: return 0.0
        fr = np.lib.stride_tricks.sliding_window_view(m, n)[::hop] * np.hanning(n)
        S = np.log1p(np.abs(np.fft.rfft(fr, axis=1)))
        fl = np.maximum(0, np.diff(S, axis=0)).sum(1); fl = np.r_[0, fl]
        t = (np.arange(len(fl)) * hop + n / 2) / SW.SR
        qq = 60 / tempo / 4; dd = np.abs(((t + qq / 2) % qq) - qq / 2)
        return float(fl[dd <= 0.02].sum() / (fl.sum() + 1e-12))
    def best_lead(y, tempo):
        """The lead in milliseconds (0-80) and the tempo within 2% that put the most of its hits on the grid."""
        m = np.abs(y.mean(1)); hop = 64
        e = m[: len(m) // hop * hop].reshape(-1, hop).max(1); rr = np.maximum(0, np.diff(e, prepend=e[:1]))
        t = (np.arange(len(rr)) + 0.5) * hop / SW.SR; tot = rr.sum() + 1e-12
        best = (0.0, 0.0, tempo)
        for k in np.linspace(-0.02, 0.02, 21):
            qq = 60 / (tempo * (1 + k)) / 4
            for lead in np.arange(0, 0.0801, 0.002):
                dd = np.abs(((t - lead + qq / 2) % qq) - qq / 2)
                f = float(rr[dd <= 0.015].sum() / tot)
                if f > best[0]: best = (f, lead, tempo * (1 + k))
        return best
    def flux_lead(y, tempo, max_s=0.06):
        """As the worker's lead (worker/swap.py, _lead) but read from the rise in the spectrum (12 ms windows, 1.5 ms
        apart), which sees a tonal loop's note starts where its level barely rises. On synthetic basslines, plucks and
        pads with known leads it reads 1 to 2 ms early; on synthetic drums, whose kicks sweep down in pitch, 8 ms late."""
        m = y.mean(1); n = 512; hop = 64
        if len(m) < n * 2: return None
        fr = np.lib.stride_tricks.sliding_window_view(m, n)[::hop] * np.hanning(n)
        S = np.log1p(np.abs(np.fft.rfft(fr, axis=1)))
        fl = np.r_[0, np.maximum(0, np.diff(S, axis=0)).sum(1)]
        t = (np.arange(len(fl)) * hop + n / 2) / SW.SR; tot = float(fl.sum()) + 1e-12
        qq = 60 / tempo / 4; leads = np.arange(0, max_s + 1e-9, 0.001)
        dd = np.abs(((t[None, :] - leads[:, None] + qq / 2) % qq) - qq / 2)
        fits = (fl[None, :] * (dd <= 0.015)).sum(1) / tot
        spread = (fl[None, :] * np.minimum(dd, 0.03)).sum(1) / tot
        ok = np.where(fits >= fits.max() - 0.02)[0]; k = int(ok[np.argmin(spread[ok])])
        return float(leads[k]), float(fits[k]), float(fits[0])
    if 0.96 <= r <= 1.04:
        te = N * 240 / d
        fx = flux_lead(x, te); lv = SW._lead(x, te)
        if fx: out.update(flux_lead_ms=round(1000 * fx[0], 1), flux_fit_lead=round(fx[1], 3), flux_fit0=round(fx[2], 3))
        out.update(level_lead_ms=round(1000 * lv[0], 1), level_fit_lead=round(lv[1], 3))
        bf, lead, tb = best_lead(x, lt * q)
        out.update(kind="clean", fit=round(SW.grid_fit(x, te), 3), flux=round(flux_fit(x, te), 3), best_fit=round(bf, 3), lead_ms=round(1000 * lead, 1), best_tempo_vs_stated=round(tb / (lt * q), 4), len_tempo_vs_stated=round(te / (lt * q), 4))
    elif 1.04 < r <= 1.2:
        # a tail: the tempo that puts the most of its hits on the grid within 2% of the stated one, cut to N bars
        bestf = None
        for k in np.linspace(-0.02, 0.02, 21):
            te = lt * q * (1 + k); y = x[: int(N * 240 / te * SW.SR)]
            ff = SW.grid_fit(y, te)
            if bestf is None or ff > bestf[0]: bestf = (ff, te, y)
        out.update(kind="tail", fit=round(bestf[0], 3), flux=round(flux_fit(bestf[2], bestf[1]), 3))
    else:
        out.update(kind="other")
    return out


@app.local_entrypoint()
def main():
    import sys, random
    sys.path.insert(0, ".")
    import pickle
    LI = pickle.load(open("worker/loop_index.pkl", "rb"))
    for fam, L in LI.items():
        metas = [m for m in L["meta"] if m.get("preview")]
        random.Random(7).shuffle(metas); metas = metas[:200]
        res = list(one.map(metas, return_exceptions=True))
        res = [r for r in res if isinstance(r, dict) and not r.get("error")]
        n = len(res)
        kinds = {k: sum(1 for r in res if r.get("kind") == k) for k in ("clean", "tail", "other")}
        def share(kind, key, th): return round(sum(1 for r in res if r.get("kind") in kind and r.get(key, 0) >= th) / max(1, n), 3)
        summ = {"family": fam, "read": n, "kinds": kinds,
                "eligible: old rule (whole bars within 4% short, 20% long)": round(sum(1 for r in res if r.get("old_rule")) / max(1, n), 3),
                "eligible: opened rule (60% long, 3/6/12-bar too)": round(sum(1 for r in res if r.get("new_rule")) / max(1, n), 3),
                "clean & level fit >= .45": share(("clean",), "fit", 0.45), "clean & level fit >= .3": share(("clean",), "fit", 0.3),
                "clean or tail & level fit >= .45": share(("clean", "tail"), "fit", 0.45), "clean or tail & flux fit >= .45": share(("clean", "tail"), "flux", 0.45),
                "clean or tail & flux fit >= .35": share(("clean", "tail"), "flux", 0.35),
                "median level fit (clean)": float(sorted([r["fit"] for r in res if r.get("kind") == "clean"] or [0])[len([r for r in res if r.get("kind") == "clean"]) // 2]),
                "median flux fit (clean)": float(sorted([r["flux"] for r in res if r.get("kind") == "clean"] or [0])[len([r for r in res if r.get("kind") == "clean"]) // 2]),
                "clean & best fit >= .45": share(("clean",), "best_fit", 0.45), "clean & best fit >= .6": share(("clean",), "best_fit", 0.6),
                "lead ms quantiles (clean)": sorted(r["lead_ms"] for r in res if r.get("kind") == "clean")[::max(1, len([r for r in res if r.get("kind") == "clean"]) // 8)],
                "best tempo / stated quantiles": sorted(r["best_tempo_vs_stated"] for r in res if r.get("kind") == "clean")[::max(1, len([r for r in res if r.get("kind") == "clean"]) // 8)],
                "length tempo / stated quantiles": sorted(r["len_tempo_vs_stated"] for r in res if r.get("kind") == "clean")[::max(1, len([r for r in res if r.get("kind") == "clean"]) // 8)],
                "ratio quantiles": [round(x, 3) for x in sorted(r["ratio_to_bars"] for r in res)[::max(1, n // 10)]]}
        cl = [r for r in res if r.get("kind") == "clean" and "flux_lead_ms" in r]
        def q8(v): v = sorted(v); return [round(float(x), 1) for x in v[::max(1, len(v) // 8)]] + ([round(float(v[-1]), 1)] if v else [])
        def med(v): v = sorted(v); return round(float(v[len(v) // 2]), 3) if v else None
        summ.update({"clean read for leads": len(cl),
                     "flux lead ms quantiles": q8([r["flux_lead_ms"] for r in cl]),
                     "flux fit at that lead / at none (medians)": [med([r["flux_fit_lead"] for r in cl]), med([r["flux_fit0"] for r in cl])],
                     "flux fit at lead >= .45": round(sum(1 for r in cl if r["flux_fit_lead"] >= 0.45) / max(1, len(cl)), 3),
                     "level lead ms quantiles (worker rule)": q8([r["level_lead_ms"] for r in cl]),
                     "flux minus level lead ms, where level fit >= .3": q8([r["flux_lead_ms"] - r["level_lead_ms"] for r in cl if r["level_fit_lead"] >= 0.3])})
        print("::notice title=swap loops " + fam + "::" + json.dumps(summ))
