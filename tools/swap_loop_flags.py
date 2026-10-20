"""Every loop in the licensed library, heard once: its length, level, how much of it is silence, how much of its sound
sits in its loudest tenth, how many hits it has, and whether its two channels cancel when summed. The worker's
loop_for refuses a prepared loop that is silent (19 October); this finds, ahead of any request, the files that are not
loops at all (a single hit decaying into silence, an empty file), so the matcher never ranks them. Writes the measures
for every loop (out/loop-audio-measures.json) and prints the shares by family; tools/swap_loop_flags_apply.py turns them
into worker/loop_flags.json under a stated rule. A loop already in out/loop-audio-measures.json (the last run's, which
the workflow fetches from the results branch) is not heard again: only loops added since.
  modal run tools/swap_loop_flags.py"""
import json, modal

app = modal.App("sonic-swap-loop-flags")
img = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg", "libsndfile1")
       .pip_install("numpy<2", "soundfile")
       .add_local_dir("worker", "/root/worker"))


@app.function(image=img, cpu=1.0, memory=1024, timeout=240, max_containers=48, retries=1)
def one(meta):
    import sys, urllib.request, math, numpy as np
    sys.path.insert(0, "/root")
    from worker import swap as SW
    lid = meta.get("id")
    try:
        x = SW.decode_bytes(urllib.request.urlopen(meta["preview"], timeout=30).read())
    except Exception as e:
        return {"id": lid, "error": type(e).__name__}
    if x is None or len(x) < SW.SR // 4: return {"id": lid, "error": "empty"}
    m = x.mean(1); n = len(m); sr = SW.SR
    rms = float(np.sqrt(np.mean(m ** 2))); both = float(np.sqrt(np.mean(x ** 2)))
    fr = int(0.05 * sr); E = (m[: n // fr * fr].reshape(-1, fr) ** 2).mean(1)
    top = np.sort(E)[::-1]; k = max(1, len(top) // 10)
    # hits: rises in the level (12 ms frames) of 3 dB or a fifth of the second-largest rise, 80 ms apart
    h = int(0.012 * sr); L = np.sqrt((m[: n // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-9; Ld = 20 * np.log10(L)
    rise = np.maximum(0, np.diff(Ld, prepend=-100.0))   # the file's first sound counts as a hit
    rise = rise * (Ld > Ld.max() - 40)                     # rises within 40 dB of the loudest frame
    hits, last = 0, -999
    thr = max(3.0, 0.2 * float(np.sort(rise)[-2])) if len(rise) > 1 else 1e9   # a fifth of the largest rise after the first
    for i in np.where(rise >= thr)[0]:
        if i - last >= int(0.08 / 0.012): hits += 1; last = i
    lt = float(meta.get("tempo") or 0)
    return {"id": lid, "fam": meta.get("_fam"), "name": (meta.get("name") or "")[:80], "d": round(n / sr, 2), "tempo": lt,
            "bars": round(n / sr * lt / 240, 2) if lt else None, "rms_db": round(20 * math.log10(rms), 1) if rms > 1e-9 else -180.0,
            "silent": round(float(np.mean(E < 1e-5)), 3), "top_tenth": round(float(top[:k].sum() / (E.sum() or 1)), 3), "hits": int(hits),
            "mono_vs_channels_db": round(20 * math.log10(rms / both), 1) if both > 1e-9 and rms > 1e-12 else None,
            "loops_by_rule": SW.loop_bars(n / sr, lt) is not None if lt else False}


@app.local_entrypoint()
def main():
    import sys, os, pickle, collections
    sys.path.insert(0, ".")
    LI = pickle.load(open("worker/loop_index.pkl", "rb"))
    metas = []
    for fam, L in LI.items():
        for m in L["meta"]:
            if m.get("preview"): metas.append(dict(m, _fam=fam))
    known = {}
    try:
        for r in json.load(open("out/loop-audio-measures.json"))["loops"]:
            if not r.get("error"): known[str(r["id"])] = r
    except Exception:
        pass
    now = {str(m.get("id")) for m in metas}
    todo = [m for m in metas if str(m.get("id")) not in known]
    print(f"{len(metas):,} loops in the library; {len(todo):,} not heard before", flush=True)
    res = [r for r in one.map(todo, return_exceptions=True, order_outputs=False) if isinstance(r, dict)] if todo else []
    res += [r for k, r in known.items() if k in now]
    os.makedirs("out", exist_ok=True)
    json.dump({"note": "every licensed loop heard once (tools/swap_loop_flags.py): d seconds; rms_db of the summed channels; silent: share of "
                       "50 ms frames under -50 dBFS; top_tenth: share of the energy in the loudest tenth of those frames; hits: rises in level "
                       "at least a fifth of the largest, 80 ms apart; mono_vs_channels_db: the summed level against the channels' own; "
                       "loops_by_rule: passes the worker's whole-bars rule", "loops": res}, open("out/loop-audio-measures.json", "w"))
    ok = [r for r in res if not r.get("error")]
    by = collections.defaultdict(list)
    for r in ok: by[r["fam"]].append(r)
    for fam, R in by.items():
        n = len(R)
        sh = lambda f: round(sum(1 for r in R if f(r)) / max(1, n), 3)
        print("::notice title=loop audio " + fam + "::" + json.dumps({
            "heard": n, "errors": sum(1 for r in res if r.get("fam") == fam and r.get("error")),
            "silent or under -45 dB": sh(lambda r: r["rms_db"] < -45 or r["silent"] >= 0.9),
            "cancel in mono (12 dB)": sh(lambda r: r["mono_vs_channels_db"] is not None and r["mono_vs_channels_db"] < -12),
            "one hit": sh(lambda r: r["hits"] <= 1), "two hits or fewer": sh(lambda r: r["hits"] <= 2),
            "loudest tenth >= .6": sh(lambda r: r["top_tenth"] >= 0.6), "silent >= .6": sh(lambda r: r["silent"] >= 0.6),
            "one hit and loudest tenth >= .5": sh(lambda r: r["hits"] <= 1 and r["top_tenth"] >= 0.5)}), flush=True)
    print(f"heard {len(ok):,} of {len(metas):,}", flush=True)
