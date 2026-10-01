"""Read one uploaded clip into its parts with exactly the corpus's code.

Separation, then the same measures in the same order as the remeasure loop in features/stems.py
(drums first, so their beats exist for the kick pattern). The bassline is left out: its step
pattern is not publishable. Nothing is written anywhere; the caller deletes the clip.
"""
import os, tempfile, shutil
import numpy as np
from features import stems as S


def voice_type(v):
    fl = v.get("flatness") or 0; sh = v.get("share_of_energy") or 0; on = v.get("onsets_per_s") or 0
    return "air" if fl > 0.015 else "chops" if on > 5 and sh < 0.07 else "singing" if sh > 0.084 and on < 3.8 else "voice"


def _part_audio(st):
    """Each separated part as a small MP3 (64 kbps mono), base64, for the listener to hear."""
    import subprocess, base64, os
    out = {}
    for k, path in st.items():
        mp3 = path + ".mp3"
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-i", path, "-ac", "1", "-b:a", "64k", mp3], check=True, timeout=60)
            with open(mp3, "rb") as f: out[k] = base64.b64encode(f.read()).decode()
        except Exception:
            pass
        finally:
            try: os.remove(mp3)
            except Exception: pass
    return out


def leverage(st, out, donor):
    """Which part moves the track most toward its target. The page names a donor, a record near the middle of the
    target with a Beatport preview, and the target's centre in the learned ear. The donor is separated like the
    bounce, its parts stretched to the bounce's tempo, and each of the bounce's parts swapped for the donor's in turn;
    every mix, the bounce rebuilt from its own parts, and the donor itself are placed in the learned ear. A part's
    share is how far its swap moves the track toward the target, as a share of the way to where the donor sits.
    Tested (worker/part_swap_test.py): swapping drums moved tracks 62% of the way on median, bass 14%."""
    import subprocess, urllib.request, soundfile as sf, librosa
    from worker.embed import learned_call
    url = str(donor.get("url") or "")
    if not url.startswith("https://geo-samples.beatport.com/"): return {"error": "donor"}
    c = np.array([float(x) for x in (donor.get("centre") or [])][:16], np.float32)
    if len(c) != 16 or not np.linalg.norm(c): return {"error": "centre"}
    c /= np.linalg.norm(c); w = tempfile.mkdtemp()
    try:
        mp3, dwav = os.path.join(w, "d.mp3"), os.path.join(w, "d.wav"); urllib.request.urlretrieve(url, mp3)
        dur = float(subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", mp3], capture_output=True, text=True).stdout.strip() or 0)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-ss", str(max(0.0, (dur - 60) / 2)), "-t", "60", "-i", mp3, "-ac", "2", "-ar", "44100", dwav], check=True, timeout=120)
        dst = S.separate(dwav, os.path.join(w, "sep"))
        B, D, sr = {}, {}, 44100
        for k, pth in st.items():
            x, sr = sf.read(pth, always_2d=True); B[k] = x.astype(np.float32)
        for k, pth in dst.items():
            x, sr_d = sf.read(pth, always_2d=True); D[k] = x.astype(np.float32)
        n = min(len(x) for x in B.values())
        tb = float(out.get("tempo_read") or 0); td = float(donor.get("bpm") or 0); stretched = False
        if tb and td:
            f = min((1, 2, 0.5), key=lambda m: abs(td * m - tb)); rate = tb / (td * f)
            if 0.8 <= rate <= 1.25 and abs(rate - 1) > 0.005:
                for k in D: D[k] = np.stack([librosa.effects.time_stretch(D[k][:, ch], rate=rate) for ch in range(D[k].shape[1])], 1).astype(np.float32)
                stretched = True
        for k in D:
            y = D[k]
            if len(y) < n: y = np.tile(y, (int(np.ceil(n / max(len(y), 1))), 1))
            D[k] = y[:n]
        for k in B: B[k] = B[k][:n]
        def ear(parts, tag):
            mix = sum(parts.values()); pk = float(np.abs(mix).max()) or 1.0; mp = os.path.join(w, tag + ".wav"); sf.write(mp, mix * (0.89 / pk if pk > 0.89 else 1.0), sr)
            mono = os.path.join(w, tag + "_32k.wav"); subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-i", mp, "-ac", "1", "-ar", "32000", mono], check=True, timeout=120)
            v = ((learned_call(mono) or {}).get("ear") or {}).get("v")
            if not v: return None
            v = np.array(v, np.float32)[:16]; return v / (np.linalg.norm(v) or 1)
        e0 = ear(B, "rebuilt"); ed = ear(D, "donor")
        if e0 is None or ed is None: return {"error": "ear"}
        s0, sd = float(e0 @ c), float(ed @ c); res = {"donor": donor.get("id"), "stretched": stretched, "baseline": round(s0, 4), "donor_score": round(sd, 4), "parts": {}}
        for k in ("drums", "bass", "other", "vocals"):
            if k not in B or k not in D: continue
            e = ear({**B, k: D[k]}, "swap_" + k)
            if e is None: continue
            gain = float(e @ c) - s0
            res["parts"][k] = {"gain": round(gain, 4), "share": round(gain / (sd - s0), 3) if (sd - s0) > 0.01 else None}
        # the openly licensed loops nearest the donor's version of each part, at the bounce's tempo and key
        try:
            drec = {k: S.measure_stem(v) for k, v in dst.items()}
            for k, v in dst.items():
                emb = S.analyse_stem(v)
                if emb and isinstance(drec.get(k), dict): drec[k].update(emb)
            from worker.scene import licensed_parts
            lp = licensed_parts(drec, tb or None, (out.get("key") or {}).get("key"))
            if lp: res["loops"] = lp
        except Exception as e_:
            res["loops_error"] = type(e_).__name__
        return res
    finally:
        shutil.rmtree(w, ignore_errors=True)


def _with_chroma(st):
    """The plain measures of each part, with the part analyser's embedding for melody, bass and voice: the key is read from
    the twelve chroma values inside that embedding, so without it key_from_parts finds no key at all."""
    rec = {k: (S.measure_stem(v) or {}) for k, v in st.items()}
    for k in ("other", "bass", "vocals"):
        if k in st:
            try: rec[k].update(S.analyse_stem(st[k]) or {})
            except Exception: pass
    return rec


def ab_render(st, ab):
    """Your track with one part replaced by the openly licensed loop nearest the target's part: the loop found by the
    target style's part centre (its records' mean, in the matcher's raw space), at a workable tempo and, for bass and
    melody, a key that mixes; stretched to the track's tempo, laid from its first beat at the level of the part it
    replaces. Returns two short clips from where that part is strongest: the mix, and the mix with the loop. A sketch of a
    direction: the loop is not arranged to the track. Nothing is kept."""
    import os, base64, pickle, subprocess, tempfile, urllib.request
    import numpy as np, soundfile as sf, librosa
    from worker.scene import _mixes, key_from_parts
    part = ab.get("part"); fam = {"drums": "drums", "bass": "bass", "other": "melody", "vocals": "vocals"}.get(part)
    if not fam or part not in st: return {"error": "that part is not in the track"}
    w = tempfile.mkdtemp()
    S_ = {k: sf.read(p, always_2d=True) for k, p in st.items()}; sr = next(iter(S_.values()))[1]
    n = min(a.shape[0] for a, _ in S_.values()); S_ = {k: a[:n] for k, (a, _) in S_.items()}
    T = float(ab.get("bpm") or 0)
    if not T:
        try: T = float((S.rhythm_of_stem(st["drums"]) or {}).get("beats_per_minute") or 0)
        except Exception: T = 0.0
    tkey = ab.get("key")
    if not tkey and fam in ("bass", "melody"):
        try: tkey = (key_from_parts(_with_chroma(st)) or {}).get("camelot")
        except Exception: tkey = None
    with open(os.path.join(os.path.dirname(__file__), "loop_index.pkl"), "rb") as f: LI = pickle.load(f)
    L = LI[fam]; keep, mu, sd = L["keep"], np.asarray(L["mu"], float), np.asarray(L["sd"], float)
    v = np.asarray(ab.get("tc") or [], float)
    if v.shape[0] != 53: return {"error": "no target for that part"}
    z = (v[keep] - mu) / sd; z /= np.linalg.norm(z) + 1e-9; sim = L["Z"].astype(np.float32) @ z.astype(np.float32)
    pick = None
    for j in np.argsort(-sim)[:400]:
        m_ = L["meta"][int(j)]; lt = float(m_.get("tempo") or 0)
        if not m_.get("preview") or not lt: continue
        f_ = min((1.0, 2.0, 0.5), key=lambda q: abs(lt * q - T)) if T else 1.0
        if T and abs(lt * f_ - T) / T > 0.08: continue
        if fam in ("bass", "melody") and tkey and m_.get("key") and _mixes(tkey, m_.get("key")) is False: continue
        pick = (int(j), m_, lt * f_); break
    if not pick: return {"error": "no licensed loop near the target at this tempo and key"}
    j, m_, lt = pick
    mp3 = os.path.join(w, "loop.mp3"); urllib.request.urlretrieve(m_["preview"], mp3)
    y, _ = librosa.load(mp3, sr=sr, mono=False); y = np.atleast_2d(y)
    if y.shape[0] == 1: y = np.vstack([y, y])
    rate = (T / lt) if T else 1.0
    if abs(rate - 1) > 0.002: y = np.vstack([librosa.effects.time_stretch(ch, rate=rate) for ch in y])
    y = y.T
    try:
        d = S_.get("drums"); mono = d.mean(1) if d is not None else sum(S_.values()).mean(1)
        _, bt = librosa.beat.beat_track(y=mono.astype(np.float32), sr=sr, units="time"); t0 = float(bt[0]) if len(bt) else 0.0
    except Exception: t0 = 0.0
    lp = np.zeros((n, 2)); s0 = int(t0 * sr); k_ = s0
    while k_ < n:
        m = min(y.shape[0], n - k_); lp[k_:k_ + m] += y[:m]; k_ += y.shape[0]
    orig = S_[part]; act = np.abs(orig).mean(1) > 1e-4
    r0 = float(np.sqrt(np.mean(orig[act] ** 2))) if act.any() else 0.0; r1 = float(np.sqrt(np.mean(lp[s0:] ** 2))) or 1.0
    if r0: lp *= r0 / r1
    A = sum(S_.values()); B = sum(a for k, a in S_.items() if k != part) + lp
    win = int(20 * sr); hop = int(sr)
    en = [float(np.mean(orig[i:i + win] ** 2)) for i in range(0, max(1, n - win), hop)]
    i0 = int(np.argmax(en)) * hop if en else 0
    pk = max(float(np.max(np.abs(A[i0:i0 + win]))), float(np.max(np.abs(B[i0:i0 + win]))), 1e-6); g = min(1.0, 0.98 / pk)
    def enc(x, nm):
        wp, mp = os.path.join(w, nm + ".wav"), os.path.join(w, nm + ".mp3"); sf.write(wp, np.clip(x[i0:i0 + win] * g, -1, 1), sr)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-i", wp, "-b:a", "112k", mp], check=True, timeout=60)
        return base64.b64encode(open(mp, "rb").read()).decode()
    return {"part": part, "from_s": round(i0 / sr, 1), "tempo": round(T, 1) if T else None, "key": tkey, "closeness": round(float(sim[j]), 3),
            "loop": {k: m_.get(k) for k in ("id", "name", "user", "license", "page", "tempo", "key")}, "a": enc(A, "a"), "b": enc(B, "b")}


def read(wav_path, with_audio=False, donor=None, only_leverage=False, ab=None):
    work = tempfile.mkdtemp()
    try:
        st = S.separate(wav_path, work)
        if ab:
            try: return {"ab": ab_render(st, ab)}
            except Exception as e_: return {"ab": {"error": type(e_).__name__ + ": " + str(e_)[:140]}}
        if only_leverage and donor:
            # the second, lighter job Aim a track sends after the reading: tempo, key and the swaps, nothing else
            rec = _with_chroma(st); tempo_ = None
            try:
                rh = S.rhythm_of_stem(st["drums"]) if "drums" in st else None; tempo_ = (rh or {}).get("beats_per_minute")
            except Exception: pass
            out = {"tempo_read": tempo_}
            try:
                from worker.scene import key_from_parts
                kf = key_from_parts(rec)
                if kf: out["key"] = kf
            except Exception: pass
            try: out["leverage"] = leverage(st, out, donor)
            except Exception as e_: out["leverage"] = {"error": type(e_).__name__ + ": " + str(e_)[:120]}
            return out
        part_audio = _part_audio(st) if with_audio else None
        rec = {k: S.measure_stem(v) for k, v in st.items()}
        for k, v in sorted(st.items(), key=lambda kv: 0 if kv[0] == "drums" else 1):
            emb = S.analyse_stem(v)
            if emb and isinstance(rec.get(k), dict): rec[k].update(emb)
            if k == "drums" and isinstance(rec.get(k), dict):
                rh = S.rhythm_of_stem(v)
                if rh: rec[k].update(rh)
                bd = S.breakdowns_of_stem(v, (rh or {}).get("beats_per_minute"))
                if bd: rec[k].update(bd)
                kp = S.kick_pattern_of_stem(v, (rh or {}).get("_beats"))
                if kp: kp.pop("_rotation", None); rec[k].update(kp)
                rec[k].pop("_beats", None)
        tot = sum((s or {}).get("level", 0) for s in rec.values()) or 1
        for s in rec.values():
            if s: s["share_of_energy"] = round(s.get("level", 0) / tot, 4)
        v = rec.get("vocals") or {}
        out = {"model": S.MODEL, "parts": rec, "voice_type": voice_type(v) if v else None}
        # the scene call: the corpus analyser's measures of the whole mix, through the trained model
        try:
            from worker.scene import call, call_parts, features_of
            out["scene"] = call(wav_path)
            mix_x, _ = features_of(wav_path)
            tri = call_parts(mix_x, rec)
            if tri: out["scene_parts"] = tri
            try:
                from worker.scene import part_calls
                pc = part_calls(rec)
                if pc: out["part_calls"] = pc
                from worker.scene import key_from_parts
                kf = key_from_parts(rec)
                if kf: out["key"] = kf
                from worker.scene import part_sounds_like
                pl = part_sounds_like(rec)
                if pl: out["part_like"] = pl
            except Exception as e_:
                out["part_calls_error"] = type(e_).__name__
            pass   # the learned ear runs on its own below, so a failure in the scene calls cannot take it with them
            from worker.scene import sounds_like
            sl = sounds_like(mix_x, rec)
            if sl: out["sounds_like"] = sl
        except Exception as e:
            out["scene_error"] = type(e).__name__ + ": " + str(e)[:160]
        # the learned ear, independent of the scene calls: where the track sits, its progress and its targets depend on it
        if "scene_learned" not in out:
            try:
                from worker.embed import learned_call
                le = learned_call(wav_path)
                if le: out["scene_learned"] = le
            except Exception as e_:
                out["learned_error"] = type(e_).__name__ + ": " + str(e_)[:160]
        if part_audio: out["part_audio"] = part_audio
        try:   # which part pulls it away from its scene: against the best scene call available
            from worker.scene import part_residuals
            sc0 = None
            for k_ in ("scene_learned", "scene_parts", "scene"):
                v_ = out.get(k_) or {}
                if isinstance(v_, dict) and v_.get("scenes"): sc0 = v_["scenes"][0][0]; break
            pr = part_residuals(rec, sc0) if sc0 else None
            if pr: out["part_residuals"] = pr
            from worker.scene import licensed_parts
            tempo_ = ((rec.get("drums") or {}).get("beats_per_minute")) if isinstance(rec.get("drums"), dict) else None
            lp = licensed_parts(rec, tempo_, (out.get("key") or {}).get("key"))
            if lp: out["licensed_parts"] = lp
            # each part's measures (no audio), so the page can compare this track with any record, part by part
            SC_ = ("level", "crest", "dynamic_span", "centroid_hz", "rolloff_hz", "flatness", "onsets_per_s", "share_of_energy")
            pv = {}
            for p_ in ("drums", "bass", "other", "vocals"):
                s_ = rec.get(p_) or {}; e_ = s_.get("embedding")
                if isinstance(e_, list) and len(e_) == 45:
                    pv[p_] = [round(float(x), 5) for x in e_] + [round(float(s_.get(c)), 5) if isinstance(s_.get(c), (int, float)) else 0.0 for c in SC_]
            if pv: out["part_vectors"] = pv; out["tempo_read"] = tempo_
        except Exception as e_:
            out["part_residuals_error"] = type(e_).__name__
        if donor:
            try: out["leverage"] = leverage(st, out, donor)
            except Exception as e_: out["leverage"] = {"error": type(e_).__name__ + ": " + str(e_)[:120]}
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)
