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


def read(wav_path, with_audio=False, donor=None, only_leverage=False):
    work = tempfile.mkdtemp()
    try:
        st = S.separate(wav_path, work)
        if only_leverage and donor:
            # the second, lighter job Aim a track sends after the reading: tempo, key and the swaps, nothing else
            rec = {k: S.measure_stem(v) for k, v in st.items()}; tempo_ = None
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
