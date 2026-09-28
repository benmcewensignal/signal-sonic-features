"""Read one uploaded clip into its parts with exactly the corpus's code.

Separation, then the same measures in the same order as the remeasure loop in features/stems.py
(drums first, so their beats exist for the kick pattern). The bassline is left out: its step
pattern is not publishable. Nothing is written anywhere; the caller deletes the clip.
"""
import tempfile, shutil
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


def read(wav_path, with_audio=False):
    work = tempfile.mkdtemp()
    try:
        st = S.separate(wav_path, work)
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
            try:
                from worker.embed import learned_call
                le = learned_call(wav_path)
                if le: out["scene_learned"] = le
            except Exception as e_:
                out["learned_error"] = type(e_).__name__
            from worker.scene import sounds_like
            sl = sounds_like(mix_x, rec)
            if sl: out["sounds_like"] = sl
        except Exception as e:
            out["scene_error"] = type(e).__name__
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
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)
