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


def read(wav_path):
    work = tempfile.mkdtemp()
    try:
        st = S.separate(wav_path, work)
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
            from worker.scene import sounds_like
            sl = sounds_like(mix_x, rec)
            if sl: out["sounds_like"] = sl
        except Exception as e:
            out["scene_error"] = type(e).__name__
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)
