"""Measure one loop, in its own process so a stuck decoder can be killed without harming the rest.
Reads the loop's Freesound record (JSON) on stdin; prints the measured row (JSON) on stdout."""
import sys, os, json, re, tempfile, subprocess
sys.path.insert(0, ".")
SC = ("level", "crest", "dynamic_span", "centroid_hz", "rolloff_hz", "flatness", "onsets_per_s", "share_of_energy")

def named_tempo(x):
    txt = " ".join([x.get("name") or ""] + list(x.get("tags") or []))
    m = re.search(r"(\d{2,3})\s*[-_ ]?\s*bpm", txt, re.I) or re.search(r"bpm\s*[-_ ]?\s*(\d{2,3})", txt, re.I)
    v = int(m.group(1)) if m else None
    return v if v and 60 <= v <= 200 else None

EDMA_MAJ = [.1652, .0475, .0829, .0669, .0999, .0927, .0529, .1316, .0522, .0744, .0694, .0643]
EDMA_MIN = [.1724, .0400, .0761, .1253, .0567, .0822, .0626, .1435, .0810, .0571, .0836, .0551]
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

def named_key(x):   # a key its creator put in the name or tags: "Am", "A minor", "C# min", "F major"
    txt = " " + " ".join([x.get("name") or ""] + list(x.get("tags") or [])) + " "
    m = re.search(r"[^A-Za-z]([A-G])([#b]?)[ _-]?((?i:minor|min|maj|major|m))?(?=[^A-Za-z])", txt)
    if not m: return None
    root = FLAT.get(m.group(1) + m.group(2), m.group(1) + m.group(2)); q = (m.group(3) or "").lower()
    if not q: return None   # a bare letter is too often not a key
    return root + ("m" if q.startswith("m") and q != "maj" and q != "major" else "")

def measured_key(emb):   # from the loop's pitch profile, with the dance-music key templates
    import numpy as np
    ch = np.array(emb[26:38], float); best = (None, -2.0)
    for i in range(12):
        r = np.roll(ch, -i)
        for prof, suf in ((EDMA_MAJ, ""), (EDMA_MIN, "m")):
            c = float(np.corrcoef(r, prof)[0, 1])
            if c > best[1]: best = (KEYS[i] + suf, c)
    return best[0]

def main():
    x = json.load(sys.stdin)
    import numpy as np, librosa, requests
    from features import stems as S
    with tempfile.TemporaryDirectory() as t:
        wav = os.path.join(t, "a.wav")
        if x.get("path"):   # a local file, for testing
            subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-i", x["path"], "-ar", "44100", "-ac", "2", wav], check=True, timeout=60)
        else:
            url = (x.get("previews") or {}).get("preview-hq-mp3")
            if not url: print(json.dumps({"_fail": "no preview"})); return
            mp3 = os.path.join(t, "a.mp3"); open(mp3, "wb").write(requests.get(url, timeout=30).content)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-i", mp3, "-ar", "44100", "-ac", "2", wav], check=True, timeout=60)
        m = S.measure_stem(wav) or {}; m.update(S.analyse_stem(wav) or {})
        tempo = named_tempo(x); src = "name" if tempo else None
        if not tempo:   # a quick tempo estimate from the loop's onsets: the drum-tempo function needs a full-length part
            y, sr = librosa.load(wav, sr=22050, mono=True); oe = librosa.onset.onset_strength(y=y, sr=sr)
            f = getattr(librosa.feature, "rhythm", None); tp = (f.tempo if f else librosa.beat.tempo)(onset_envelope=oe, sr=sr)
            tp = float(np.atleast_1d(tp)[0])
            while tp and tp < 85: tp *= 2
            while tp > 190: tp /= 2
            tempo, src = (round(tp, 1), "measured") if tp else (None, None)
    emb = m.get("embedding")
    if not (isinstance(emb, list) and len(emb) == 45): print(json.dumps({"_fail": "no sound profile"})); return
    v = [float(z) for z in emb] + [float(m.get(k)) if isinstance(m.get(k), (int, float)) else 0.0 for k in SC]
    nk = named_key(x); key, key_from = (nk, "name") if nk else (measured_key(emb), "measured")
    print(json.dumps({"id": x.get("id"), "cat": x.get("_cat") or "drums", "key": key, "key_from": key_from, "name": x.get("name"), "user": x.get("username"), "license": x.get("license"),
                      "preview": (x.get("previews") or {}).get("preview-hq-mp3"), "tempo": tempo, "tempo_from": src, "duration": x.get("duration"), "v": v,
                      "words": ((x.get("name") or "") + " " + " ".join(x.get("tags") or [])).lower()[:400]}))

if __name__ == "__main__":
    try: main()
    except Exception as ex: print(json.dumps({"_fail": type(ex).__name__ + ": " + str(ex)[:80]}))
