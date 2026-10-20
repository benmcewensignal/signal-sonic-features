"""Sample matching, first part: openly licensed drum loops from Freesound, measured as Sonic measures a
separated drum part, matched to each scene's drums.  FS_KEY in the environment.
  python tools/samples.py --pages 14 --out sample-matches.json
"""
import os, re, sys, json, argparse, tempfile, subprocess, numpy as np, requests
from multiprocessing import Pool
sys.path.insert(0, ".")
SC = ("level", "crest", "dynamic_span", "centroid_hz", "rolloff_hz", "flatness", "onsets_per_s", "share_of_energy")
KEY = os.environ.get("FS_KEY", "")
WORDS = {"drum-and-bass": ["drum and bass", "dnb", "jungle", "d&b", "amen"], "140-deep-dubstep-grime": ["dubstep", "grime", "140"], "dubstep": ["dubstep", "brostep"],
         "uk-garage-speed-garage": ["garage", "2step", "2-step", "ukg"], "breaks-breakbeat-uk-bass": ["break", "breaks"], "techno-peak-time": ["techno"],
         "hard-techno": ["techno", "hard"], "techno-raw-deep-hypnotic": ["techno", "hypnotic", "minimal"], "minimal-deep-tech": ["minimal", "tech house"],
         "tech-house": ["tech house", "tech-house"], "house": ["house"], "deep-house": ["deep house", "house"], "jackin-house": ["jackin", "house"],
         "funky-house": ["funky", "house"], "bass-house": ["bass house", "house"], "progressive-house": ["progressive"], "organic-house": ["organic", "house"],
         "afro-house": ["afro"], "african": ["afro", "african", "afrobeat"], "amapiano": ["amapiano", "log drum", "piano"], "uk-funky-gqom": ["gqom", "funky"],
         "brazilian-funk": ["baile", "funk", "brazil"], "latin-electronic": ["latin", "reggaeton", "salsa"], "psy-trance": ["psy", "trance"],
         "trance-main-floor": ["trance"], "trance-raw-deep-hypnotic": ["trance"], "hard-dance-hardcore": ["hardstyle", "hardcore", "hard dance", "gabber"],
         "trap-future-bass": ["trap", "future bass"], "downtempo": ["downtempo", "chill", "trip hop", "trip-hop"], "ambient-experimental": ["ambient", "experimental"],
         "electro": ["electro"], "electronica": ["electronica", "idm"], "nu-disco-disco": ["disco"], "indie-dance": ["indie", "disco"],
         "melodic-house-techno": ["melodic"], "mainstage": ["edm", "big room", "bigroom"]}

def known_ids():
    """The loops already in the library (data/loops-measured.json): searched past, not measured again."""
    try: return {l["id"] for l in json.load(open("data/loops-measured.json"))}
    except Exception: return set()


def search(pages):
    out, seen = [], set(known_ids())
    import time as _t
    QUERIES = ["drum loop", "percussion loop", "breakbeat loop", "house drum loop", "deep house drum loop", "tech house drum loop",
               "techno drum loop", "hard techno loop", "minimal techno loop", "drum and bass loop", "jungle break loop", "dnb drum loop 174",
               "dubstep drum loop 140", "grime drum loop", "uk garage drum loop", "2 step garage loop", "breaks loop 130", "trance drum loop",
               "psytrance drum loop", "hardstyle kick loop", "hardcore drum loop 160", "amapiano log drum loop", "afro house drum loop",
               "afrobeat drum loop", "baile funk loop", "brazilian funk beat", "gqom drum loop", "trap drum loop", "future bass drum loop",
               "downtempo drum loop", "ambient percussion loop", "electro drum loop", "disco drum loop", "nu disco loop", "latin percussion loop",
               "reggaeton drum loop", "progressive house drum loop", "big room drum loop", "jackin house loop", "funky house drum loop",
               "amapiano loop", "log drum", "afro house percussion loop", "afro tech loop", "shaker loop", "conga loop", "bongo loop",
               "tribal house loop", "tech house groove", "house percussion loop", "melodic techno drum loop", "organic house percussion",
               "latin house loop", "baile funk drums", "gqom beat", "nu disco drum loop", "electro breakbeat loop", "idm drum loop",
               "chillout drum loop", "hi hat loop", "top loop", "groove loop 124"]
    # beyond drums: bass, melody and vocal loops, so a record's other parts can be matched to licensed sounds too
    # a second round (October 2026): searches by tempo and by sound, so the thin families (bass, melody, voice) grow
    QUERIES = QUERIES + [f"drum loop {t} bpm" for t in (118, 120, 122, 124, 125, 126, 128, 130, 132, 135, 138, 140, 145, 150, 160, 170, 172, 174, 180)] + [
        "kick loop", "clap loop", "snare loop", "ride loop", "amen break", "think break", "two step drum loop", "techno percussion loop", "house top loop",
        "minimal percussion loop", "tribal percussion loop", "djembe loop", "tabla loop", "timbale loop", "cowbell loop", "tambourine loop", "drum fill loop",
        "broken beat loop", "halftime drum loop", "footwork drum loop", "juke drum loop", "ghetto house loop", "speed garage drum loop", "bassline house drums",
        "deep techno loop", "dub techno loop", "acid techno loop", "industrial techno loop", "hypnotic techno loop", "peak time techno loop", "schranz loop",
        "melodic house drums", "organic house drums", "downtempo percussion", "lofi drum loop", "trip hop drum loop", "breakbeat 128", "electro beat loop",
        "miami bass loop", "freestyle drum loop", "dancehall drum loop", "reggaeton loop", "dembow loop", "afrobeats drum loop", "kuduro loop", "batida loop",
        "singeli loop", "jersey club drum loop", "baltimore club loop", "footwork loop 160", "hardstyle drum loop", "rawstyle loop", "frenchcore loop",
        "uptempo loop", "happy hardcore drum loop", "gabber loop", "donk loop", "makina loop", "hard house drum loop", "bounce drum loop", "trance 138 drums",
        "psytrance kick bass loop", "goa trance loop", "progressive trance drums", "eurodance drum loop", "italo disco drum loop", "synthwave drum loop",
        "funk drum loop", "soul drum loop", "disco house drums", "french house drum loop", "filter house loop", "jackin drums 126", "latin house percussion"]
    FAMILIES = {"drums": (QUERIES, {"drums", "drum", "drum-loop", "drumloop", "beat", "breakbeat", "percussion", "loop"}),
                "bass": (["bass loop", "bassline loop", "sub bass loop", "reese bass loop", "acid bassline", "log drum loop", "house bassline", "techno bassline",
                          "dnb bass loop", "garage bassline", "808 bass loop", "deep house bass",
                          "synth bass loop", "moog bass loop", "fm bass loop", "wobble bass loop", "rolling bass loop", "neurofunk bass loop", "dubstep bass loop",
                          "trance bass loop", "psytrance bassline", "offbeat bass loop", "electro bass loop", "funk bass loop", "disco bass loop", "slap bass loop",
                          "minimal bass loop", "progressive house bass loop", "tech house bassline", "deep tech bass loop", "amapiano bass loop", "afro house bass loop",
                          "uk garage bass loop", "speed garage bass loop", "bassline house loop", "303 acid loop", "tb-303 loop", "acid line loop", "bass riff loop",
                          "bass groove loop", "bass sequence loop", "plucked bass loop", "picked bass loop", "fretless bass loop", "upright bass loop",
                          "bass loop 120 bpm", "bass loop 124 bpm", "bass loop 125 bpm", "bass loop 126 bpm", "bass loop 128 bpm", "bass loop 130 bpm",
                          "bass loop 135 bpm", "bass loop 140 bpm", "bass loop 150 bpm", "bass loop 160 bpm", "bass loop 170 bpm", "bass loop 174 bpm",
                          "hardstyle bass loop", "donk bass loop", "melodic techno bass loop", "deep house bassline 122", "jackin house bass", "organic house bass",
                          "latin bass loop", "reggaeton bass loop", "afrobeat bass loop", "dub bass loop", "breaks bass loop", "electro bassline"],
                         {"bass", "bassline", "sub", "808", "reese", "acid", "log-drum", "logdrum", "sub-bass"}),
                "melody": (["synth loop", "chord loop", "piano loop", "pad loop", "arp loop", "lead loop", "keys loop", "rhodes loop", "stab loop",
                            "amapiano piano loop", "deep house chords", "trance lead loop", "melodic techno loop",
                            "melody loop", "synth melody loop", "techno synth loop", "house piano loop", "organ loop", "strings loop", "string loop", "pluck loop",
                            "sequence loop", "arpeggio loop", "chord progression loop", "trance pluck loop", "deep house synth loop", "lofi piano loop",
                            "guitar loop", "electric guitar loop", "acoustic guitar loop", "funk guitar loop", "brass loop", "horn loop", "flute loop",
                            "sax loop", "saxophone loop", "bell loop", "marimba loop", "kalimba loop", "synth stab loop", "ambient pad loop", "drone loop",
                            "amapiano keys loop", "afro house synth loop", "latin piano loop", "disco strings loop", "psytrance lead loop", "hardstyle lead loop",
                            "dubstep synth loop", "dnb pad loop", "jungle pad loop", "garage chords loop", "2 step chords", "house organ loop", "house stab loop",
                            "piano house loop", "italo synth loop", "synthwave loop", "electro synth loop", "acid synth loop", "techno stab loop", "techno chord loop",
                            "trance chords loop", "uplifting trance loop", "progressive house synth loop", "big room lead loop", "future bass chords",
                            "trap melody loop", "lofi keys loop", "jazz piano loop", "soul keys loop", "gospel chords loop", "choir pad loop", "vox pad loop",
                            "chord stab loop 124", "synth loop 120 bpm", "synth loop 124 bpm", "synth loop 126 bpm", "synth loop 128 bpm", "synth loop 130 bpm",
                            "synth loop 135 bpm", "synth loop 140 bpm", "synth loop 150 bpm", "synth loop 170 bpm", "synth loop 174 bpm", "melody loop 128 bpm",
                            "piano loop 124 bpm", "piano loop 128 bpm", "chords loop 122 bpm", "keys loop 126 bpm", "pad loop 120 bpm", "arp loop 128 bpm",
                            "balearic loop", "afrobeats melody loop", "amapiano lead loop", "gqom synth loop", "baile funk melody", "kuduro synth loop"], {"synth", "chord", "chords", "piano", "keys", "pad", "arp", "arpeggio", "lead", "melody", "melodic", "rhodes", "organ", "stab"}),
                "vocals": (["vocal loop", "vocal chop", "acapella", "vocal phrase", "vocal hook", "house vocal", "vocal sample loop", "spoken word loop",
                            "female vocal loop", "male vocal loop", "vocal chant loop", "choir loop", "vocal ad lib", "rap vocal loop", "spoken vocal loop",
                            "vocal hook loop", "house vocal loop", "techno vocal loop", "garage vocal loop", "dnb vocal loop", "vocal shout loop", "vocal hum loop",
                            "vocoder vocal loop", "soul vocal loop", "gospel vocal loop", "afro vocal loop", "amapiano vocal loop", "chant loop", "vocal oh loop",
                            "vocal ah loop", "vocal phrase 120 bpm", "vocal loop 122 bpm", "vocal loop 124 bpm", "vocal loop 125 bpm", "vocal loop 126 bpm",
                            "vocal loop 128 bpm", "vocal loop 130 bpm", "vocal loop 135 bpm", "vocal loop 140 bpm", "vocal loop 150 bpm", "vocal loop 160 bpm",
                            "vocal loop 170 bpm", "vocal loop 174 bpm", "acapella 124 bpm", "acapella 128 bpm", "acapella house", "acapella techno", "acapella garage",
                            "acapella dnb", "acapella trance", "vocal sample house", "diva vocal loop", "soulful vocal loop", "deep house vocal loop",
                            "tech house vocal loop", "vocal stab loop", "vocal fx loop", "vocal texture loop", "humming loop", "whisper loop", "singing loop",
                            "sung phrase loop", "vocal melody loop", "vocal harmony loop", "vocal riff loop", "scat loop", "beatbox loop", "vocal percussion loop",
                            "female vocal 128", "male vocal 128", "female vocal 124", "vocal chop loop 128", "vocal chop house", "vocal chop future bass",
                            "vocal chop garage", "vocal chop dnb", "vocal spoken loop 128", "poetry loop", "vocal hook house 126", "pop vocal loop",
                            "rnb vocal loop", "vocal loop reggaeton", "latin vocal loop", "african vocal loop", "zulu vocal loop", "arabic vocal loop",
                            "indian vocal loop", "gregorian chant loop", "monk chant loop", "vocal drone loop",
                            # a third round (20 October 2026): the voice is the thinnest family (637 loops against 9,384 drums)
                            "voice loop", "vox loop", "vox chop", "vocal snippet loop", "vocal phrase loop", "vocal line loop", "vocal cut loop",
                            "vocal sample 120 bpm", "vocal sample 126 bpm", "vocal sample 128 bpm", "vocal sample 140 bpm", "vocal sample 174 bpm",
                            "vocal loop 118 bpm", "vocal loop 120 bpm", "vocal loop 132 bpm", "vocal loop 138 bpm", "vocal loop 145 bpm", "vocal loop 100 bpm",
                            "acapella 120 bpm", "acapella 126 bpm", "acapella 130 bpm", "acapella 140 bpm", "acapella 170 bpm", "acapella loop",
                            "mc vocal loop", "rap acapella", "grime vocal", "jungle vocal loop", "ragga vocal loop", "dub vocal loop", "dancehall vocal loop",
                            "trance vocal loop", "psytrance vocal", "breakbeat vocal loop", "disco vocal loop", "funk vocal loop", "afrobeat vocal loop",
                            "baile funk vocal", "gqom vocal", "amapiano vocal", "deep house vocal phrase", "melodic techno vocal", "techno spoken word",
                            "robot voice loop", "talkbox loop", "vocoder loop", "vocal pad loop", "ooh vocal loop", "aah vocal loop", "hey vocal loop",
                            "vocal shout", "crowd chant loop", "chant 128", "vocal yeah loop", "vocal riff", "sung loop", "singing phrase", "vocal adlib loop"],
                           {"vocal", "vocals", "voice", "acapella", "a-cappella", "vox", "chop", "singing", "sung", "spoken"})}
    for fam, (qs, _tags) in FAMILIES.items():
      for q in qs:
          for p in range(1, pages + 1):
              for attempt in range(6):   # Freesound limits requests per minute: wait and retry when told to slow down
                  r = requests.get("https://freesound.org/apiv2/search/text/", timeout=40, params={
                  "query": q, "filter": 'duration:[2.0 TO 30.0] license:("Creative Commons 0" OR "Attribution")',
                      "fields": "id,name,username,license,previews,tags,duration,ac_analysis", "page_size": 150, "page": p, "token": KEY})
                  _t.sleep(1.1)   # stay under sixty requests a minute
                  if r.status_code != 429: break
                  import time; time.sleep(15 + 10 * attempt)
              if r.status_code != 200: print(f"::notice title=search stopped::{q} page {p}: HTTP {r.status_code}", flush=True); break
              d = r.json()
              for x in d.get("results", []):
                  tags = set(x.get("tags") or [])
                  if x["id"] in seen or not (tags & _tags): continue
                  x["_q"] = q; x["_cat"] = fam
                  if fam != "drums" and not clean_family(x): continue
                  seen.add(x["id"]); out.append(x)
              if not d.get("next"): break
    return out


NEG = {"vocals": ("drum", "kick", "perc", "hat", "snare", "clap", "beat", "fight", "karate", "punch", "impact", "foley", "sfx", "effect"), "bass": ("drum", "kick", "hat", "snare", "clap"), "melody": ("drum", "kick", "hat", "snare", "clap", "vocal", "vox")}
POS_NAME = {"vocals": ("vocal", "vox", "voice", "acapella", "a capella", "chant", "sing", "sung", "spoken", "choir")}   # not "chop": a karate chop is not a voice

def clean_family(x):
    """Stricter family rules: vocal loops must name a voice, and no family but drums may name drum sounds."""
    fam = x.get("_cat"); txt = ((x.get("name") or "") + " " + " ".join(x.get("tags") or [])).lower()
    if any(w in txt for w in NEG.get(fam, ())): return False
    if fam in POS_NAME and not any(w in txt for w in POS_NAME[fam]): return False
    return True

def openverse_search(seen):
    """Openverse: one search across openly licensed audio (Freesound, Jamendo, Wikimedia and others). No key needed."""
    import time as _t
    Q = {"bass": ["bass loop", "bassline"], "melody": ["synth loop", "piano loop", "chord loop"], "vocals": ["vocal loop", "acapella"], "drums": ["drum loop"]}
    out = []
    for fam, qs in Q.items():
        for q in qs:
            for page in (1, 2):
                try:
                    r = requests.get("https://api.openverse.org/v1/audio/", params={"q": q, "license": "cc0,by", "page_size": 50, "page": page}, timeout=40, headers={"User-Agent": "earlysignal.live sonic"})
                    _t.sleep(1.5)
                    if r.status_code != 200: print(f"::notice title=openverse::{q} p{page}: HTTP {r.status_code}", flush=True); break
                    for it in r.json().get("results", []):
                        land = it.get("foreign_landing_url") or ""
                        if "freesound.org" in land: continue   # already searched at the source
                        if not it.get("url") or (it.get("duration") or 0) > 60000: continue   # loops, not whole tracks
                        x = {"id": "ov:" + str(it.get("id")), "name": it.get("title"), "username": it.get("creator"), "license": it.get("license_url") or it.get("license"),
                             "previews": {"preview-hq-mp3": it.get("url")}, "tags": [t_.get("name") for t_ in (it.get("tags") or []) if isinstance(t_, dict)],
                             "duration": (it.get("duration") or 0) / 1000, "_cat": fam, "_page": land, "_src": "openverse/" + str(it.get("source"))}
                        if x["id"] in seen or not clean_family(x): continue
                        seen.add(x["id"]); out.append(x)
                except Exception as e:
                    print(f"::notice title=openverse::{q}: {type(e).__name__}", flush=True); break
    return out

def ccmixter_search(seen):
    """ccMixter: stems and a cappellas made for remixing. No key needed. Attribution and CC0 only."""
    import time as _t
    Q = {"vocals": "acappella", "bass": "bass", "melody": "synth", "drums": "drums"}
    out = []
    for fam, tag in Q.items():
        try:
            r = requests.get("https://ccmixter.org/api/query", params={"f": "json", "tags": tag, "limit": 80, "sort": "rank"}, timeout=40, headers={"User-Agent": "earlysignal.live sonic"})
            _t.sleep(1.5)
            if r.status_code != 200: print(f"::notice title=ccmixter::{tag}: HTTP {r.status_code}", flush=True); continue
            for it in r.json() if isinstance(r.json(), list) else []:
                lic = (it.get("license_url") or "") + " " + (it.get("license_name") or "")
                if not (("/by/" in lic and "/by-" not in lic) or "zero" in lic.lower()): continue   # Attribution or CC0 only
                files = [f_ for f_ in (it.get("files") or []) if str(f_.get("download_url", "")).lower().endswith(".mp3")]
                if not files: continue
                x = {"id": "ccm:" + str(it.get("upload_id")), "name": it.get("upload_name"), "username": it.get("user_name"), "license": it.get("license_url") or it.get("license_name"),
                     "previews": {"preview-hq-mp3": files[0]["download_url"]}, "tags": str(it.get("upload_tags") or "").split(","), "duration": None,
                     "_cat": fam, "_page": it.get("file_page_url") or "", "_src": "ccmixter"}
                if x["id"] in seen or not clean_family(x): continue
                seen.add(x["id"]); out.append(x)
        except Exception as e:
            print(f"::notice title=ccmixter::{tag}: {type(e).__name__}", flush=True)
    return out

def named_tempo(x):
    import re
    txt = " ".join([x.get("name") or ""] + list(x.get("tags") or []))
    m = re.search(r"(\d{2,3})\s*[-_ ]?\s*bpm", txt, re.I) or re.search(r"bpm\s*[-_ ]?\s*(\d{2,3})", txt, re.I)
    v = int(m.group(1)) if m else None
    return v if v and 60 <= v <= 200 else None


def _alarm(signum, frame): raise TimeoutError("one loop took over 90 seconds")


def measure(x):
    import signal
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(90)   # one bad file cannot stall a shard
    try:
        from features import stems as S   # loaded only where loops are measured
        url = (x.get("previews") or {}).get("preview-hq-mp3")
        if not url: return {"_fail": "no preview"}
        a = requests.get(url, timeout=40).content
        with tempfile.TemporaryDirectory() as t:
            mp3, wav = os.path.join(t, "a.mp3"), os.path.join(t, "a.wav")
            open(mp3, "wb").write(a)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-i", mp3, "-ar", "44100", "-ac", "2", wav], check=True, timeout=60)
            m = S.measure_stem(wav) or {}; e = S.analyse_stem(wav) or {}; m.update(e)
            tempo_named = named_tempo(x); tempo_measured = None
            try:
                rh = S.rhythm_of_stem(wav) or {}; tempo_measured = rh.get("beats_per_minute")
            except Exception:
                pass
            if not tempo_measured:   # short loops: a beat tracker on the loop itself
                try:
                    import librosa
                    y_, sr_ = librosa.load(wav, sr=22050, mono=True)
                    bt = librosa.beat.beat_track(y=y_, sr=sr_)[0]
                    bt = float(bt[0] if hasattr(bt, "__len__") else bt)
                    while bt < 90: bt *= 2
                    while bt > 185: bt /= 2
                    tempo_measured = round(bt, 1) if bt else None
                except Exception:
                    pass
        emb = m.get("embedding")
        if not (isinstance(emb, list) and len(emb) == 45): return {"_fail": "no sound profile (" + str(len(emb) if isinstance(emb, list) else type(emb).__name__) + ")"}
        v = [float(z) for z in emb] + [float(m.get(k)) if isinstance(m.get(k), (int, float)) else 0.0 for k in SC]
        tempo = tempo_named or tempo_measured
        return {"id": x["id"], "name": x["name"], "user": x["username"], "license": x["license"], "preview": url, "tempo": tempo,
                "tempo_from": "name" if tempo_named else ("measured" if tempo_measured else None), "duration": x.get("duration"), "v": v,
                "words": ((x.get("name") or "") + " " + " ".join(x.get("tags") or [])).lower()[:400]}
    except Exception as ex:
        return {"_fail": type(ex).__name__ + ": " + str(ex)[:80]}
    finally:
        signal.alarm(0)

def measure_stage(a):
    """One shard: every loop whose Freesound id falls to this shard, measured until the time budget runs out."""
    import time, glob
    if not KEY: sys.exit("no Freesound key")
    allf = json.load(open("found.json")) if os.path.exists("found.json") else search(a.pages)
    import zlib
    found = [x for x in allf if (x["id"] if isinstance(x["id"], int) else zlib.crc32(str(x["id"]).encode())) % a.of == a.shard][:a.max]
    print(f"shard {a.shard}: {len(found)} loops to measure", flush=True)
    import collections
    rows, t0, why = [], time.time(), collections.Counter()
    from concurrent.futures import ThreadPoolExecutor, as_completed
    def one(x):   # each loop in its own process, killed after 90 seconds: a stuck loop costs one loop, not the shard
        try:
            p = subprocess.run([sys.executable, "tools/measure_one.py"], input=json.dumps(x), capture_output=True, text=True, timeout=90)
            lines = [l for l in p.stdout.strip().splitlines() if l.startswith("{")]
            return json.loads(lines[-1]) if lines else {"_fail": f"crashed (code {p.returncode}): " + p.stderr.strip()[-50:]}
        except subprocess.TimeoutExpired:
            return {"_fail": "took over 90 seconds"}
    # compile the audio library's code once, before four processes start at once: simultaneous first runs can corrupt its
    # shared compiled-code cache, after which every loop's process on that runner crashes without a word
    subprocess.run([sys.executable, "-c", "import numpy as np, librosa; y=np.random.randn(22050).astype('float32'); "
                    "oe=librosa.onset.onset_strength(y=y, sr=22050); f=getattr(librosa.feature,'rhythm',None); (f.tempo if f else librosa.beat.tempo)(onset_envelope=oe, sr=22050); "
                    "librosa.feature.melspectrogram(y=y, sr=22050); import sys; sys.path.insert(0,'.'); from features import stems"], timeout=600)
    ex_ = ThreadPoolExecutor(4); futs = [ex_.submit(one, x) for x in found]
    try:
        for i, fu in enumerate(as_completed(futs)):
            r = fu.result()
            if r and "_fail" in r: why[r["_fail"][:60]] += 1
            elif r: rows.append(r)
            if i % 25 == 0:
                print(f"measured {i} of {len(found)} in {int(time.time() - t0)} s", flush=True)
                json.dump(rows, open(f"loops-{a.shard}.json", "w"))   # saved as it goes
            if time.time() - t0 > a.budget * 60: print("time budget reached; keeping what is measured", flush=True); break
    finally:
        for f_ in futs: f_.cancel()
        ex_.shutdown(wait=False, cancel_futures=True)
    json.dump(rows, open(f"loops-{a.shard}.json", "w")); print(f"shard {a.shard}: {len(rows)} loops measured", flush=True)
    print(f"::notice title=shard {a.shard}::" + json.dumps({"to measure": len(found), "measured": len(rows), "failed": dict(why.most_common(5)), "minutes": round((time.time() - t0) / 60, 1)}))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pages", type=int, default=14); ap.add_argument("--max", type=int, default=2000); ap.add_argument("--out", default="sample-matches.json")
    ap.add_argument("--stage", default="all"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--of", type=int, default=1); ap.add_argument("--budget", type=float, default=80)
    a = ap.parse_args()
    if a.stage == "measure": return measure_stage(a)
    if a.stage == "search":   # searched once, and the list handed to every shard, so ten runners do not all hit Freesound's limit
        f = search(a.pages); seen = {x["id"] for x in f} | known_ids()
        ov = openverse_search(seen); cc = ccmixter_search(seen); f = f + ov + cc
        json.dump(f, open("found.json", "w")); print("::notice title=search::" + json.dumps({"found": len(f), "openverse": len(ov), "ccmixter": len(cc)})); return
    if a.stage == "match":
        import glob
        rows = [r for f in sorted(glob.glob("loops-*.json")) for r in json.load(open(f))]
        print(f"loops measured across shards: {len(rows)}", flush=True)
        try:   # the library grows: what was measured before stays, in its order, and the new loops follow
            old = json.load(open("data/loops-measured.json")); have = {l["id"] for l in old}
            new = [r for r in rows if r.get("id") not in have]
            print(f"::notice title=library::{len(old)} loops kept, {len(new)} new of {len(rows)} measured this run", flush=True)
            rows = old + new
        except Exception as e:
            print(f"::warning title=library::no earlier library read ({type(e).__name__}); this run's loops alone", flush=True)
    else:
        if not KEY: sys.exit("no Freesound key")
        found = search(a.pages)[:a.max]; print(f"loops found: {len(found)}", flush=True)
        with Pool(4) as pool: rows = [r for r in pool.map(measure, found) if r and "_fail" not in r]
    print(f"loops measured: {len(rows)}", flush=True)
    all_rows = rows; rows = [r for r in all_rows if (r.get("cat") or "drums") == "drums"]
    P = json.load(open("data/drum-profiles.json")); keep = P["keep"]; mu, sd = np.array(P["mu"]), np.array(P["sd"])
    V = np.array([np.array(r["v"])[keep] for r in rows]); Z = (V - mu) / sd; Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    out = {"part": "drums", "loops": len(rows), "source": "Freesound (Creative Commons 0 and Attribution)", "scenes": {}}
    T = P.get("tempo", {}); fits = {}
    for s, m in P["scenes"].items():
        sim = Z @ np.array(m); st = T.get(s)
        ok = np.array([bool(st and r.get("tempo") and abs(r["tempo"] - st) / st <= 0.06) for r in rows])   # the same tempo: a half-tempo loop is a different groove
        fits[s] = int(ok.sum())
        kw = WORDS.get(s, [])
        named = np.array([any(k in (r.get("words") or "") for k in kw) for r in rows])
        score = sim + 0.25 * named   # sound first; a loop its creator labelled as this kind of music gets a disclosed nudge
        order, seen_ = [], set()
        for i in np.argsort(-score):
            if not ok[i]: continue
            key = (rows[i]["user"], re.sub(r"[^a-z]", "", (rows[i]["name"] or "").lower())[:10])
            if key in seen_ or rows[i]["user"] in {rows[j]["user"] for j in order[-3:]}: continue   # one per pack, varied creators
            seen_.add(key); order.append(i)
            if len(order) >= 25: break
        out["scenes"][s] = [{k: rows[i][k] for k in ("id", "name", "user", "license", "preview", "tempo", "duration")} | {"sim": round(float(sim[i]), 3), "labelled": bool(named[i])} for i in order]
    json.dump(out, open(a.out, "w"), separators=(",", ":"))
    # every measured loop's numbers, kept so the matches can be checked independently (does the drums model agree?)
    json.dump([{"id": r["id"], "cat": r.get("cat") or "drums", "page": r.get("page"), "src": r.get("src"), "name": r.get("name"), "user": r.get("user"), "license": r.get("license"), "preview": r.get("preview"),
                "tempo": r.get("tempo"), "key": r.get("key"), "key_from": r.get("key_from"), "duration": r.get("duration"), "words": r.get("words", "")[:120],
                "v": [round(z, 4) for z in r["v"]]} for r in all_rows],
              open("loops-measured.json", "w"), separators=(",", ":"))
    for s in ("drum-and-bass", "techno-peak-time", "deep-house", "amapiano"):
        if s in out["scenes"]: print(f"::notice title=matches {s}::" + "; ".join(f"{r['name'][:40]} ({round(r['tempo']) if r['tempo'] else '?'} BPM)" for r in out["scenes"][s][:5]))
    print("::notice title=loops at each scene's tempo::" + json.dumps(fits))
    import collections as _c
    print("::notice title=loops by family::" + json.dumps(dict(_c.Counter(r.get("cat") or "drums" for r in all_rows))))
    lab = {s: sum(1 for x in v[:5] if x.get("labelled")) for s, v in out["scenes"].items()}
    print("::notice title=top five labelled as the scene's music::" + json.dumps(lab))
    thin = sorted([s for s, n in fits.items() if n < 6]); print("::notice title=scenes with fewer than six loops at their tempo::" + json.dumps(thin))
    src = {}
    for r in rows: src[r.get("tempo_from")] = src.get(r.get("tempo_from"), 0) + 1
    print("::notice title=where each loop's tempo came from::" + json.dumps({str(k): v for k, v in src.items()}))

if __name__ == "__main__": main()
