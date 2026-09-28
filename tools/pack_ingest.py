"""A sample pack from Drive, measured loop by loop exactly as Freesound loops are, for a private demo library.
Only measurements and the pack's own file paths are kept: the audio is not stored, published or played publicly.
  python tools/pack_ingest.py --zip pack.zip --pack "Loopmasters Free Sample Pack Drum & Bass" --out data/private/pack.json"""
import os, sys, json, re, zipfile, argparse, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
FAM = [("vocals", r"vocal|vox|acapella|a-capella|chant|spoken|phrase"), ("bass", r"bass|sub|reese|808"),
       ("melody", r"synth|chord|pad|lead|keys|piano|stab|arp|melod|music|rhodes|organ|string"), ("drums", r"drum|kick|snare|hat|perc|top|break|beat|groove|clap|ride|cymbal|full")]
def family(path):
    p = path.lower().replace(chr(92), "/")
    for fam, pat in FAM:
        if re.search(pat, p): return fam
    return None
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--zip"); ap.add_argument("--pack"); ap.add_argument("--out"); a = ap.parse_args()
    import io
    work = tempfile.mkdtemp(); items = []
    def walk(z, prefix=""):   # sound files, including inside zips within the zip (pack bundles often nest them)
        for n in z.namelist():
            if "__MACOSX" in n: continue
            if n.lower().endswith(".zip"):
                try: walk(zipfile.ZipFile(io.BytesIO(z.read(n))), prefix + os.path.splitext(os.path.basename(n))[0] + "/")
                except Exception as e: print("inner zip not read:", n, type(e).__name__, flush=True)
                continue
            if not re.search(r"[.](wav|aif|aiff|mp3|flac)$", n, re.I): continue
            yield_file(z, prefix + n, n)
    def yield_file(z, full, n):
        fam = family(full)
        if not fam: return
        dest = os.path.join(work, str(len(items)) + os.path.splitext(n)[1]); open(dest, "wb").write(z.read(n))
        items.append({"id": "pack:" + str(len(items)), "name": os.path.basename(full), "username": a.pack, "license": "pack licence (private demo)", "path": dest,
                      "tags": re.split(r"[/_ -]+", os.path.dirname(full)), "_cat": fam, "_page": "", "_src": "pack:" + a.pack, "rel": full})
    walk(zipfile.ZipFile(a.zip))
    print(f"sounds in the pack by family: " + json.dumps({f: sum(1 for i in items if i["_cat"] == f) for f, _ in FAM}), flush=True)
    def one(x):
        try:
            p = subprocess.run([sys.executable, "tools/measure_one.py"], input=json.dumps(x), capture_output=True, text=True, timeout=120)
            lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
            r = json.loads(lines[-1]) if lines else {"_fail": "no output"}
            if "_fail" not in r: r["rel"] = x["rel"]; r["duration"] = r.get("duration")
            return r
        except Exception as e:
            return {"_fail": type(e).__name__}
    with ThreadPoolExecutor(4) as ex: rows = list(ex.map(one, items))
    ok = [r for r in rows if "_fail" not in r]
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump({"pack": a.pack, "loops": ok}, open(a.out, "w"), separators=(",", ":"))
    print(f"::notice title=pack measured::" + json.dumps({"pack": a.pack, "sounds": len(items), "measured": len(ok), "by family": {f: sum(1 for r in ok if r.get("cat") == f) for f, _ in FAM}}))
if __name__ == "__main__": main()
