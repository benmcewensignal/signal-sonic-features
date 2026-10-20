"""The loops the matcher should never rank, from every licensed loop heard once (tools/swap_loop_flags.py, whose
measures the swap-loop-flags workflow keeps on the swap-test-results branch as loop-audio-measures.json), under one
stated rule. A loop is flagged when:
  - it is silent: summed to mono it is under -45 dBFS, or nine tenths of its 50 ms frames are under -50 dBFS (the rule
    the worker already applies to a prepared cycle, applied here before anyone asks for it);
  - its two channels cancel when summed: the mono sum is 12 dB or more under the channels' own level;
  - it is one hit: a single rise in level, with half or more of its sound in its loudest tenth (a kick, a crash, one
    piano note or a swell decaying, not a loop);
  - it is not whole bars at its tempo (worker/swap.py loop_bars, the rule loop_shape refuses it by once downloaded).
Writes worker/loop_flags.json: {id: reason}. The worker's choose() skips them, so the five loops a part is offered, and
/reach's estimate of how close the library gets, come from loops that can be laid.
  python tools/swap_loop_flags_apply.py loop-audio-measures.json"""
import collections, datetime, json, sys

RULES = (("silent", lambda r: r["rms_db"] < -45 or r["silent"] >= 0.9),
         ("cancels in mono", lambda r: r.get("mono_vs_channels_db") is not None and r["mono_vs_channels_db"] < -12),
         ("one hit", lambda r: r["hits"] <= 1 and r["top_tenth"] >= 0.5),
         ("not whole bars", lambda r: not r.get("loops_by_rule")))


def main(src, out="worker/loop_flags.json"):
    R = [r for r in json.load(open(src))["loops"] if not r.get("error")]
    flags, n = {}, collections.Counter(); by = collections.defaultdict(collections.Counter); fam = collections.Counter()
    for r in R:
        fam[r["fam"]] += 1
        for why, f in RULES:
            if f(r):
                flags[str(r["id"])] = why; n[why] += 1; by[r["fam"]][why] += 1
                break
    json.dump({"note": "licensed loops the matcher skips (tools/swap_loop_flags_apply.py): " + "; ".join(w for w, _ in RULES) +
                       " (the first that applies is given)", "built": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
               "heard": len(R), "by_family": {f: {"heard": fam[f], **by[f]} for f in sorted(fam)}, "flags": flags},
              open(out, "w"), separators=(",", ":"))
    print(f"{len(flags):,} of {len(R):,} loops flagged: " + ", ".join(f"{w} {n[w]:,}" for w, _ in RULES))
    for f in sorted(fam): print(f"  {f}: {sum(by[f].values()):,} of {fam[f]:,} ({sum(by[f].values()) / fam[f]:.1%})")


if __name__ == "__main__":
    main(*sys.argv[1:])
