"""The parts worker's log during a Swap test (modal app logs --timestamps), summed up as annotations: each web call's
count, status and slowest time by path, and every line that looks like a container stopping (preempted, out of memory,
killed, an exit code, a traceback).
  python tools/swap_logsum.py modal-logs.txt"""
import re, sys, json
from collections import defaultdict

REQ = re.compile(r"^(\S+ \S+)\s+(GET|POST|OPTIONS|PUT|DELETE) (/[\w-]*)\S* -> (\d{3})[^(]*\(duration: ([\d.]+) (ms|s)")
SIGN = re.compile(r"preempt|out of memory|\boom\b|killed|exit code|terminat|interrupt|SIGTERM|SIGKILL|memory limit|Traceback|"
                  r"timed out|timeout|restart|Runner|crash|Error", re.I)


def main(path):
    lines = open(path, errors="replace").read().splitlines()
    calls = defaultdict(lambda: {"n": 0, "status": defaultdict(int), "slowest_s": 0.0, "over_100_s": 0})
    signs = []
    for L in lines:
        m = REQ.search(L)
        if m:
            t, meth, p, st, d, unit = m.groups()
            secs = float(d) / (1000 if unit == "ms" else 1)
            c = calls[f"{meth} {p}"]; c["n"] += 1; c["status"][st] += 1; c["slowest_s"] = max(c["slowest_s"], round(secs, 1))
            if secs > 100: c["over_100_s"] += 1
            if not st.startswith("2") and st not in ("401", "404"): signs.append(L[:240])
        elif SIGN.search(L) and "tensorflow" not in L and "dlopen" not in L:
            signs.append(L[:240])
    summ = {k: {"n": v["n"], "status": dict(v["status"]), "slowest_s": v["slowest_s"], "over_100_s": v["over_100_s"]} for k, v in sorted(calls.items())}
    print(f"::notice title=worker log calls::{len(lines)} lines | " + json.dumps(summ)[:3500], flush=True)
    if signs:
        for i in range(0, min(len(signs), 60), 12):
            print("::warning title=worker log signs " + str(i // 12 + 1) + "::" + " || ".join(signs[i:i + 12]).replace("%", "%25")[:3800], flush=True)
    else:
        print("::notice title=worker log signs::none", flush=True)


main(sys.argv[1])
