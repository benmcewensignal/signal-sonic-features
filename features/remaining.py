"""How many records still need converting. Prints one number."""
import json, os, sqlite3, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.convert import already_done, to_do
try:
    from features.analyser_local import LocalAnalyser
    want = LocalAnalyser().version.split("+")[0]
except Exception:
    want = "2"
try:
    _, remaining = to_do("sonic.db", already_done("out"), want, 1)
    print(remaining)
except Exception:
    print(0)
