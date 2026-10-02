"""Adds the smudge-robust measure to worker/loop_index.pkl: per family, W re-weights a part's measure (a Fisher
discriminant learned on practice mixes, keeping each loop's clean and smudged readings together and different loops apart)
and Zp is the library in that space. Separated parts are ranked by it; the closeness shown is still the plain measure.
On loops it never saw, an unmixed loop found its own original in the top three: bass 56 in 100, drums 68, melody 60,
vocals 87 (clean library 19, 39, 30, 52; smudged library 40, 49, 31, 65). worker/gap_tests.py::robust
  python tools/add_robust_measure.py"""
import pickle, numpy as np
LI = pickle.load(open("worker/loop_index.pkl", "rb")); R = np.load("data/robust-measure.npz")
for f, L in LI.items():
    if f + "_W" not in R: continue
    W, Zp = R[f + "_W"].astype(np.float32), R[f + "_Zp"].astype(np.float16)
    if W.shape[0] != L["Z"].shape[1] or Zp.shape[0] != L["Z"].shape[0]: print(f"  {f}: shapes differ, left as is"); continue
    L["W"], L["Zp"] = W, Zp; print(f"  {f}: measure {W.shape[0]} -> {W.shape[1]}, library {Zp.shape[0]}")
pickle.dump(LI, open("worker/loop_index.pkl", "wb"))
