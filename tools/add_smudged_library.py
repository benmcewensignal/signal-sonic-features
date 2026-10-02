"""Adds the smudged loop library to worker/loop_index.pkl as Zs, one per family.

Each library loop was mixed with others and unmixed with the records' own htdemucs (worker/gap_tests.py::cal); Zs is the
mean of its smudged measurements, so a record's separated part is searched against loops that went through the same
separation. Loops never smudged (about a third of drums) keep their clean entry. On fresh practice mixes, an unmixed loop
found its own original in the top three: bass 18 -> 37 in 100, drums 41 -> 57 (on covered loops), vocals 55 -> 67,
melody 29 -> 31 (worker/gap_tests.py::loops). Z, the clean library, stays for anything measured clean.
  python tools/add_smudged_library.py"""
import pickle, numpy as np
LI = pickle.load(open("worker/loop_index.pkl", "rb")); S = np.load("data/smudged-library.npz")
for f, L in LI.items():
    if f + "_Zs" not in S: continue
    Zs = S[f + "_Zs"].astype(np.float32)
    if Zs.shape != L["Z"].shape: print(f"  {f}: shapes differ, left clean"); continue
    has = np.linalg.norm(Zs, axis=1) > 0.5
    L["Zs"] = np.where(has[:, None], Zs, L["Z"].astype(np.float32)).astype(np.float16)
    print(f"  {f}: {int(has.sum())} of {len(has)} loops smudged")
pickle.dump(LI, open("worker/loop_index.pkl", "wb"))
