"""Per-case OOF predictions of each block on the matched folds of ablation_candidates.py (same seeds, C, threshold .5).

    python analysis/report/errors_collect.py   # -> analysis/report/errors_oof.npz
"""
import argparse, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold as SGK
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "cvbench"))
import cv, stack, whitebox as wb
REPEATS = 20
args = argparse.Namespace(crossfit=False, sample=0, frac=.5)
CFG = {"seg": "seg_safe", "traj": "traj", "attn": "attn"}
cases = cv.load_cases(list(cv.CASE_FILES))
y = cases.is_robust.astype(bool).to_numpy(); groups = cases.group.to_numpy()
blocks = stack.make_blocks(cases, sorted(CFG.values()), args)
P = {k: np.zeros((REPEATS, len(y))) for k in CFG}
for r in range(REPEATS):
    for tr, te in SGK(5, shuffle=True, random_state=r).split(y, y, groups):
        for k, b in CFG.items():
            F = blocks[b](tr)
            P[k][r, te] = wb._predict(wb._fit_logistic(F[tr], y[tr], stack.C), F[te], True)[1]
    print("repeat", r + 1, file=sys.stderr, flush=True)
A = np.load(HERE / "ablation_candidates.npz", allow_pickle=True)
assert (A["y"] == y).all()
P |= {"stack": A["attn FULL (tx)"], "whitebox": A["attn -tx (tx)"], "tx": A["tx only (tx)"]}
np.savez(HERE / "errors_oof.npz", y=y, case_id=cases.index.to_numpy().astype(str), **P)
