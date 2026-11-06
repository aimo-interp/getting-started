"""Each white-box block alone, on the same outer folds as ablation_candidates.py (same seeds, same C, threshold .5).

    python analysis/report/standalone.py > analysis/report/standalone.txt
"""
import argparse, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold as SGK
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "cvbench"))
import cv, stack, whitebox as wb
REPEATS = 20
args = argparse.Namespace(crossfit=False, sample=0, frac=.5)
CFG = {"seg alone": "seg_safe", "traj alone": "traj", "attn alone": "attn"}
cases = cv.load_cases(list(cv.CASE_FILES))
y = cases.is_robust.astype(bool).to_numpy(); groups = cases.group.to_numpy()
short = np.array([m.split("/")[-1] for m in cases.model_id])
small = ~np.char.startswith(short, "gpt-oss")
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
P |= {"white-box (seg+traj+attn, nested)": A["attn -tx (tx)"], "full stack (16k)": A["attn FULL (tx)"], "tx only (16k)": A["tx only (tx)"], "tx only (4k)": A["tx only (tx4)"]}


def line(k, m, name):
    acc = ((P[k][:, m] >= .5) == y[m]).mean(1); au = np.mean([roc_auc_score(y[m], p[m]) for p in P[k]])
    return f"{name:6s} {k:36s} acc {acc.mean():.3f} +- {acc.std():.3f}  AUC {au:.3f}"


for m, name in ((np.ones(len(y), bool), "main"), (small, "small")):
    print(f"always robust {name}: {y[m].mean():.3f}")
    for k in P: print(line(k, m, name))
ms = sorted(set(short)); print("\nper model acc", ms)
for k in CFG: print(f"{k:12s}", [round(float(((P[k][:, short == s] >= .5) == y[short == s]).mean()), 3) for s in ms])
