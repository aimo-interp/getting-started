"""Audit of the 'difficulty fill' dead end: where does the seg gate switch off, and does difficulty carry
signal on exactly those models/cases?"""
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cvbench"))
import cv, stack, whitebox as wb, difficulty

cases = cv.load_cases(list(cv.CASE_FILES))
y = cases.is_robust.astype(bool).to_numpy()
models, groups = cases.model_id.to_numpy(), cases.group.to_numpy()
short = np.array([m.split("/")[-1] for m in models])
acts = wb.load_acts(cases, cv)
D = difficulty.diff_feats(cases, ("diff_last", "dim_last"))
Dz = pd.DataFrame(np.asarray(cv.zscore_within_model(D, cases), float), columns=D.columns, index=D.index)
print("NaN in difficulty:", D.isna().sum().to_dict(), "| unique values per model:", D.groupby(short).diff_last.nunique().to_dict())

def auc(yy, s):
    return roc_auc_score(yy, s) if 0 < yy.sum() < len(yy) else np.nan

print("== per-model: n, fragile, AUC(robust) of -difficulty (diff_last, dim_last), raw difficulty mean robust vs fragile")
for m in np.unique(short):
    k = short == m
    a1, a2 = auc(y[k], -D.diff_last.to_numpy()[k]), auc(y[k], -D.dim_last.to_numpy()[k])
    print(f"  {m:28s} n={k.sum():3d} frag={(~y[k]).sum():2d}  AUC -diff_last {a1:.2f}  -dim_last {a2:.2f}")
print(f"  pooled (z within model) AUC -diff_last {auc(y, -Dz.diff_last.to_numpy()):.2f}")

# gate frequency over outer folds (5 repeats x 5 folds)
gate = defaultdict(lambda: np.zeros(2)); nf = Counter()
rows_gated = np.zeros((len(y), 2)); rows_seen = np.zeros(len(y))
for rep in range(5):
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(np.zeros(len(y)), y, groups):
        g = stack.safe_seg_scores(acts, models, y, groups, tr)
        for m in np.unique(short):
            k = short == m
            gate[m] += (np.abs(g[k]).sum(0) == 0)
            nf[m] += 1
        rows_gated[te] += (g[te] == 0); rows_seen[te] += 1
print("\n== share of outer folds where each seg window is gated OFF (0-5, 5-9)")
for m in np.unique(short):
    print(f"  {m:28s} {gate[m][0]/nf[m]:.2f} {gate[m][1]/nf[m]:.2f}")

# signal of difficulty on cases where BOTH windows are gated off (that's what diff_fill feeds)
off = (rows_gated / rows_seen[:, None]).min(1) > 0.5
print(f"\n== cases with both windows gated off in >50% of folds: {off.sum()} ({dict(Counter(short[off]))})")
print(f"   robust rate there {y[off].mean():.2f}; AUC -diff_last within those cases (z within model) {auc(y[off], -Dz.diff_last.to_numpy()[off]):.2f}")
for m in np.unique(short[off]):
    k = off & (short == m)
    print(f"     {m:28s} n={k.sum()} frag={(~y[k]).sum()} AUC {auc(y[k], -D.diff_last.to_numpy()[k]):.2f}")

# what accuracy is even attainable? oracle threshold on difficulty within gated-off models
print("\n== ceiling: best single-threshold accuracy of difficulty within each model (in-sample, optimistic) vs always-True")
for m in np.unique(short):
    k = short == m; s = -D.diff_last.to_numpy()[k]; yy = y[k]
    best = max(((s >= t) == yy).mean() for t in np.r_[s, np.inf])
    print(f"  {m:28s} always-True {yy.mean():.2f}  best threshold {best:.2f}")
