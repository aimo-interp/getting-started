"""Permutation null for the out-of-fold seg-probe AUC per model (dead-end audit 2026-11-02).
Labels are shuffled within each model; if the shuffled OOF AUC is far from .5, the CV procedure itself is
biased (small-n diff-in-means anti-learning), and observed AUCs must be read against this null, not .5."""
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cvbench"))
import cv, stack, whitebox as wb

cases = cv.load_cases(list(cv.CASE_FILES))
y0 = cases.is_robust.astype(bool).to_numpy()
models, groups = cases.model_id.to_numpy(), cases.group.to_numpy()
acts = wb.load_acts(cases, cv)
ks, NP, NR = [0, 1, 3, 5], 30, 2
rng = np.random.default_rng(0)

def oof(y, k, r):
    s = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(y, y, groups):
        s[te] = np.mean([stack.dim_variant_scores(acts, models, y, tr, site, .5, pcs=k)[:, 0][te]
                         for site in ("seg_0_5", "seg_5_9")], 0)
    return s

def per_model_auc(y, s):
    return {m: roc_auc_score(y[models == m], s[models == m]) for m in np.unique(models)}

obs = {k: [per_model_auc(y0, oof(y0, k, r)) for r in range(NR)] for k in ks}
null = {k: [] for k in ks}
for p in range(NP):
    y = y0.copy()
    for m in np.unique(models):
        i = np.where(models == m)[0]; y[i] = rng.permutation(y[i])
    for k in ks:
        null[k].append(per_model_auc(y, oof(y, k, p)))
    print(f"perm {p+1}/{NP}", end="\r", file=sys.stderr)
print(f"observed OOF AUC (mean of {NR} fold seeds) | null mean [5%,95%] over {NP} within-model label shuffles; p = share of null >= observed")
for m in np.unique(models):
    line = f"{m.split('/')[-1]:28s} n={(models==m).sum():2d} frag={(~y0[models==m]).sum():2d} "
    for k in ks:
        o = np.mean([d[m] for d in obs[k]]); n = np.array([d[m] for d in null[k]])
        line += f"| k={k}: {o:.2f} vs {n.mean():.2f} [{np.quantile(n,.05):.2f},{np.quantile(n,.95):.2f}] p={(n>=o).mean():.2f} "
    print(line)
