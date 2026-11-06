"""Paired comparison of stack.py block sets on identical folds (analysis of 'dead ends', 2026-11-02).

The ±.016 quoted as noise is the across-repeat sd of ONE method's accuracy; two methods on the same folds are
paired, so the right yardstick is the sd of the per-repeat difference. Also reports log-loss / AUC, per-model
deltas, and how many case-predictions actually flip (right->wrong vs wrong->right) on an optional subset.

    python analysis/deadends/paired.py --base seg_safe+traj+tx --alt seg_safe+traj+tx+diff_fill seg_fill+traj+tx
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cvbench"))
import cv, stack, whitebox as wb

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--alt", nargs="+", required=True)
ap.add_argument("--repeats", type=int, default=20)
ap.add_argument("--frac", type=float, default=0.5)
ap.add_argument("--subset", default=None, help="'gated': cases whose seg windows are both gated off in the fold")
ap.add_argument("--save", default=None)
args = ap.parse_args()
args.crossfit = False; args.sample = 0

cases = cv.load_cases(list(cv.CASE_FILES))
y = cases.is_robust.astype(bool).to_numpy()
groups = cases.group.to_numpy()
short = np.array([m.split("/")[-1] for m in cases.model_id])
sets = [args.base] + args.alt
names = sorted({b for s in sets for b in s.split("+")} | ({"seg_safe"} if args.subset else set()))
blocks = stack.make_blocks(cases, names, args)

P = {s: np.zeros((args.repeats, len(y))) for s in sets}
G = np.zeros((args.repeats, len(y)), bool)
for r in range(args.repeats):
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(y, y, groups):
        cache = {b: blocks[b](tr) for b in names}
        for s in sets:
            F = np.concatenate([cache[b] for b in s.split("+")], 1)
            P[s][r, te] = wb._predict(wb._fit_logistic(F[tr], y[tr], stack.C), F[te], True)[1]
        if args.subset:
            G[r, te] = (cache["seg_safe"][te] == 0).all(1)
    print(f"repeat {r+1}/{args.repeats}", end="\r", file=sys.stderr)

def metrics(p, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    acc = ((p >= .5) == y)[m].mean()
    auc = roc_auc_score(y[m], p[m]) if 0 < y[m].sum() < m.sum() else np.nan
    ll = log_loss(y[m], np.clip(p[m], 1e-6, 1 - 1e-6), labels=[False, True])
    return acc, auc, ll

def summary(s):
    M = np.array([metrics(P[s][r]) for r in range(args.repeats)])
    return M
base = summary(args.base)
print(f"\n{'method':34s} {'acc':>13s} {'Δacc paired':>16s} {'auc':>6s} {'Δauc':>7s} {'logloss':>8s} {'Δll':>7s}")
for s in sets:
    M = summary(s); d = M - base
    print(f"{s:34s} {M[:,0].mean():.4f}±{M[:,0].std():.3f} {d[:,0].mean():+.4f}±{d[:,0].std():.4f} "
          f"{M[:,1].mean():.3f} {d[:,1].mean():+.3f}  {M[:,2].mean():.4f} {d[:,2].mean():+.4f}")
print("\nper-model accuracy (mean over repeats); Δ vs base")
hdr = "  ".join(f"{m[:12]:>12s}" for m in np.unique(short))
print(f"{'':34s} {hdr}")
for s in sets:
    acc = {m: ((P[s] >= .5) == y)[:, short == m].mean() for m in np.unique(short)}
    bac = {m: ((P[args.base] >= .5) == y)[:, short == m].mean() for m in np.unique(short)}
    print(f"{s:34s} " + "  ".join(f"{acc[m]:.3f}({acc[m]-bac[m]:+.2f})" for m in np.unique(short)))
print("\nprediction flips vs base (per repeat, mean): right->wrong / wrong->right / total cases" +
      (" | within gated-off subset" if args.subset else ""))
for s in args.alt:
    rb, ra = (P[args.base] >= .5) == y, (P[s] >= .5) == y
    rw, wr = (rb & ~ra).sum(1).mean(), (~rb & ra).sum(1).mean()
    line = f"{s:34s} {rw:5.2f} / {wr:5.2f} / {len(y)}"
    if args.subset:
        rwg, wrg = (rb & ~ra & G).sum(1).mean(), (~rb & ra & G).sum(1).mean()
        accb = (rb & G).sum() / G.sum(); acca = (ra & G).sum() / G.sum()
        aucs = [(roc_auc_score(y[G[r]], P[args.base][r][G[r]]), roc_auc_score(y[G[r]], P[s][r][G[r]]))
                for r in range(args.repeats) if 0 < y[G[r]].sum() < G[r].sum()]
        line += (f" | gated: {G.sum(1).mean():.1f} cases/rep, acc {accb:.3f}->{acca:.3f}, AUC "
                 f"{np.mean([a for a,_ in aucs]):.3f}->{np.mean([b for _,b in aucs]):.3f}, flips {rwg:.2f}/{wrg:.2f}")
    print(line)
if args.save:
    np.savez(args.save, y=y, short=short, G=G, **{s: P[s] for s in sets})
