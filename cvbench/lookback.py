#!/usr/bin/env python3
"""Lookback Lens attention features as a robustness signal (cvbench, 2026-10-29).

How much the CoT attends back to the problem rather than to its own reasoning, per attention head and
CoT segment (``generate.py --stages lookback`` -> ``data/cvbench/lookback/``). Hypothesis: a solve that
keeps re-reading the problem is anchored to it, so a perturbation of the problem moves it, while one that
runs on its own reasoning may not; which way it goes is left to the data. Variants fixed before the
data came in, all with cv.py conventions (z-score within model, 20x5-fold grouped CV, logistic C=0.3):

* ``lb_scalar``: mean lookback ratio over every head, CoT windows 0-5 and 5-9 (2 label-free features).
* ``lb_scalar_num``: + mean attention mass on prompt tokens with a digit, same windows (4 features).
* ``lb_heads``: Lookback Lens proper: the per-head ratio vector per window, projected per model on
  mean(robust) - mean(not robust) (the seg_probe recipe, since ~17 cases per model cannot fit one
  weight per head), two windows -> logistic. Training-row scores are cross-fitted (``crossfit_scores``):
  on random features the plain recipe dragged a stacked seg_probe from .771 to .729.
* ``lb_heads+seg``: ``lb_heads`` stacked with the seg_probe 0-5/5-9 scores; ``avg lb_heads|seg``: mean
  of the two models' probabilities.
* ``length``: log response tokens alone, the confound every attention statistic shares.

    uv run cvbench/lookback.py              # cases with lookback data only, print
    uv run cvbench/lookback.py --write      # append rows to cvbench/results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv  # noqa: E402
import whitebox as wb  # noqa: E402

WINDOWS = ((0, 5), (5, 9))  # as seg_probe: drop the final 10% (answer wrap-up)
C = 0.3


def lookback_file(model_id: str, group: str) -> Path:
    return cv.DATA / "lookback" / cv.safe(model_id) / f"{group}.npz"


def load(cases):
    """Per case: {window: per-head ratio vector [1, L_attn*H]} plus scalar features."""
    heads, scalars = [], []
    for c in tqdm(list(cases.itertuples()), desc="load lookback", leave=False):
        z = np.load(lookback_file(c.model_id, c.group))
        ratio, num = z["lb_ratio"].astype(np.float64), z["num_mass"].astype(np.float64)  # [S, L_attn, H]
        heads.append({f"lb_{a}_{b}": ratio[a:b].mean(0).reshape(1, -1) for a, b in WINDOWS})
        row = {"log_len": np.log(int(z["n_resp"]))}
        for a, b in WINDOWS:
            row[f"ratio_{a}_{b}"] = np.nanmean(ratio[a:b])
            row[f"num_{a}_{b}"] = np.nanmean(num[a:b])
        scalars.append(row)
    return heads, pd.DataFrame(scalars, index=cases.index)


def crossfit_scores(feats, models, y, groups, tr, site, inner=5, seed=0):
    """``wb.diffmean_scores`` with honest training rows: each training case is scored by a direction fitted
    on the other inner folds of ``tr``; test rows by the direction from all of ``tr``. With hundreds of
    heads and ~13 training cases per model the plain direction separates its own training cases even on
    noise, and a logistic stacked on such scores trusts them too much."""
    out = wb.diffmean_scores(feats, models, y, tr, None, site, 0.0)  # test rows: direction from all of tr
    folds = StratifiedGroupKFold(min(inner, len(np.unique(groups[tr]))), shuffle=True, random_state=seed)
    for itr, ite in folds.split(tr, y[tr], groups[tr]):
        out[tr[ite]] = wb.diffmean_scores(feats, models, y, tr[itr], None, site, 0.0)[tr[ite]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    cases = cv.load_cases()
    have = np.array([lookback_file(c.model_id, c.group).is_file() for c in cases.itertuples()])
    if not have.all():
        print(f"lookback data for {have.sum()}/{len(cases)} cases; scoring those only")
    cases = cases[have]
    y = cases.is_robust.astype(bool).to_numpy()
    models, groups = cases.model_id.to_numpy(), cases.group.to_numpy()
    heads, S = load(cases)
    Zs = {k: cv.zscore_within_model(S[cols], cases) for k, cols in {
        "length": ["log_len"],
        "lb_scalar": [f"ratio_{a}_{b}" for a, b in WINDOWS],
        "lb_scalar_num": [f"ratio_{a}_{b}" for a, b in WINDOWS] + [f"num_{a}_{b}" for a, b in WINDOWS]}.items()}
    for col in S:
        print(f"  univariate AUC {col:10s} {roc_auc_score(y, S[col]):.3f}   corr with log_len "
              f"{np.corrcoef(S[col], S.log_len)[0, 1]:+.2f}")
    acts = wb.load_acts(cases, cv)

    def lr(F, tr, te):
        return wb._predict(wb._fit_logistic(F[tr], y[tr], C), F[te], True)[1]

    names = ["length", "lb_scalar", "lb_scalar_num", "lb_heads", "seg_probe", "lb_heads+seg", "avg lb_heads|seg"]
    acc = {n: [] for n in names}
    auc = {n: [] for n in names}
    per = {n: [] for n in names}
    for r in tqdm(range(args.repeats), desc="CV repeats"):
        P = {n: np.zeros(len(y)) for n in names}
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(y, y, groups):
            H = np.concatenate([crossfit_scores(heads, models, y, groups, tr, f"lb_{a}_{b}", seed=r)
                                for a, b in WINDOWS], 1)
            Sg = np.concatenate([wb.diffmean_scores(acts, models, y, tr, None, s, 0.5)
                                 for s in ("seg_0_5", "seg_5_9")], 1)
            for n in ("length", "lb_scalar", "lb_scalar_num"):
                P[n][te] = lr(Zs[n], tr, te)
            P["lb_heads"][te] = lr(H, tr, te)
            P["seg_probe"][te] = lr(Sg, tr, te)
            P["lb_heads+seg"][te] = lr(np.c_[H, Sg], tr, te)
            P["avg lb_heads|seg"][te] = (P["lb_heads"][te] + P["seg_probe"][te]) / 2
        for n in names:
            pred = P[n] >= 0.5
            acc[n].append((pred == y).mean())
            auc[n].append(roc_auc_score(y, P[n]))
            per[n].append({m: (pred[models == m] == y[models == m]).mean() for m in np.unique(models)})
    print(f"\n{len(cases)} cases, always_true {y.mean():.4f}")
    rows = []
    for n in names:
        pm = "; ".join(f"{m.split('/')[-1]}={np.mean([d[m] for d in per[n]]):.3f}" for m in np.unique(models))
        print(f"{n:18s} acc {np.mean(acc[n]):.4f} ±{np.std(acc[n]):.4f}  auc {np.mean(auc[n]):.4f} | {pm}")
        if n.startswith(("lb", "avg")):
            rows.append({"method": "lookback", "type": "whitebox", "n_cases": len(cases),
                         "hyperparams": f"{n}; Lookback Lens ratio (sink excluded), windows 0-5/5-9; logistic C={C}",
                         "cv_score": float(np.mean(acc[n])), "cv_std": float(np.std(acc[n])),
                         "balanced_acc": float("nan"), "auc": float(np.mean(auc[n])), "per_model": pm})
    if args.write:
        cv.write_rows(rows)


if __name__ == "__main__":
    main()
