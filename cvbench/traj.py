#!/usr/bin/env python3
"""CoT trajectory geometry as a robustness signal (autoresearch loop, 2026-10-29).

seg_probe averages the hidden state over a CoT window; this asks how the state *moves*
across the 10 cached segments (``data/cvbench/segs/``): how early it settles where it
ends, how much it wanders, whether its steps shrink. Hypothesis: a robust solve commits
early and moves smoothly; a fragile one wanders, so a perturbation tips it elsewhere.
Trajectory features are label-free, z-scored within model, and stacked with the
seg_probe 0-5/5-9 scores in the same logistic (cv.py conventions, 20x5-fold grouped CV).

    uv run cvbench/traj.py                      # seg_probe baseline vs seg_traj
    uv run cvbench/traj.py --score-sources train-main-v2
    uv run cvbench/traj.py --write              # append rows to cvbench/results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv  # noqa: E402
import whitebox as wb  # noqa: E402

FRAC = 0.5            # layer depth of the trajectory
END = 9               # last segment kept: drop the answer wrap-up (segment 9)
FEATS = ["step_mean", "norm_slope"]
WITH_PROBE = True     # stack with the seg_probe 0-5 / 5-9 scores
C = 0.3


def load_segs(cases) -> list[np.ndarray | None]:
    out = []
    for c in cases.itertuples():
        p = cv.DATA / "segs" / cv.safe(c.model_id) / f"{c.group}.npz"
        out.append(np.load(p)["seg_mean"].astype(np.float64) if p.is_file() else None)
    return out


def traj_features(seg: np.ndarray | None) -> dict:
    if seg is None:
        return {}
    H = seg[:END, wb.layer_of(seg.shape[1], FRAC)]
    end = H[-1]
    cos = H @ end / (np.linalg.norm(H, axis=1) * np.linalg.norm(end) + 1e-12)
    steps = np.linalg.norm(np.diff(H, axis=0), axis=1)
    norms = np.linalg.norm(H, axis=1)
    return {
        "conv": cos[:-1].mean(),                                   # how close to the end state it already is
        "tort": steps.sum() / (np.linalg.norm(end - H[0]) + 1e-12),  # path length / displacement
        "step_slope": np.polyfit(np.arange(len(steps)), np.log(steps + 1e-12), 1)[0],  # do steps shrink?
        "step_mean": np.log(steps + 1e-12).mean(),                 # how far it moves per segment
        "norm_slope": np.polyfit(np.arange(len(norms)), norms / norms.mean(), 1)[0],  # does the state grow?
    }


def seg_traj(cases, args):
    T = pd.DataFrame([traj_features(s) for s in load_segs(cases)], index=cases.index)[FEATS]
    Z = cv.zscore_within_model(T, cases)
    if not WITH_PROBE:
        return Z, None
    acts = wb.load_acts(cases, cv)
    y = cases.is_robust.astype(bool).to_numpy()
    models = cases.model_id.to_numpy()

    def predict(tr, te):
        F = np.concatenate([wb.diffmean_scores(acts, models, y, tr, None, s, 0.5) for s in ("seg_0_5", "seg_5_9")]
                           + [Z], 1)
        return wb._predict(wb._fit_logistic(F[tr], y[tr], C), F[te], True)
    return Z, predict


def describe() -> str:
    return (f"CoT trajectory geometry {'+'.join(FEATS)} at layer frac {FRAC}, segments 0-{END - 1}"
            f"{' + seg_probe 0-5/5-9' if WITH_PROBE else ''}; logistic C={C}")


def seg_probe_baseline(cases, args):
    acts = wb.load_acts(cases, cv)
    y = cases.is_robust.astype(bool).to_numpy()
    models = cases.model_id.to_numpy()

    def predict(tr, te):
        F = np.concatenate([wb.diffmean_scores(acts, models, y, tr, None, s, 0.5) for s in ("seg_0_5", "seg_5_9")], 1)
        return wb._predict(wb._fit_logistic(F[tr], y[tr], C), F[te], True)
    return predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--score-sources", nargs="+", default=None)
    ap.add_argument("--baseline", action="store_true", help="Also run the seg_probe 0-5/5-9 baseline.")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    cases = cv.load_cases()
    y = cases.is_robust.astype(bool).to_numpy()
    mask = cases.source.isin(args.score_sources).to_numpy() if args.score_sources else None
    rows = []
    if args.baseline:
        res = cv.evaluate(cases, seg_probe_baseline(cases, args), args.repeats, score_mask=mask)
        print(f"seg_probe  acc={res['cv_score']:.4f} auc={res['auc']:.4f}")
    Z, predict = seg_traj(cases, args)
    res = cv.evaluate(cases, predict or cv.logistic_predictor(Z, y, C), args.repeats, score_mask=mask)
    print(f"seg_traj   acc={res['cv_score']:.4f} auc={res['auc']:.4f} bacc={res['balanced_acc']:.4f}  {res['per_model']}")
    print(f"METRIC {res['cv_score']:.4f}\nAUC {res['auc']:.4f}")
    hyper = describe()
    if args.score_sources:
        hyper += f"; SCORED ON sources={','.join(args.score_sources)} ({int(mask.sum())} cases)"
    rows.append({"method": "seg_traj", "type": "whitebox", "hyperparams": hyper, "n_cases": len(cases), **res})
    if args.write:
        cv.write_rows(rows)


if __name__ == "__main__":
    main()
