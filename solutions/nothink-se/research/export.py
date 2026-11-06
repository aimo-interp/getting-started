#!/usr/bin/env python3
"""Train the no-reasoning semantic-entropy model and write artifacts/model.json.

Features from ``data/cvbench/nothink/<model>__direct.jsonl`` (``cvbench/generate.py --stages nothink``),
z-scored within model, one pooled logistic regression (C=0.1, as cvbench/stack.py). Per model, "nt" vs
"prior" (always robust) is chosen by out-of-fold accuracy (5-fold grouped by problem, 10 seeds).

    pyine/bin/python solutions/nothink-se/research/export.py [--features se top_share ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "cvbench"))
import cv  # noqa: E402
import stack  # noqa: E402
import whitebox as wb  # noqa: E402

N, MAX_NEW = 50, 16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", default=stack.NT_FEATS)
    ap.add_argument("--no-gate", action="store_true", help="every model uses nt")
    args = ap.parse_args()
    cases = cv.load_cases()
    y = cases.is_robust.astype(bool).to_numpy()
    models, groups = cases.model_id.to_numpy(), cases.group.to_numpy()
    X = stack.nt_feats(cases, feats=args.features).astype(float)
    print(f"{len(cases)} cases, {int(X.notna().all(1).sum())} with features")
    Xf = X.fillna(X.median())
    entries = {}
    for m in np.unique(models):
        b = Xf[models == m]
        sd = b.std(ddof=0).replace(0, 1)
        entries[m] = {"mean": b.mean().tolist(), "sd": sd.tolist(), "median": X[models == m].median().tolist()}
    Z = np.asarray(cv.zscore_within_model(X, cases), dtype=float)

    seeds, acc = 10, np.zeros(len(y))
    for seed in range(seeds):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(y, y, groups):
            p = wb._predict(wb._fit_logistic(Z[tr], y[tr], stack.C), Z[te], True)[1]
            acc[te] += ((p >= 0.5) == y[te]) / seeds
    for m in entries:
        idx = models == m
        nt, prior = acc[idx].mean(), y[idx].mean()
        entries[m]["option"] = "nt" if args.no_gate or nt > prior else "prior"
        print(f"  {m:40s} nt {nt:.3f}  prior {prior:.3f}  -> {entries[m]['option']}")
    clf = wb._fit_logistic(Z, y, stack.C)
    coef, icpt = (clf.coef_[0], clf.intercept_[0]) if hasattr(clf, "coef_") else clf
    art = {"features": list(args.features), "n": N, "max_new": MAX_NEW, "models": entries,
           "logistic": {"coef": np.asarray(coef).tolist(), "intercept": float(icpt)}}
    (HERE / "artifacts" / "model.json").write_text(json.dumps(art, indent=1))
    print("coef", dict(zip(args.features, np.round(coef, 3))), "intercept", round(float(icpt), 3))


if __name__ == "__main__":
    main()
