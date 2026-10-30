#!/usr/bin/env python3
"""Effort probe: predict the sampled response length from the early chain of thought (loop 2026-10-31).

The mean log-length of a model's sampled responses is the strongest single robustness signal (long
searches go with fragility) but costs 8 x 16k tokens at test time. This probe predicts it from the
hidden state of the greedy trace's first segments (``segs/``: 10 segments over the 4k trace), which the
seg_probe pass already computes. Per model, a ridge from the activation at one layer to the target,
fitted on the training folds; the prediction is the feature (z-scored within model, as every block).

Prototype target: the mean log tokens of the 8 cached 16k samples of each cvbench case (``tx`` length).
Later: the same target on many unlabelled problems (``generate.py`` + sampling), which needs no labels.

    pyine/bin/python cvbench/effort.py            # out-of-fold fit quality per site / layer / alpha
    pyine/bin/python cvbench/stack.py --blocks seg traj effort
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv  # noqa: E402
import whitebox as wb  # noqa: E402

SITE = "seg_0_2"      # mean over the first two of 10 segments (the first ~800 tokens of a 4k trace)
FRAC = 0.5
ALPHA = 1e2
WINDOWS = {"seg_0_1": (0, 1), "seg_0_2": (0, 2), "seg_0_3": (0, 3), "seg_0_5": (0, 5), "seg_0_9": (0, 9)}


def load_sites(cases) -> list[dict]:
    """Per case {site: [L+1, d]} for every window in WINDOWS, from segs/ (resp_mean when too short)."""
    acts = wb.load_acts(cases, cv)
    out = []
    for c, a in zip(cases.itertuples(), acts):
        p = cv.DATA / "segs" / cv.safe(c.model_id) / f"{c.group}.npz"
        seg = np.load(p)["seg_mean"].astype(np.float32) if p.is_file() else None
        row = {}
        for name, (lo, hi) in WINDOWS.items():
            row[name] = seg[lo:hi].mean(0) if seg is not None else a["resp_mean"]
        row["last_prompt"] = a["last_prompt"]
        out.append(row)
    return out


def target_length(cases) -> np.ndarray:
    """Mean log tokens of the 8 cached 16k samples (NaN when a case has none)."""
    sys.path.insert(0, str(cv.ROOT / "cvbench"))
    import stack
    return stack.tx_feats(cases)["length"].to_numpy(float)


def ridge_scores(acts, models, t, tr, site, frac, alpha) -> np.ndarray:
    """Per model: ridge(activation @ layer -> t) fitted on the training rows, predicting every row.
    Rows with a NaN target are left out of the fit."""
    feat = np.zeros(len(acts))
    for m in np.unique(models):
        idx = np.where(models == m)[0]
        tr_m = np.intersect1d(idx, tr)
        tr_m = tr_m[np.isfinite(t[tr_m])]
        L = acts[idx[0]][site].shape[0]
        layer = wb.layer_of(L, frac)
        H = np.stack([acts[i][site][layer] for i in idx]).astype(np.float64)
        if len(tr_m) < 4:
            continue
        H_tr = np.stack([acts[i][site][layer] for i in tr_m]).astype(np.float64)
        mu, sd = H_tr.mean(0), H_tr.std(0) + 1e-6
        r = Ridge(alpha=alpha).fit((H_tr - mu) / sd, t[tr_m])
        feat[idx] = r.predict((H - mu) / sd)
    return feat[:, None]


# ---------------------------------------------------------------- unlabelled training set (generate.py effort)
SITE2, FRAC2, ALPHA2 = "seg_0_9", 0.5, 1e4
TARGET2 = "log_len"   # or cap_rate
_U: dict = {}


def _norm(t: str) -> str:
    import re
    return re.sub(r"\s+", " ", t).strip().lower()


def load_unlabelled(model_id: str, site: str, frac: float) -> dict:
    """Effort-stage rows of one model at one site/layer: X [n, d], targets, problem ids."""
    key = (model_id, site, frac)
    if key not in _U:
        d = cv.DATA / "effort" / cv.safe(model_id)
        files = sorted(d.glob("*.npz")) if d.is_dir() else []
        X, ll, cr, pid = [], [], [], []
        for f in files:
            z = np.load(f)
            fi = int(np.argmin(np.abs(z["fracs"] - frac)))
            X.append(z[site][fi].astype(np.float32)), ll.append(float(z["log_len"])), cr.append(float(z["cap_rate"]))
            pid.append(str(z["problem_id"]))
        _U[key] = {"X": np.stack(X) if X else np.zeros((0, 1)), "log_len": np.array(ll), "cap_rate": np.array(cr),
                   "pid": np.array(pid)}
    return _U[key]


def case_pids(cases) -> np.ndarray:
    pids = {_norm(r["problem"]): r["problem_id"] for r in cv.read_jsonl(cv.ROOT / "data" / "labelgen" / "problems.jsonl")}
    return np.array([pids.get(_norm(p), "") for p in cases.problem])


def unlabelled_scores(cases, acts, tr, site=SITE2, frac=FRAC2, alpha=ALPHA2, target=TARGET2) -> np.ndarray:
    """Per model: ridge(activation -> sampled length) trained on that model's effort-stage prompts whose
    problem is not a test-fold problem, predicting every cvbench case. Models without data score 0."""
    models = cases.model_id.to_numpy()
    pids = case_pids(cases)
    held_out = set(pids[np.setdiff1d(np.arange(len(cases)), tr)]) - {""}
    feat = np.zeros(len(cases))
    for m in np.unique(models):
        u = load_unlabelled(m, site, frac)
        keep = np.array([p not in held_out for p in u["pid"]], bool)
        if keep.sum() < 20:
            continue
        idx = np.where(models == m)[0]
        L = acts[idx[0]][site].shape[0]
        layer = wb.layer_of(L, frac)
        Xu, tu = u["X"][keep].astype(np.float64), u[target][keep]
        mu, sd = Xu.mean(0), Xu.std(0) + 1e-6
        r = Ridge(alpha=alpha).fit((Xu - mu) / sd, tu)
        H = np.stack([acts[i][site][layer] for i in idx]).astype(np.float64)
        feat[idx] = r.predict((H - mu) / sd)
    return feat[:, None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", default=list(WINDOWS) + ["last_prompt"])
    ap.add_argument("--fracs", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    ap.add_argument("--alphas", nargs="+", type=float, default=[1e2, 1e3, 1e4])
    ap.add_argument("--unlabelled", action="store_true", help="Probe trained on the effort-stage prompts instead.")
    ap.add_argument("--target", default="log_len", choices=["log_len", "cap_rate"])
    args = ap.parse_args()
    cases = cv.load_cases()
    y = cases.is_robust.astype(bool).to_numpy()
    models, groups = cases.model_id.to_numpy(), cases.group.to_numpy()
    acts = load_sites(cases)
    t = target_length(cases)
    print(f"target: AUC of true sampled log-length vs robust = {roc_auc_score(y, -t):.3f} (sign flipped: long = fragile)")
    for site in args.sites:
        for frac in args.fracs:
            for alpha in args.alphas:
                oof = np.zeros(len(t))
                for tr, te in GroupKFold(5).split(t, groups=groups):
                    oof[te] = (unlabelled_scores(cases, acts, tr, site, frac, alpha, args.target)
                               if args.unlabelled else ridge_scores(acts, models, t, tr, site, frac, alpha))[te, 0]
                ok = np.isfinite(t)
                # within-model correlation, pooled over models (z-score within model first)
                z = lambda v: np.concatenate([(v[models == m] - v[models == m].mean()) / (v[models == m].std() + 1e-9)
                                              for m in np.unique(models)])
                r = np.corrcoef(z(oof)[z(t) == z(t)], z(t)[z(t) == z(t)])[0, 1] if ok.all() else float("nan")
                print(f"{site:12s} frac {frac:.1f} alpha {alpha:7.0f}  oof corr {r:+.3f}  AUC vs robust {roc_auc_score(y, -oof):.3f}")


if __name__ == "__main__":
    main()
