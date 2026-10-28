#!/usr/bin/env python3
"""Export the cvbench ``seg_probe`` (windows 0-5 + 5-9, layer frac 0.5, combined) as a submission artifact.

Reuses ``cvbench/whitebox.py`` so the shipped probe is the one that was
cross-validated (``cv.py --methods seg_probe --seg-combine --seg-windows 0-5 5-9
--layer-frac 0.5``): each model gets one diff-in-means direction per window,
with centre and scale from all of its cases; one shared logistic regression maps
the two z-scored projections to robust / not robust.

    uv run solutions/seg-probe/research/train.py            # all cvbench models
    uv run solutions/seg-probe/research/train.py --subset Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "cvbench"))

import cv  # noqa: E402
import whitebox  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "seg_probe.json"
SEGMENTS = 10
WINDOWS = ((0, 5), (5, 9))
FRAC = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--C", type=float, default=0.3)
    parser.add_argument("--subset", nargs="+", default=None, help="Train on these models' cases only.")
    parser.add_argument("--repeats", type=int, default=20, help="Repeats for the CV report.")
    args = parser.parse_args()

    cases = cv.load_cases()
    if args.subset:
        cases = cases[cases.model_id.isin(args.subset)]
    acts = whitebox.load_acts(cases, cv)
    y = cases.is_robust.astype(bool).to_numpy()
    models = cases.model_id.to_numpy()
    everything = np.arange(len(cases))
    sites = [f"seg_{a}_{b}" for a, b in WINDOWS]

    # CV estimate of exactly this procedure, for the record (windows and layer are fixed, so no inner CV).
    cv.register_whitebox()
    ns = argparse.Namespace(C=args.C, seg_windows=[f"{a}-{b}" for a, b in WINDOWS], seg_combine=True,
                            layer_frac=FRAC, layer_window=0, with_bb=None)
    predict, desc = cv.METHODS["seg_probe"][1](cases, ns)
    report = cv.evaluate(cases, predict, args.repeats)
    always_true = float(y.mean())
    print(f"CV ({len(cases)} cases): {report['cv_score']:.4f} +- {report['cv_std']:.4f} "
          f"(always-True {always_true:.4f}); {report['per_model']}")

    # Final fit on every training case.
    F = np.concatenate([whitebox.diffmean_scores(acts, models, y, everything, None, s, FRAC) for s in sites], 1)
    clf = whitebox._fit_logistic(F, y, args.C)
    train_acc = float(((clf.predict_proba(F)[:, 1] >= 0.5) == y).mean())

    per_model = {}
    for m in np.unique(models):
        idx = np.where(models == m)[0]
        n_hidden = acts[idx[0]][sites[0]].shape[0]
        layer = whitebox.layer_of(n_hidden, FRAC)
        y_m = y[idx]
        entry = {"n_cases": int(len(idx)), "n_robust": int(y_m.sum()), "n_hidden_states": int(n_hidden),
                 "layer": int(layer)}
        if not (y_m.all() or (~y_m).all()):
            entry["windows"] = []
            for s in sites:
                # Same arithmetic as whitebox.diffmean_scores_layer.
                H = np.stack([acts[i][s][layer] for i in idx])
                mu = H.mean(0)
                d = H[y_m].mean(0) - H[~y_m].mean(0)
                d /= np.linalg.norm(d) + 1e-8
                entry["windows"].append({"mean": mu.tolist(), "direction": d.tolist(),
                                         "scale": float(((H - mu) @ d).std() + 1e-8)})
        per_model[m] = entry
        print(f"  {m}: layer {layer}/{n_hidden - 1}, {entry['n_robust']}/{entry['n_cases']} robust, "
              f"{'directions' if 'windows' in entry else 'no direction (prior)'}")

    artifact = {
        "schema_version": 1,
        "method": desc,
        "segments": SEGMENTS,
        "windows": [list(w) for w in WINDOWS],
        "layer_frac": FRAC,
        "logistic": {"coef": [float(c) for c in clf.coef_[0]], "intercept": float(clf.intercept_[0]), "C": args.C},
        "generation": {"max_prompt_length": 2048, "max_new_tokens": 4096, "do_sample": False,
                       "cache_implementation": "static", "batch_size": 16, "chat_template_role": "user"},
        "models": per_model,
        "training": {"cases": int(len(cases)), "models": sorted(set(models)), "subset": args.subset,
                     "train_accuracy": train_acc,
                     "cv": {k: report[k] for k in ("cv_score", "cv_std", "balanced_acc", "auc", "per_model")},
                     "always_true": always_true, "source": "data/cvbench (cases.jsonl + cases_oss.jsonl, acts/, segs/)"},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB); "
          f"logistic coef {artifact['logistic']['coef']} intercept {artifact['logistic']['intercept']:.3f}; "
          f"train acc {train_acc:.3f}")


if __name__ == "__main__":
    main()
