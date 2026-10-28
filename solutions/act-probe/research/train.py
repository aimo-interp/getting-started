#!/usr/bin/env python3
"""Export the cvbench ``act_probe`` (site resp_mean, nested layer frac) as a submission artifact.

Reuses ``cvbench/whitebox.py`` so the shipped probe is the one that was
cross-validated: the layer frac is picked by the same grouped inner CV
(``_inner_select``), now over every cached case; each model's direction, centre
and scale come from all of its cases; one shared logistic regression maps the
z-scored projection to robust / not robust.

    uv run solutions/act-probe/research/train.py            # all cvbench models
    uv run solutions/act-probe/research/train.py --subset Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "cvbench"))

import cv  # noqa: E402
import whitebox  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "act_probe.json"
SITE = "resp_mean"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--C", type=float, default=0.3)
    parser.add_argument("--subset", nargs="+", default=None, help="Train on these models' cases only.")
    parser.add_argument("--select-on", nargs="+", default=None,
                        help="Pick the layer frac on these models' cases only (default: the training cases).")
    parser.add_argument("--repeats", type=int, default=20, help="Outer repeats for the nested-CV report.")
    args = parser.parse_args()

    cases = cv.load_cases()
    if args.subset:
        cases = cases[cases.model_id.isin(args.subset)]
    acts = whitebox.load_acts(cases, cv)
    y = cases.is_robust.astype(bool).to_numpy()
    models = cases.model_id.to_numpy()
    groups = cases.group.to_numpy()
    everything = np.arange(len(cases))

    # Nested-CV estimate of exactly this procedure, for the record.
    cv.register_whitebox()
    ns = argparse.Namespace(C=args.C, site=SITE, layer_frac=None, layer_window=0, with_bb=None)
    predict, desc = cv.METHODS["act_probe"][1](cases, ns)
    chosen = next(c.cell_contents for c in predict.__closure__
                  if isinstance(c.cell_contents, list) and not c.cell_contents)
    report = cv.evaluate(cases, predict, args.repeats)
    always_true = float(y.mean())
    print(f"nested CV ({len(cases)} cases): {report['cv_score']:.4f} +- {report['cv_std']:.4f} "
          f"(always-True {always_true:.4f}); {report['per_model']}")
    print("outer-fold layer fracs:", Counter(f for _, f in chosen).most_common())

    # Final selection and fit on every training case.
    cands = [(SITE, f) for f in whitebox.LAYER_FRACS]
    make = lambda cand, rows: whitebox.diffmean_scores(acts, models, y, rows, None, *cand)
    sel_rows = everything if not args.select_on else np.where(np.isin(models, args.select_on))[0]
    _, frac = whitebox._inner_select(cands, make, y, groups, sel_rows, args.C)
    print(f"selected layer frac: {frac}")

    F = make((SITE, frac), everything)
    clf = whitebox._fit_logistic(F, y, args.C)
    train_acc = float(((clf.predict_proba(F)[:, 1] >= 0.5) == y).mean())

    per_model = {}
    for m in np.unique(models):
        idx = np.where(models == m)[0]
        n_hidden = acts[idx[0]][SITE].shape[0]
        layer = whitebox.layer_of(n_hidden, frac)
        H = np.stack([acts[i][SITE][layer] for i in idx])
        y_m = y[idx]
        entry = {"n_cases": int(len(idx)), "n_robust": int(y_m.sum()), "n_hidden_states": int(n_hidden),
                 "layer": int(layer)}
        if not (y_m.all() or (~y_m).all()):
            # Same arithmetic as whitebox.diffmean_scores_layer.
            mu = H.mean(0)
            d = H[y_m].mean(0) - H[~y_m].mean(0)
            d /= np.linalg.norm(d) + 1e-8
            entry.update(mean=mu.tolist(), direction=d.tolist(), scale=float(((H - mu) @ d).std() + 1e-8))
        per_model[m] = entry
        print(f"  {m}: layer {layer}/{n_hidden - 1}, {entry['n_robust']}/{entry['n_cases']} robust, "
              f"{'direction' if 'direction' in entry else 'no direction (prior)'}")

    artifact = {
        "schema_version": 1,
        "method": desc,
        "site": SITE,
        "layer_frac": frac,
        "logistic": {"coef": float(clf.coef_[0, 0]), "intercept": float(clf.intercept_[0]), "C": args.C},
        "generation": {"max_prompt_length": 2048, "max_new_tokens": 4096, "do_sample": False,
                       "cache_implementation": "static", "batch_size": 16, "chat_template_role": "user"},
        "models": per_model,
        "training": {"cases": int(len(cases)), "models": sorted(set(models)), "subset": args.subset,
                     "select_on": args.select_on, "train_accuracy": train_acc,
                     "nested_cv": {k: report[k] for k in ("cv_score", "cv_std", "balanced_acc", "auc", "per_model")},
                     "always_true": always_true, "source": "data/cvbench (cases.jsonl, acts/)"},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB); "
          f"logistic coef {artifact['logistic']['coef']:.3f} intercept {artifact['logistic']['intercept']:.3f}; "
          f"train acc {train_acc:.3f}")


if __name__ == "__main__":
    main()
