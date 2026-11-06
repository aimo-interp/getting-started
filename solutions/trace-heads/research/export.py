#!/usr/bin/env python3
"""Train every trace-heads method on the cvbench cases and package it as its own Codabench solution.

One shared runtime (``trace_engine``), one head per method, every head trained on the same 118
cases (``data/cvbench/cases.jsonl`` + ``cases_oss.jsonl``: four small models and gpt-oss-120b
per effort) and cross-validated with the same protocol (``cv.evaluate``: 5-fold
StratifiedGroupKFold by problem, 20 repeats). The CV score reported for each package is the
exact cvbench run of the procedure the package ships. For each package this writes
``solutions/<name>/{solution.py, README.md, artifacts/head.json}`` and copies in
``trace_engine/`` and ``uncertainty_profile/`` (the zip must be self-contained).

    uv run solutions/trace-heads/research/export.py                 # every package
    uv run solutions/trace-heads/research/export.py --only coe saplma
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "cvbench"), str(HERE)]

import cv  # noqa: E402
import traj  # noqa: E402
import whitebox  # noqa: E402
from trace_engine import heads  # noqa: E402

C = 0.3
FRAC = 0.5
GENERATION = {"max_prompt_length": 2048, "max_new_tokens": 4096, "do_sample": False,
              "cache_implementation": "static", "batch_size": 16, "chat_template_role": "user"}

# name -> head config; ``cv`` is the cvbench method + options whose run the package reproduces.
PACKAGES = {
    "length-probe": {"head": "length", "title": "Greedy response length", "family": "black-box (length)",
                     "cv": ("length", {})},
    "cot-text": {"head": "cot_text", "title": "CoT text statistics", "family": "black-box (text)",
                 "cv": ("cot_text", {})},
    "uncertainty-lr": {"head": "uncertainty", "title": "Uncertainty profile, all models",
                       "family": "grey-box (token probabilities)", "cv": ("uncertainty_cv", {})},
    "act-probe": {"head": "diffmean", "sites": ["resp_mean"], "title": "Response-mean diff-in-means probe",
                  "family": "white-box (activations)", "cv": ("act_probe", {"site": "resp_mean", "layer_frac": FRAC})},
    "saplma": {"head": "saplma", "site": "resp_mean", "title": "SAPLMA (MLP on activations)",
               "family": "white-box (activations)",
               "cv": ("saplma", {"saplma_target": "robust", "site": "resp_mean", "layer_frac": FRAC})},
    "coe": {"head": "coe", "site": "seg_0_5", "features": ["coe_ang"], "title": "Chain-of-Embedding (layer angle)",
            "family": "white-box (activation geometry, label-free features)",
            "cv": ("coe", {"coe_feats": ["coe_ang"], "coe_site": "seg_0_5"})},
    "seg-traj": {"head": "seg_traj", "sites": ["seg_0_5", "seg_5_9"], "features": list(traj.FEATS),
                 "title": "CoT trajectory geometry + segment probe",
                 "family": "white-box (activation trajectory over the CoT; label-free features + probe)",
                 "cv": ("seg_traj", {})},
    "introspect": {"head": "introspect", "features": ["robust", "correct", "long"],
                   "title": "Introspection (zero-shot self-prediction)", "family": "black-box (self-report, no CoT)",
                   "cv": ("introspect", {})},
}
NEEDS_HIDDEN = {"diffmean", "saplma", "coe", "seg_traj"}


def cv_args(**over) -> argparse.Namespace:
    base = dict(C=C, repeats=20, layer_frac=None, layer_window=0, site=None, with_bb=None, seg_windows=None,
                seg_combine=False, saplma_target="robust", saplma_alpha=1e-2, len_with_seg=False,
                coe_feats=["coe_r", "coe_c"], coe_site="resp_mean", n=4, budget=2048)
    return argparse.Namespace(**{**base, **over})


def run_cv(cases, method: str, opts: dict) -> tuple[dict, pd.DataFrame | None, str]:
    """The cvbench run of ``method`` (same branches as cv.main); also returns its feature table, if any."""
    y = cases.is_robust.astype(bool).to_numpy()
    feats, desc = cv.METHODS[method][1](cases, cv_args(**opts))
    if callable(feats):
        return cv.evaluate(cases, feats, 20), None, desc
    X = cv.zscore_within_model(feats, cases)
    return cv.evaluate(cases, cv.logistic_predictor(X, y, C), 20), feats, desc


def zscore_entries(feats: pd.DataFrame, cases) -> tuple[dict, np.ndarray]:
    """Per-model statistics reproducing cv.zscore_within_model (global median fill, population sd, 0 -> 1)."""
    X = feats.astype(float)
    median = X.median()
    X = X.fillna(median)
    entries = {}
    for m, idx in cases.groupby("model_id").groups.items():
        block = X.loc[idx]
        if block.isna().all().all():
            continue
        sd = block.std(ddof=0).replace(0, 1)
        entries[m] = {"n_cases": int(len(idx)), "median": median.fillna(0).tolist(), "mean": block.mean().tolist(),
                      "sd": sd.fillna(1).tolist()}
    return entries, cv.zscore_within_model(feats, cases)


def diffmean_entries(cases, acts, y, sites) -> tuple[dict, np.ndarray]:
    models = cases.model_id.to_numpy()
    F = np.concatenate([whitebox.diffmean_scores(acts, models, y, np.arange(len(cases)), None, s, FRAC)
                        for s in sites], 1)
    entries = {}
    for m in np.unique(models):
        idx = np.where(models == m)[0]
        y_m = y[idx]
        if y_m.all() or (~y_m).all():
            continue  # no direction: predict robust
        n_hidden = acts[idx[0]][sites[0]].shape[0]
        layer = whitebox.layer_of(n_hidden, FRAC)
        windows = []
        for s in sites:  # same arithmetic as whitebox.diffmean_scores_layer on every case
            H = np.stack([acts[i][s][layer] for i in idx])
            mu = H.mean(0)
            d = H[y_m].mean(0) - H[~y_m].mean(0)
            d /= np.linalg.norm(d) + 1e-8
            windows.append({"site": s, "mean": mu.tolist(), "direction": d.tolist(),
                            "scale": float(((H - mu) @ d).std() + 1e-8)})
        entries[m] = {"n_cases": int(len(idx)), "n_hidden_states": int(n_hidden), "layer": int(layer), "windows": windows}
    return entries, F


def seg_traj_entries(cases, acts, y, sites) -> tuple[dict, np.ndarray]:
    """diffmean entries for the probe windows, plus each model's trajectory z-score statistics."""
    entries, F = diffmean_entries(cases, acts, y, sites)
    T = pd.DataFrame([traj.traj_features(s) for s in traj.load_segs(cases)], index=cases.index)[traj.FEATS]
    stats, Z = zscore_entries(T, cases)
    for m, e in entries.items():
        e["traj"] = {"end": traj.END, **{k: stats[m][k] for k in ("median", "mean", "sd")}}
    return entries, np.concatenate([F, Z], 1)


def saplma_entries(cases, acts, y, site) -> tuple[dict, np.ndarray]:
    """whitebox.saplma (target robust) fitted on every case: one MLP per model, its in-sample logit as feature."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    models = cases.model_id.to_numpy()
    F = np.zeros((len(cases), 1))
    entries = {}
    for m in np.unique(models):
        idx = np.where(models == m)[0]
        if len(set(y[idx])) < 2:
            continue
        n_hidden = acts[idx[0]][site].shape[0]
        layer = whitebox.layer_of(n_hidden, FRAC)
        H = np.stack([acts[i][site][layer] for i in idx])
        clf = make_pipeline(StandardScaler(), MLPClassifier((256, 128, 64), alpha=1e-2, max_iter=300,
                                                            random_state=0)).fit(H, y[idx])
        p = clf.predict_proba(H)[:, 1].clip(1e-4, 1 - 1e-4)
        F[idx, 0] = np.log(p / (1 - p))
        scaler, net = clf[0], clf[1]
        mlp = {"scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
               "coefs": [w.tolist() for w in net.coefs_], "intercepts": [b.tolist() for b in net.intercepts_]}
        check = heads.mlp_logit(mlp, H[0].astype(np.float64))
        assert abs(check - F[idx[0], 0]) < 1e-3, (m, check, F[idx[0], 0])
        entries[m] = {"n_cases": int(len(idx)), "n_hidden_states": int(n_hidden), "layer": int(layer), "mlp": mlp}
    return entries, F


def write_package(name: str, spec: dict, art: dict) -> None:
    out = ROOT / "solutions" / name
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    (out / "artifacts" / "head.json").write_text(json.dumps(art))
    for pkg, src in (("trace_engine", HERE / "trace_engine"),
                     ("uncertainty_profile", ROOT / "solutions" / "uncertainty-profiling" / "uncertainty_profile")):
        shutil.rmtree(out / pkg, ignore_errors=True)
        shutil.copytree(src, out / pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (out / "solution.py").write_text(
        f'"""Codabench entry point: {spec["title"]} (cvbench `{spec["cv"][0]}`) on the shared trace-heads engine.\n\n'
        f'Generated by ``solutions/trace-heads/research/export.py``; see README.md.\n"""\n\n'
        "from pathlib import Path\n\nfrom trace_engine.engine import predict_robustness\n\n"
        'ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "head.json"\n\n\n'
        "def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:\n"
        '    """Predict one native boolean robustness label per problem, in input order."""\n'
        "    return predict_robustness(ARTIFACT, model_id, list(problems), reasoning_effort)\n")
    t = art["training"]
    per = "\n".join(f"| {m.split('=')[0]} | {m.split('=')[1]} |" for m in t["cv"]["per_model"].split("; "))
    missing = sorted(set(t["models"]) - set(art["models"]))
    pipeline = ("three prompt-only forward passes per problem (no generation): the Yes-vs-No next-token "
                "log-odds for 'answer unchanged under rewording', 'answer correct', 'solution over 4k tokens'"
                if art["head"] == "introspect" else
                "greedy CoT with the uncertainty-profiling generator (user-only template, 2,048 prompt / 4,096 new "
                "tokens, static cache, batches of 16)"
                + (", then one forward pass over prompt + response for the hidden states" if art["needs_hidden"] else ""))
    (out / "README.md").write_text(f"""# {spec['title']} (`{name}`)

Family: {spec['family']}. One of the trace-heads packages: every one shares the runtime in
`trace_engine/` and is trained on the same 118 cvbench cases with the same CV protocol, so
their scores compare on even ground. Generated by `solutions/trace-heads/research/export.py`.

**Pipeline:** {pipeline}; then the head: {art['method']}.

gpt-oss-120b is keyed by reasoning effort (`"default"` = medium). Fallbacks predict robust
(True): a model or effort without an entry, a failed load or batch, or a batch that would
overrun the run's time budget (`TRACE_BUDGET_S`, default 3,300 s from import).

## CV (cvbench, {t['cases']} cases, 5-fold grouped by problem, 20 repeats)

| | Accuracy | AUC |
|---|---|---|
| {name} | {t['cv']['cv_score']:.3f} ± {t['cv']['cv_std']:.3f} | {t['cv']['auc']:.3f} |
| always-True | {t['always_true']:.3f} | 0.500 |

| Model | CV accuracy |
|---|---|
{per}

{"Without an entry (predicted robust): " + ", ".join(missing) + "." if missing else "Every training model has an entry."}
Train accuracy (final fit, all cases): {t['train_accuracy']:.3f}.
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", default=None, choices=list(PACKAGES))
    args = parser.parse_args()
    cv.register_whitebox()
    cv.METHODS["seg_traj"] = ("whitebox", lambda cases, a: (traj.seg_traj(cases, a)[1], traj.describe()))
    cases = cv.load_cases()
    y = cases.is_robust.astype(bool).to_numpy()
    acts = None
    for name, spec in PACKAGES.items():
        if args.only and name not in args.only:
            continue
        method, opts = spec["cv"]
        report, feats, desc = run_cv(cases, method, opts)
        head = spec["head"]
        if head == "diffmean":
            acts = acts or whitebox.load_acts(cases, cv)
            entries, F = diffmean_entries(cases, acts, y, spec["sites"])
        elif head == "seg_traj":
            assert traj.FRAC == FRAC, "trajectory and probe share the head's layer"
            acts = acts or whitebox.load_acts(cases, cv)
            entries, F = seg_traj_entries(cases, acts, y, spec["sites"])
        elif head == "saplma":
            acts = acts or whitebox.load_acts(cases, cv)
            entries, F = saplma_entries(cases, acts, y, spec["site"])
        else:
            entries, F = zscore_entries(feats, cases)
        clf = whitebox._fit_logistic(F, y, C)
        train_acc = float(((clf.predict_proba(F)[:, 1] >= 0.5) == y).mean())
        art = {"schema_version": 1, "head": head, "method": desc, "zscore": head not in ("diffmean", "saplma", "seg_traj"),
               "needs_hidden": head in NEEDS_HIDDEN, "segments": 10, "generation": GENERATION,
               "site": spec.get("site"), "features": spec.get("features"),
               "logistic": {"coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]), "C": C},
               "models": entries,
               "training": {"cases": int(len(cases)), "models": sorted(set(cases.model_id)), "train_accuracy": train_acc,
                            "always_true": float(y.mean()),
                            "cv": {k: report[k] for k in ("cv_score", "cv_std", "balanced_acc", "auc", "per_model")},
                            "source": "data/cvbench (cases.jsonl + cases_oss.jsonl)"}}
        write_package(name, spec, art)
        print(f"{name:15s} cv {report['cv_score']:.3f}±{report['cv_std']:.3f} auc {report['auc']:.3f} "
              f"train {train_acc:.3f} | {len(entries)}/{len(set(cases.model_id))} models | {report['per_model']}",
              flush=True)


if __name__ == "__main__":
    main()
