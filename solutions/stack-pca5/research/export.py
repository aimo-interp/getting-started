#!/usr/bin/env python3
"""Train the per-model stack (seg probe + CoT trajectory + transcript features) and package it.

Blocks, each the one cvbench cross-validated (``cvbench/stack.py``):

* ``seg``: diff-in-means probe scores on the greedy 4k trace's segments 0-5 and 5-9 (layer frac 0.5), each window
  zeroed for models where its inner cross-validated AUC is below 0.6;
* ``traj``: trajectory geometry (step_mean, norm_slope) of the same trace;
* ``tx``: transcript features of 8 sampled traces (pre_answer_logchars, anchor, agree, length), one fit per
  thinking budget (2k..16k) since the leaderboard cannot always afford 16k.

Options are block subsets. Per model the option is chosen by inner cross-validation on that model's cases
(``stack.run_gated``, seg+traj+tx vs traj+tx: the seg probe hurts DeepSeek), and one logistic regression per
option and budget is fitted on all 118 cases. Everything is written to ``artifacts/stack.json``; the runtime
(``stack_probe/engine.py``) decodes the greedy trace and the 8 samples in one batch.

    pyine/bin/python solutions/stack-probe/research/export.py
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "cvbench"), str(ROOT / "solutions" / "trace-heads")]
import cv  # noqa: E402
import stack  # noqa: E402
import traj  # noqa: E402
import whitebox as wb  # noqa: E402

spec = importlib.util.spec_from_file_location("th_export", ROOT / "solutions" / "trace-heads" / "research" / "export.py")
th = importlib.util.module_from_spec(spec)
spec.loader.exec_module(th)

C = stack.C
FRAC = 0.5
SEG_SITES = ["seg_0_5", "seg_5_9"]
BUDGETS = [2048, 4096, 8192, 12288, 16384]
N_SAMPLES = 8
TX_FEATS = ["pre_answer_logchars", "anchor", "agree", "length"]
OPTIONS = {"seg+traj+tx": ["seg", "traj", "tx"], "traj+tx": ["traj", "tx"], "seg+traj": ["seg", "traj"],
           "tx": ["tx"], "traj": ["traj"], "seg": ["seg"]}
DS = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
PCS = 5
import os
PCA_K, PCA_C = 3, 0.1
SEG_KIND = os.environ.get("SEG_KIND", "pca")      # pca | wdim | rpca | upca (2026-11-03 variants)
SEG_C = float(os.environ.get("SEG_C", PCA_C))
OUT_DIR = Path(os.environ["OUT_DIR"]) if os.environ.get("OUT_DIR") else None
REP = int(os.environ.get("REPORT_REPEATS", 20))   # FAST builds: 1 (README CV only)
SEEDS = int(os.environ.get("OPTION_SEEDS", 10))
GATE = ["seg+traj+tx", "traj+tx"]          # the per-model choice (first wins ties)
FALLBACK = list(OPTIONS)                   # runtime preference when a block is unavailable


def choose_options(cases, blocks: dict, y, seeds=None) -> dict:
    seeds = seeds or SEEDS
    """Per model: the GATE option with the best out-of-fold accuracy on that model's cases (all cases,
    5-fold grouped by problem, averaged over seeds), as stack.run_gated does inside each outer fold.
    ``blocks``: name -> callable(tr) -> [n, k] features fitted on the training rows (the seg probe)."""
    models, groups = cases.model_id.to_numpy(), cases.group.to_numpy()
    acc = {k: np.zeros(len(y)) for k in GATE}
    for seed in range(seeds):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(y, y, groups):
            for k in GATE:
                F = np.concatenate([blocks[b](tr) for b in OPTIONS[k]], 1)
                p = wb._predict(wb._fit_logistic(F[tr], y[tr], C), F[te], True)[1]
                acc[k][te] += ((p >= 0.5) == y[te]) / seeds
    out = {}
    for m in np.unique(models):
        idx = models == m
        out[m] = max(GATE, key=lambda k: (acc[k][idx].mean(), k == GATE[0]))
        print(f"  {m:40s} " + "  ".join(f"{k} {acc[k][idx].mean():.3f}" for k in GATE) + f"  -> {out[m]}")
    return out


def main() -> None:
    cases = cv.load_cases()
    y = cases.is_robust.astype(bool).to_numpy()
    models = cases.model_id.to_numpy()
    acts = wb.load_acts(cases, cv)
    print(f"{len(cases)} cases, {len(set(models))} models")

    # blocks on all cases. seg windows are gated per model by their inner out-of-fold AUC (stack.safe_seg_scores):
    # a window whose cross-validated AUC on that model is below SAFE_MIN_AUC scores 0 (an inverted direction on
    # few fragile cases otherwise misleads the shared logistic).
    seg_entries, F_seg = th.diffmean_entries(cases, acts, y, SEG_SITES)
    groups = cases.group.to_numpy()
    all_rows = np.arange(len(y))
    # stack-pca5 (2026-11-03): every model's seg window direction is a PCA(3) + L2 logistic (C=0.1) probe
    # instead of diff-in-means (cvbench stack.py segp_pca3_0.1: CV .816 at .5, .839 with per-model thresholds).
    # Rows are centred and divided by their RMS norm sd, PCA(5) on them, logistic on the 5 coordinates; the score
    # w . P((h - mu) / sd) is linear in h, so the engine's window format holds it: direction = P^T w / sd.
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge
    drop = cases.max_drop.to_numpy(float)
    stack._DROP = drop
    for m, e in seg_entries.items():
        idx = np.where(models == m)[0]
        ym = y[idx]
        for j, s in enumerate(SEG_SITES):
            H = np.stack([acts[i][s][e["layer"]] for i in idx]).astype(np.float64)
            mu = H.mean(0)
            U = stack.unlabelled(m, s, FRAC) if SEG_KIND == "upca" else None
            if SEG_KIND == "upca" and U is not None:
                Uo = np.stack(list(U.values())); umu = Uo.mean(0)
                sd = np.sqrt(((Uo - umu) ** 2).sum(1).mean()) + 1e-8
                P = PCA(PCA_K, random_state=0).fit((Uo - umu) / sd)
                Z = P.transform((H - umu) / sd)
            else:
                sd = np.sqrt(((H - mu) ** 2).sum(1).mean()) + 1e-8
                P = PCA(min(PCA_K, len(idx) - 1), random_state=0).fit((H - mu) / sd)
                Z = P.transform((H - mu) / sd)
            if SEG_KIND == "wdim":
                w = (Z[ym].mean(0) - Z[~ym].mean(0)) / (Z.std(0) + 1e-8) ** 2
            elif SEG_KIND == "rpca":
                w = Ridge(alpha=SEG_C).fit(Z, -drop[idx]).coef_
            else:
                w = LogisticRegression(C=SEG_C, max_iter=5000, class_weight="balanced").fit(Z, ym).coef_[0]
            d = P.components_.T @ w / sd
            f = (H - mu) @ d
            ref = Z @ w
            assert np.allclose(f, ref - ref.mean(), atol=1e-6 * (np.abs(ref).max() + 1)), m
            e["windows"][j] = {"site": s, "mean": mu.tolist(), "direction": d.tolist(), "scale": float(f.std() + 1e-8)}
    if SEG_KIND == "upca":
        score = lambda trr, s: stack.upca_scores(acts, models, y, trr, s, FRAC, PCA_K, SEG_C, "lr")
    else:
        kind = {"pca": "pca", "wdim": "wdim", "rpca": "rpca"}[SEG_KIND] + str(PCA_K)
        score = lambda trr, s: stack.probe_scores(acts, models, y, trr, s, FRAC, kind, SEG_C)
    seg_scorer = lambda tr: stack.safe_seg_scores(acts, models, y, groups, tr, SEG_SITES, FRAC, scorer=score)
    ones = np.concatenate([score(all_rows, s) for s in SEG_SITES], 1)
    gated = seg_scorer(all_rows)
    for m, e in seg_entries.items():
        k = np.where(models == m)[0][0]
        e["seg_weight"] = [float(abs(ones[k, j]) > 0 and abs(gated[k, j]) > 0) for j in range(len(SEG_SITES))]
        print(f"  seg gate {m:40s} {e['seg_weight']}")
    F_seg = gated
    T = pd.DataFrame([traj.traj_features(s) for s in traj.load_segs(cases)], index=cases.index)[traj.FEATS]
    traj_stats, Z_traj = th.zscore_entries(T, cases)
    tx_stats, Z_tx = {}, {}
    for b in BUDGETS:
        X = stack.tx_feats(cases, budget=b, n=N_SAMPLES)[TX_FEATS]
        tx_stats[b], Z_tx[b] = th.zscore_entries(X, cases)
    Z_traj, Z_tx = np.asarray(Z_traj), {b: np.asarray(z) for b, z in Z_tx.items()}

    def features(option: str, budget: int) -> np.ndarray:
        parts = {"seg": F_seg, "traj": Z_traj, "tx": Z_tx[budget]}
        return np.concatenate([parts[bk] for bk in OPTIONS[option]], 1)

    seg_block = seg_scorer
    blocks = {"seg": seg_block, "traj": lambda tr: Z_traj, "tx": lambda tr: Z_tx[16384]}
    print("== per-model option (inner CV, seg probe fitted per fold, tx at 16384)")
    chosen = choose_options(cases, blocks, y)

    print("== logistic fits per option and budget")
    logistic = {}
    for name, opt_blocks in OPTIONS.items():
        logistic[name] = {}
        for b in (BUDGETS if "tx" in opt_blocks else [0]):
            F = features(name, b or 16384)
            clf = wb._fit_logistic(F, y, C)
            logistic[name][str(b)] = {"coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
                                      "train_acc": float(((clf.predict_proba(F)[:, 1] >= 0.5) == y).mean())}
        print(f"  {name:12s} train acc " + " ".join(f"{k}:{v['train_acc']:.3f}" for k, v in logistic[name].items()))

    # the CV number the package reproduces: the gated stack at 16k (stack.run_gated); seg fitted per fold
    report, picks = stack.run_gated(cases, {k: {bk: blocks[bk] for bk in OPTIONS[k]} for k in GATE}, REP)
    print(f"gated CV: acc {report['cv_score']:.4f}±{report['cv_std']:.4f} auc {report['auc']:.4f} | {report['per_model']}")
    plain = stack.run(cases, blocks, REP)
    print(f"seg+traj+tx CV: acc {plain['cv_score']:.4f} auc {plain['auc']:.4f}")
    by_budget = {}
    for b in BUDGETS:
        rb = stack.run(cases, {**blocks, "tx": lambda tr, b=b: Z_tx[b]}, max(1, REP // 2))
        by_budget[b] = rb["cv_score"]
    print("seg+traj+tx CV by tx budget: " + " ".join(f"{b}:{a:.3f}" for b, a in by_budget.items()))

    entries = {}
    for m in sorted(set(models)):
        e = {"option": chosen[m], "n_cases": int((models == m).sum()),
             "traj": {"end": traj.END, "frac": FRAC, **{k: traj_stats[m][k] for k in ("median", "mean", "sd")}},
             "tx": {str(b): {k: tx_stats[b][m][k] for k in ("median", "mean", "sd")} for b in BUDGETS}}
        if m in seg_entries:
            e.update({k: seg_entries[m][k] for k in ("layer", "n_hidden_states", "windows", "seg_weight")})
        entries[m] = e
    art = {"schema_version": 1, "head": "stack",
           "generation": {"max_prompt_length": 2048, "greedy_tokens": 4096, "n_samples": N_SAMPLES, "budgets": BUDGETS},
           "segments": 10, "frac": FRAC, "C": C,
           "blocks": {"seg": SEG_SITES, "traj": list(traj.FEATS), "tx": TX_FEATS},
           "options": OPTIONS, "fallback": FALLBACK, "logistic": logistic, "models": entries,
           "training": {"cases": int(len(cases)), "models": sorted(set(models)), "always_true": float(y.mean()),
                        "gated_cv": {k: report[k] for k in ("cv_score", "cv_std", "balanced_acc", "auc", "per_model")},
                        "stack_cv": {k: plain[k] for k in ("cv_score", "cv_std", "auc", "per_model")},
                        "stack_cv_by_budget": by_budget, "picks": {f"{m}|{o}": n for (m, o), n in picks.items()},
                        "source": "data/cvbench (cases.jsonl + cases_oss.jsonl)"}}
    OUT = OUT_DIR or HERE
    (OUT / "artifacts").mkdir(parents=True, exist_ok=True)
    (OUT / "artifacts" / "stack.json").write_text(json.dumps(art))
    for pkg, src in (("trace_engine", ROOT / "solutions" / "trace-heads" / "trace_engine"),
                     ("uncertainty_profile", ROOT / "solutions" / "uncertainty-profiling" / "uncertainty_profile"),
                     ("transcript_probe", ROOT / "solutions" / "transcript-probe" / "transcript_probe")):
        shutil.rmtree(HERE / pkg, ignore_errors=True)
        shutil.copytree(src, HERE / pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    per = "\n".join(f"| {m} | {a} |" for m, a in (x.split("=") for x in report["per_model"].split("; ")))
    (OUT / "README.md").write_text(f"""# Per-model stack, seg direction = PCA(3) + logistic (`stack-pca5`)

One engine per case: the greedy 4k chain of thought (seg probe windows 0-5 / 5-9 at layer frac 0.5 and the
trajectory features step_mean / norm_slope, as `seg-traj`) and 8 sampled traces (transcript features, as
`transcript-probe`), decoded together in one batch (stream 0 forced greedy). Per model the feature set is
chosen by inner cross-validation: {', '.join(f'{m.split("/")[-1]}: {o}' for m, o in chosen.items())}.
Logistic regression (C={C}) per option and thinking budget; the budget each case reaches is set by the
run's time budget (fair share weighted by the model's decode cost, rollover of unused time, snapping to the
largest reachable trained budget). Generated by `research/export.py`.

## CV (cvbench, {len(cases)} cases, 5-fold grouped by problem, 20 repeats, tx at 16k)

| | Accuracy | AUC |
|---|---|---|
| per-model gate (this package) | {report['cv_score']:.3f} ± {report['cv_std']:.3f} | {report['auc']:.3f} |
| seg+traj+tx for every model | {plain['cv_score']:.3f} ± {plain['cv_std']:.3f} | {plain['auc']:.3f} |
| always-True | {y.mean():.3f} | 0.500 |

seg+traj+tx by tx budget: {' '.join(f'{b}: {a:.3f}' for b, a in by_budget.items())}.

| Model | CV accuracy (gated) |
|---|---|
{per}
""")
    print(f"wrote {HERE / 'artifacts' / 'stack.json'}")


if __name__ == "__main__":
    main()
