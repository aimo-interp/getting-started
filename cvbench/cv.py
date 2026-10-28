#!/usr/bin/env python3
"""Cross-validate robustness methods on the small-model CV set.

Every method turns the cached signals in ``data/cvbench/`` (``generate.py``)
into a feature table indexed by case. Features are z-scored within each model
over all of that model's cases (label-free), and a logistic regression is
fitted on the training folds. Folds are grouped by problem text across models
and stratified by label: 5 folds, repeated ``--repeats`` times. The reported
``cv_score`` is the mean out-of-fold accuracy, the leaderboard metric.

Each run appends one row per method to ``cvbench/results.csv``.

    uv run cvbench/cv.py --methods always_true uncertainty_cv cot_text
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cvbench"
RESULTS = Path(__file__).resolve().parent / "results.csv"
FIELDS = ["method", "type", "hyperparams", "cv_score", "cv_std", "balanced_acc", "auc", "n_cases", "per_model"]
sys.path[:0] = [str(ROOT / "solutions" / "uncertainty-profiling")]


def safe(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


CASE_FILES = ("cases.jsonl", "cases_oss.jsonl")  # small models; gpt-oss-120b by effort (model_id "...:<effort>")


def load_cases(files=CASE_FILES) -> pd.DataFrame:
    return pd.DataFrame([r for f in files for r in read_jsonl(DATA / f)]).set_index("case_id")


def by_group(cases: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    """Join per-(model, group) rows onto cases."""
    table = {(r["model_id"], r["group"]): r for r in rows}
    return pd.DataFrame([table.get((c.model_id, c.group), {}) for c in cases.itertuples()], index=cases.index)


# --------------------------------------------------------------------------- methods
# Each returns (features DataFrame indexed like cases, hyperparams string).

def m_always_true(cases, args):
    return None, "predict True for every case"


def greedy_table(cases):
    rows = []
    for model_id in cases.model_id.unique():
        rows += read_jsonl(DATA / "greedy" / f"{safe(model_id)}.jsonl")
    return by_group(cases, rows)


def m_uncertainty_shipped(cases, args):
    """The checked-in artifact as the solution runs it: DeepSeek only, False elsewhere."""
    return "shipped", "checked-in RF artifact (DeepSeek only); max_new_tokens=4096 greedy"


def m_uncertainty_cv(cases, args):
    from uncertainty_profile.config import FEATURE_NAMES
    g = greedy_table(cases)
    return g[list(FEATURE_NAMES)], "14 baseline features; greedy max_new_tokens=4096; logistic C=%g" % args.C


def m_uncertainty_rf(cases, args):
    """The baseline recipe retrained per fold: RF regressor on max_drop, threshold from inner OOF (balanced acc)."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold
    from uncertainty_profile.config import FEATURE_NAMES
    g = greedy_table(cases)
    X = g[list(FEATURE_NAMES)].astype(float).fillna(g[list(FEATURE_NAMES)].median()).to_numpy()
    y = cases.is_robust.astype(bool).to_numpy()
    drop = cases.max_drop.astype(float).fillna(pd.Series(np.where(y, 0.0, 0.5), index=cases.index)).clip(0, 1).to_numpy()
    groups = cases.group.to_numpy()

    def rf():
        return RandomForestRegressor(n_estimators=500, max_features=1.0, min_samples_leaf=1, random_state=0, n_jobs=-1)

    def predict(tr, te):
        oof = np.zeros(len(tr))
        for itr, ite in GroupKFold(4).split(tr, groups=groups[tr]):
            oof[ite] = rf().fit(X[tr][itr], drop[tr][itr]).predict(X[tr][ite])
        best = max(np.unique(oof), key=lambda t: (balanced_accuracy_score(y[tr], oof <= t), -t))
        pred = rf().fit(X[tr], drop[tr]).predict(X[te])
        return pred <= best, -pred
    return predict, "baseline recipe: RF(500, max_features=1.0) on max_drop; threshold by inner OOF balanced acc; greedy 4096"


HESITATION = re.compile(
    r"\b(wait|hmm+|alternatively|actually|let me (?:double[- ]check|re-?check|verify|reconsider)|"
    r"but (?:wait|hold on)|on second thought|i made (?:a|an) (?:mistake|error)|that'?s (?:not right|wrong))\b",
    re.IGNORECASE)
CANDIDATE = re.compile(r"(?:answer|result|get|is)\s*(?:is|=|:)?\s*\$?\\?(?:boxed\{)\s*(-?\d[\d,]*)", re.IGNORECASE)
NUMBER_CLAIM = re.compile(r"(?:=|is)\s*\$?(-?\d{1,7})\$?[.\s]")


def cot_features(text: str, n_tokens: int, cap: int = 4096) -> dict:
    words = max(len(text.split()), 1)
    think, _, after = text.partition("</think>")
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    claims = NUMBER_CLAIM.findall(think)
    counts = Counter(claims)
    return {
        "log_tokens": math.log(n_tokens + 1),
        "truncated": float(n_tokens >= cap - 1),
        "finished_thinking": float(bool(after)),
        "hesitation_per_1k": 1000 * len(HESITATION.findall(think)) / words,
        "wait_per_1k": 1000 * len(re.findall(r"\bwait\b", think, re.IGNORECASE)) / words,
        "n_boxed_distinct": len(set(boxed)),
        "has_boxed": float(bool(boxed)),
        "claim_entropy": -sum(v / len(claims) * math.log(v / len(claims)) for v in counts.values()) if claims else 0.0,
        "verify_per_1k": 1000 * len(re.findall(r"\b(check|verify|confirm)\b", think, re.IGNORECASE)) / words,
    }


def m_cot_text(cases, args):
    g = greedy_table(cases)
    feats = pd.DataFrame([cot_features(t if isinstance(t, str) else "", int(n) if n == n else 0)
                          for t, n in zip(g["generated_text"], g["generation_num_tokens"])], index=cases.index)
    return feats, "greedy CoT, max_new_tokens=4096; hesitation/wait/verify rates, length, truncation, answer churn"


def m_length(cases, args):
    g = greedy_table(cases)
    n = g["generation_num_tokens"].fillna(0).astype(float)
    feats = pd.DataFrame({"log_tokens": np.log1p(n), "truncated": (n >= 4095).astype(float)}, index=cases.index)
    return feats, "greedy response length: log tokens + hit the 4,096 cap"


def m_sample_length(cases, args):
    """Mean log length and cap rate of the 10 sep2 samples of the original problem (NaN -> median when absent)."""
    norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()
    pids = {norm(r["problem"]): r["problem_id"] for r in read_jsonl(ROOT / "data" / "labelgen" / "problems.jsonl")}
    stats = {}
    for model_id in cases.model_id.unique():
        path = DATA / "sep2" / "samples" / f"{safe(model_id)}__default.jsonl"
        for r in read_jsonl(path) if path.is_file() else []:
            if r["type"] == "original":
                n = np.array(r["n_tokens"], float)
                stats[(model_id, r["problem_id"])] = (np.log1p(n).mean(), (n >= 8192).mean())
    rows = [stats.get((c.model_id, pids.get(norm(c.problem))), (np.nan, np.nan)) for c in cases.itertuples()]
    feats = pd.DataFrame(rows, columns=["sample_log_tokens", "sample_capped"], index=cases.index)
    return feats, f"sep2 samples of the original (10, 8,192 cap): mean log length + cap rate; {int(feats.iloc[:, 0].notna().sum())} cases covered"


def m_introspect(cases, args):
    rows = []
    for model_id in cases.model_id.unique():
        rows += read_jsonl(DATA / "introspect" / f"{safe(model_id)}.jsonl")
    t = by_group(cases, rows)
    feats = t[["robust", "correct", "long"]].astype(float)
    return feats, ("zero-shot self-prediction (Looking Inward): Yes-vs-No log-odds, no CoT, for 'answer unchanged "
                   "under rewording', 'answer correct', 'solution over 4k tokens'")


def m_uncertainty_cot(cases, args):
    a, _ = m_uncertainty_cv(cases, args)
    b, _ = m_cot_text(cases, args)
    return pd.concat([a, b], axis=1), "14 uncertainty + 9 CoT features; greedy max_new_tokens=4096"


def sample_table(cases, args):
    rows = []
    for model_id in cases.model_id.unique():
        rows += read_jsonl(DATA / "sample" / f"{safe(model_id)}__n{args.n}_b{args.budget}.jsonl")
    return by_group(cases, rows)


def m_resample(cases, args):
    s = sample_table(cases, args)
    n = args.n
    feats = pd.DataFrame(index=cases.index)
    feats["x_consistency"] = s["x_consistency"].fillna(0.0)
    feats["no_answer"] = s["reference"].isna().astype(float)
    feats["lp_x"] = s["lp_x"].fillna(s["lp_x"].min())
    feats["finish_x"] = s["finish_x"]
    feats["tokens_x"] = s["tokens_x"]
    feats["n_distinct_x"] = [len(set(a[:n])) if isinstance(a, list) else n for a in s["answers"]]
    return feats, f"original only; n={n} samples T=1.0; thinking budget={args.budget}; forced boxed answer"


def m_perturb(cases, args):
    s = sample_table(cases, args)
    feats = pd.DataFrame(index=cases.index)
    for col in ["agree_mean", "agree_min", "x_consistency"]:
        feats[col] = s[col].fillna(0.0)
    for col in ["dlogp", "dlogp_max"]:
        feats[col] = s[col].fillna(s[col].max())
    feats["no_answer"] = s["reference"].isna().astype(float)
    return feats, (f"X + 4 code perturbations (typos/rename/distract/rephrase); n={args.n} each T=1.0; "
                   f"budget={args.budget}; forced-answer agreement + dlogp")


def acts_file(model_id, group):
    return DATA / "acts" / safe(model_id) / f"{group}.npz"


def m_entropy_trace(cases, args):
    rows = []
    for c in cases.itertuples():
        path = acts_file(c.model_id, c.group)
        if not path.is_file():
            rows.append({})
            continue
        z = np.load(path)
        e = z["entropy"].astype(np.float32)
        if e.size < 8:
            rows.append({})
            continue
        q = np.array_split(e, 4)
        t = np.linspace(0, 1, e.size)
        win = np.convolve(e, np.ones(64) / 64, mode="valid") if e.size > 64 else e
        rows.append({"ent_q1": q[0].mean(), "ent_q2": q[1].mean(), "ent_q3": q[2].mean(), "ent_q4": q[3].mean(),
                     "ent_slope": np.polyfit(t, e, 1)[0], "ent_late_minus_early": q[3].mean() - q[0].mean(),
                     "ent_spike_frac": float((e > 2.0).mean()), "ent_win_max": float(win.max()),
                     "ent_win_std": float(win.std())})
    return pd.DataFrame(rows, index=cases.index), "greedy trace (4096) entropy profile: quartiles, slope, spikes, 64-tok windows"


# --------------------------------------------------------------------------- evaluation

def zscore_within_model(feats: pd.DataFrame, cases: pd.DataFrame) -> np.ndarray:
    X = feats.astype(float).copy()
    X = X.fillna(X.median())
    for _, idx in cases.groupby("model_id").groups.items():
        block = X.loc[idx]
        X.loc[idx] = (block - block.mean()) / block.std(ddof=0).replace(0, 1)
    return X.fillna(0.0).to_numpy()


def evaluate(cases, predict_fold, repeats: int, seed: int = 0, score_mask=None):
    """predict_fold(train_idx, test_idx) -> (bool preds, scores) for test rows. ``score_mask``: train on
    every case but compute the metrics on these cases only (e.g. organizer labels)."""
    y = cases.is_robust.astype(bool).to_numpy()
    keep = np.ones(len(y), dtype=bool) if score_mask is None else np.asarray(score_mask)
    groups = cases.group.to_numpy()
    accs, baccs, aucs, per_model = [], [], [], {m: [] for m in cases.model_id.unique()}
    for r in range(repeats):
        pred = np.zeros(len(y), dtype=bool)
        score = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed + r).split(y, y, groups):
            p, s = predict_fold(tr, te)
            pred[te], score[te] = p, s
        accs.append((pred[keep] == y[keep]).mean())
        baccs.append(balanced_accuracy_score(y[keep], pred[keep]))
        aucs.append(roc_auc_score(y[keep], score[keep]) if np.ptp(score[keep]) > 0 else 0.5)
        for m in per_model:
            mask = (cases.model_id == m).to_numpy() & keep
            per_model[m].append((pred[mask] == y[mask]).mean())
    return {"cv_score": float(np.mean(accs)), "cv_std": float(np.std(accs)), "balanced_acc": float(np.mean(baccs)),
            "auc": float(np.mean(aucs)),
            "per_model": "; ".join(f"{m.split('/')[-1]}={np.mean(v):.3f}" for m, v in per_model.items())}


def logistic_predictor(X, y, C):
    def predict(tr, te):
        if len(set(y[tr])) < 2:
            return np.full(len(te), y[tr][0]), np.zeros(len(te))
        clf = LogisticRegression(C=C, max_iter=2000).fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        return prob >= 0.5, prob
    return predict


def shipped_predictions(cases) -> np.ndarray:
    from uncertainty_profile.artifact import load_artifact
    art = load_artifact("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    g = greedy_table(cases)
    pred = np.zeros(len(cases), dtype=bool)
    score = np.zeros(len(cases))
    ds = (cases.model_id == "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B").to_numpy() & g["generation_mean_logprob"].notna().to_numpy()
    if ds.any():
        drop = np.asarray(art.estimator.predict(g.loc[ds, list(art.feature_names)].to_numpy(dtype=float)))
        pred[ds] = drop < art.decision_threshold
        score[ds] = -drop
    return pred, score


METHODS = {
    "always_true": ("reference", m_always_true),
    "uncertainty_shipped": ("blackbox", m_uncertainty_shipped),
    "uncertainty_cv": ("blackbox", m_uncertainty_cv),
    "uncertainty_rf": ("blackbox", m_uncertainty_rf),
    "cot_text": ("blackbox", m_cot_text),
    "length": ("blackbox", m_length),
    "introspect": ("blackbox", m_introspect),
    "sample_length": ("blackbox", m_sample_length),
    "uncertainty_cot": ("blackbox", m_uncertainty_cot),
    "entropy_trace": ("blackbox", m_entropy_trace),
    "resample": ("blackbox", m_resample),
    "perturb": ("blackbox", m_perturb),
}


def register_whitebox():
    try:
        import whitebox  # noqa: F401  (cvbench/whitebox.py adds its methods to METHODS)
        whitebox.register(METHODS, sys.modules[__name__])
    except ImportError:
        pass


def write_rows(rows):
    new = not RESULTS.is_file()
    with RESULTS.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    register_whitebox()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=["always_true"])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--C", type=float, default=0.3)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--layer-frac", type=float, default=None)
    parser.add_argument("--with-bb", nargs="*", default=None, help="Blackbox feature methods to append to act_probe.")
    parser.add_argument("--layer-window", type=int, default=0)
    parser.add_argument("--pca-C", type=float, nargs="+", default=None)
    parser.add_argument("--pca-k", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--site", default=None, choices=["last_prompt", "resp_mean", "prob_mean", "tmpl_last"])
    parser.add_argument("--shift-sites", nargs="+", default=["last"], choices=["last", "prob_mean", "tmpl_last"])
    parser.add_argument("--shift-feats", nargs="+", default=["resid"], choices=["proj", "resid", "norm"])
    parser.add_argument("--with-probe", action="store_true", help="shift_resid: also add the resp_mean probe score")
    parser.add_argument("--seg-windows", nargs="+", default=None, help="seg_probe: fixed windows a-b over 10 segments.")
    parser.add_argument("--seg-combine", action="store_true", help="seg_probe: use every window's score as a feature.")
    parser.add_argument("--sep-alpha", type=float, default=1000.0, help="sep_probe ridge penalty.")
    parser.add_argument("--sep-with-seg", action="store_true", help="sep_probe: add the seg_probe 0-5/5-9 scores.")
    parser.add_argument("--sep-pos", default="tbg", choices=["tbg", "slt"], help="sep2_probe: token position.")
    parser.add_argument("--sep-block", type=int, default=5, help="sep2_probe: adjacent layers concatenated.")
    parser.add_argument("--sep-none", default="cluster", choices=["cluster", "drop"],
                        help="sep2_probe: a sample with no answer is its own cluster, or left out.")
    parser.add_argument("--saplma-target", default="robust", choices=["robust", "correct"], help="saplma: training target.")
    parser.add_argument("--saplma-alpha", type=float, default=1e-2, help="saplma: MLP L2 penalty.")
    parser.add_argument("--len-frac", type=float, default=0.7, help="len_probe: layer fraction.")
    parser.add_argument("--len-alpha", type=float, default=1e4, help="len_probe: ridge penalty.")
    parser.add_argument("--len-with-seg", action="store_true", help="len_probe: add the seg_probe 0-5/5-9 scores.")
    parser.add_argument("--sep-answers", default="sampled", choices=["sampled", "forced"],
                        help="sep2_probe: answers as sampled, or with truncated samples budget-forced (force_answers.py).")
    parser.add_argument("--layer-report", action="store_true", help="Write per-layer AUCs to data/cvbench/layer_report.csv")
    parser.add_argument("--cases", nargs="+", default=list(CASE_FILES), help="Case files under data/cvbench.")
    parser.add_argument("--pool-efforts", action="store_true",
                        help="seg_probe: one direction per base model, shared by its reasoning efforts.")
    parser.add_argument("--models", nargs="+", default=None, help="Restrict to these model ids (subset CV).")
    parser.add_argument("--sources", nargs="+", default=None, help="Restrict to these label sources, e.g. train-main-v2.")
    parser.add_argument("--score-sources", nargs="+", default=None,
                        help="Train on every case but score only cases from these label sources.")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.models:
        cases = cases[cases.model_id.isin(args.models)]
    if args.sources:
        cases = cases[cases.source.isin(args.sources)]
    score_mask = cases.source.isin(args.score_sources).to_numpy() if args.score_sources else None
    if args.layer_report:
        import whitebox
        df = whitebox.layer_report(cases, sys.modules[__name__], DATA / "layer_report.csv")
        print(df.groupby("model")[[c for c in df.columns if c.startswith("auc")]].agg(["max", "idxmax"]))
        return
    y = cases.is_robust.astype(bool).to_numpy()
    rows = []
    for name in args.methods:
        kind, fn = METHODS[name]
        out = fn(cases, args)
        feats, hyper = out
        if name == "always_true":
            res = evaluate(cases, lambda tr, te: (np.ones(len(te), bool), np.zeros(len(te))), 1, score_mask=score_mask)
        elif isinstance(feats, str) and feats == "shipped":
            pred, score = shipped_predictions(cases)
            res = evaluate(cases, lambda tr, te: (pred[te], score[te]), 1, score_mask=score_mask)
        elif callable(feats):  # method supplies its own fold predictor
            res = evaluate(cases, feats, args.repeats, score_mask=score_mask)
        else:
            X = zscore_within_model(feats, cases)
            res = evaluate(cases, logistic_predictor(X, y, args.C), args.repeats, score_mask=score_mask)
            hyper += f"; logistic C={args.C}" if "logistic" not in hyper else ""
        if list(args.cases) != list(CASE_FILES):
            hyper += f"; cases={','.join(args.cases)}"
        if args.models:
            hyper += f"; SUBSET models={','.join(m.split('/')[-1] for m in args.models)}"
        if args.sources:
            hyper += f"; SUBSET sources={','.join(args.sources)}"
        if args.score_sources:
            hyper += f"; SCORED ON sources={','.join(args.score_sources)} ({int(score_mask.sum())} cases)"
        row = {"method": name, "type": kind, "hyperparams": hyper, "n_cases": len(cases), **res}
        rows.append(row)
        print(f"{name:24s} acc={res['cv_score']:.3f}±{res['cv_std']:.3f} bacc={res['balanced_acc']:.3f} "
              f"auc={res['auc']:.3f} | {res['per_model']}", flush=True)
    if not args.no_write:
        write_rows(rows)


if __name__ == "__main__":
    main()
