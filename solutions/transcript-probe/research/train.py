#!/usr/bin/env python3
"""Cross-validate the transcript probe on the 82 small-model cases and export its artifact.

Training transcripts are the original-prompt samples in
``data/anchors/work/**/screen.jsonl`` (8 per problem, vLLM, temperature 1.0,
16384-token budget). Labels are ``data/cvbench/cases.jsonl``. A shorter
thinking budget B is emulated by cutting each sample to its first B tokens
(characters scaled by that sample's characters per token) and counting it as
finished only if it ended within B tokens.

Two evaluations per budget, both with features z-scored within model and a
logistic regression (C=0.3) shared across models:

* CV: cvbench's protocol, 5-fold StratifiedGroupKFold by problem, 20 repeats.
* LOMO: train on three models, test on the fourth.

``length`` (log mean tokens) is reported as the reference. Results go to
``research/results.csv``; ``--export`` fits on all cases and writes
``artifacts/model.json`` with one fit per budget. ``--oss`` adds the gpt-oss-120b
cases (sampled by ``research/sample_oss.py``), z-scored per effort; the artifact
then carries stats under "openai/gpt-oss-120b:<effort>".

    .venv/bin/python solutions/transcript-probe/research/train.py --export
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

from transcript_probe.features import FEATURES, transcript_features  # noqa: E402

WORK = ROOT / "data" / "anchors" 
SCREENS = {"Qwen/Qwen3.5-4B": WORK, "Skywork/Skywork-OR1-Math-7B": WORK / "Skywork_Skywork-OR1-Math-7B",
           "allenai/Olmo-3-7B-Think": WORK / "allenai_Olmo-3-7B-Think",
           "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": WORK / "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B"}
OSS = ROOT / "data" / "transcript_probe" / "oss"  # research/sample_oss.py: screen_<effort>.jsonl
BUDGETS = (2048, 4096, 8192, 12288, 16384)
C = 0.3


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def clean(text: str) -> str:
    """DeepSeek's screen traces were saved as byte-level BPE pieces; map space and newline back."""
    return text.replace("Ġ", " ").replace("Ċ", "\n") if "Ġ" in text else text


def load_cases(oss: bool = False) -> list[dict]:
    """Small-model cases; with ``oss`` also gpt-oss-120b, one pseudo-model per effort ("...:high")."""
    cases = read_jsonl(ROOT / "data" / "cvbench" / "cases.jsonl")
    screens = {model_id: directory / "screen.jsonl" for model_id, directory in SCREENS.items()}
    if oss:
        cases += read_jsonl(ROOT / "data" / "cvbench" / "cases_oss.jsonl")
        screens.update({f"openai/gpt-oss-120b:{e}": OSS / f"screen_{e}.jsonl" for e in ("low", "medium", "high")})
    originals = {}
    for model_id, path in screens.items():
        for row in read_jsonl(path):
            if row["variant"] == "X":
                originals[(model_id, row["group"])] = row
    for case in cases:
        case["screen"] = originals[(case["model_id"], case["group"])]
    return cases


def case_features(case: dict, budget: int, n: int) -> dict:
    x = case["screen"]
    texts, finished, tokens = [], [], []
    for text, n_tok, finish in list(zip(x["texts"], x["n_tokens"], x["finish"]))[:n]:
        text, n_tok = clean(text), int(n_tok)
        if n_tok > budget:
            text = text[: int(budget * len(text) / max(n_tok, 1))]
        texts.append(text)
        finished.append(finish == "stop" and n_tok <= budget)
        tokens.append(min(n_tok, budget))
    feats = transcript_features(case["problem"], texts, finished)
    feats["length"] = mean(math.log(t + 1) for t in tokens)
    return feats


def zscore_within_model(X: np.ndarray, models: np.ndarray) -> np.ndarray:
    Z = X.copy()
    for m in np.unique(models):
        idx = models == m
        sd = X[idx].std(0)
        Z[idx] = (X[idx] - X[idx].mean(0)) / np.where(sd > 0, sd, 1.0)
    return Z


def fit(Z, y):
    return LogisticRegression(C=C, max_iter=2000).fit(Z, y)


def summary(y, pred, score, models, mask=None) -> dict:
    per = {m.split("/")[-1]: float((pred[models == m] == y[models == m]).mean()) for m in np.unique(models)}
    return {"acc": float((pred == y).mean()), "bacc": float(balanced_accuracy_score(y, pred)),
            "auc": float(roc_auc_score(y, score)), "per_model": per}


def cross_validate(Z, y, groups, models, repeats=20) -> dict:
    runs = []
    for r in range(repeats):
        pred, score = np.zeros(len(y), bool), np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(Z, y, groups):
            score[te] = fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
            pred[te] = score[te] >= 0.5
        runs.append(summary(y, pred, score, models))
    out = {k: float(np.mean([run[k] for run in runs])) for k in ("acc", "bacc", "auc")}
    out["acc_std"] = float(np.std([run["acc"] for run in runs]))
    out["per_model"] = {m: float(np.mean([run["per_model"][m] for run in runs])) for m in runs[0]["per_model"]}
    return out


def leave_one_model_out(Z, y, models) -> dict:
    pred, score = np.zeros(len(y), bool), np.zeros(len(y))
    for m in np.unique(models):
        te = models == m
        score[te] = fit(Z[~te], y[~te]).predict_proba(Z[te])[:, 1]
        pred[te] = score[te] >= 0.5
    return summary(y, pred, score, models)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=8, help="Samples per problem (at most 8).")
    parser.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--export", action="store_true", help="Write artifacts/model.json.")
    parser.add_argument("--oss", action="store_true",
                        help="Add the 37 gpt-oss-120b cases (research/sample_oss.py); results to results_oss.csv.")
    args = parser.parse_args()

    cases = load_cases(args.oss)
    y = np.array([bool(c["is_robust"]) for c in cases])
    models = np.array([c["model_id"] for c in cases])
    groups = np.array([c["group"] for c in cases])
    print(f"{len(cases)} cases, {y.mean():.3f} robust (always-True accuracy)")

    results, fits = [], {}
    for budget in args.budgets:
        rows = [case_features(c, budget, args.n) for c in cases]
        for name, cols in (("length", ["length"]), ("transcript_probe", list(FEATURES))):
            raw = np.array([[r[k] for k in cols] for r in rows], dtype=float)
            Z = zscore_within_model(raw, models)
            for kind, res in (("cv", cross_validate(Z, y, groups, models, args.repeats)),
                              ("lomo", leave_one_model_out(Z, y, models))):
                per = "; ".join(f"{m}={v:.3f}" for m, v in res["per_model"].items())
                print(f"budget {budget:5d} {name:16s} {kind:4s} acc {res['acc']:.3f}"
                      f"{'±%.3f' % res['acc_std'] if 'acc_std' in res else '      '} bacc {res['bacc']:.3f} "
                      f"auc {res['auc']:.3f} | {per}")
                results.append({"budget": budget, "n": args.n, "features": name, "eval": kind,
                                **{k: round(res[k], 4) for k in ("acc", "bacc", "auc")}, "per_model": per})
            if name == "transcript_probe":
                clf = fit(Z, y)
                stats = {m: {"mean": raw[models == m].mean(0).tolist(), "sd": raw[models == m].std(0).tolist()}
                         for m in np.unique(models)}
                fits[str(budget)] = {"coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
                                     "stats": stats}

    results_path = HERE / ("results_oss.csv" if args.oss else "results.csv")
    with results_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {results_path}")
    if args.export:
        out = HERE.parent / "artifacts" / "model.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"features": list(FEATURES), "n": args.n, "C": C, "n_cases": len(cases),
                                   "robust_share": float(y.mean()), "fits": fits}, indent=1))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
