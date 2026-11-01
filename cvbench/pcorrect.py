#!/usr/bin/env python3
"""Competence probe (2026-11-02): predict a model's own solve rate p (fraction of 10 sep2 samples correct on
the original problem) from the last-prompt hidden state, trained on labelgen problems without robustness labels.

    python cvbench/pcorrect.py cv        # 1. grouped 5-fold CV of p per model and layer (all labelgen problems)
    python cvbench/pcorrect.py bench     # 2. train on non-benchmark problems, score the 118 cases -> AUC for is_robust
    python cvbench/pcorrect.py selflabel # 4. Skywork self-made robustness labels from sep2 perturbation samples

``bench`` also writes data/cvbench/pcorrect/scores.csv (case_id -> pc_l0/pc_l1/pc_l2), used by stack.py's
``pc`` block. Benchmark states come from data/cvbench/acts (same layer by frac, centred per source).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
import cv  # noqa: E402
import whitebox as wb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cvbench"
ALL_MODELS = ("Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
              "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "openai/gpt-oss-120b:low", "openai/gpt-oss-120b:medium",
              "openai/gpt-oss-120b:high")
safe_id = lambda m: re.sub(r"[^A-Za-z0-9._-]+", "_", m)  # noqa: E731  state dir name (as generate.py)


def samples_file(model_id: str) -> Path:
    base, _, effort = model_id.partition(":")
    return DATA / "sep2" / "samples" / f"{safe_id(base)}__{effort or 'default'}.jsonl"


# model_id -> state dir name, for every model with sep2 samples and last-prompt states
MODELS = {m: safe_id(m) for m in ALL_MODELS
          if samples_file(m).is_file() and (DATA / "effort" / safe_id(m)).is_dir()}
ALPHAS = (1e2, 1e3, 1e4)
norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()  # noqa: E731


def ans(a) -> str:
    try:
        return str(int(float(str(a).strip())))
    except (ValueError, TypeError, OverflowError):
        return norm(str(a))


def problems():
    return {r["problem_id"]: r for r in map(json.loads, open(ROOT / "data" / "labelgen" / "problems.jsonl"))}


def samples(model_id: str):
    out = {}
    for r in map(json.loads, open(samples_file(model_id))):
        a = r["answers"]
        out[(r["problem_id"], r["type"], str(r["variant"]))] = json.loads(a) if isinstance(a, str) else a
    return out


def solve_rate(answers, ref) -> float:
    return float(np.mean([ans(x) == ans(ref) for x in answers]))


def table(model_id: str, P: dict):
    """Rows (problem_id, p, X[3, d]) for every labelgen problem with samples, an answer and an effort state."""
    safe = MODELS[model_id]
    S = samples(model_id)
    rows = []
    for (pid, typ, _), a in S.items():
        if typ != "original" or not P.get(pid, {}).get("answer"):
            continue
        f = DATA / "effort" / safe / f"{pid}_original_0.npz"
        if f.is_file():
            z = np.load(f)
            rows.append((pid, solve_rate(a, P[pid]["answer"]), z["last_prompt"].astype(np.float64), z["fracs"]))
    return rows


def fit_ridge(X, p, alpha):
    mu = X.mean(0)
    return Ridge(alpha=alpha).fit(X - mu, p), mu


def cmd_cv(_):
    P = problems()
    for m in MODELS:
        rows = table(m, P)
        pids = np.array([r[0] for r in rows]); p = np.array([r[1] for r in rows])
        X3 = np.stack([r[2] for r in rows])
        print(f"\n{m}: n={len(p)}  mean p={p.mean():.2f}  p>=.7: {np.mean(p >= .7):.2f}  p==0: {np.mean(p == 0):.2f}")
        for j in range(X3.shape[1]):
            X = X3[:, j]
            for alpha in ALPHAS:
                pred = np.zeros(len(p)); prob = np.zeros(len(p))
                for tr, te in GroupKFold(5).split(X, p, pids):
                    r, mu = fit_ridge(X[tr], p[tr], alpha)
                    pred[te] = r.predict(X[te] - mu)
                    yb = p[tr] >= .7
                    if 0 < yb.sum() < len(yb):
                        lg = LogisticRegression(C=1e-3, max_iter=2000).fit(X[tr] - mu, yb)
                        prob[te] = lg.decision_function(X[te] - mu)
                yb = p >= .7
                print(f"  layer frac {rows[0][3][j]:.1f} alpha {alpha:g}: spearman {spearmanr(pred, p)[0]:.3f}"
                      f"  AUC(p>=.7) ridge {roc_auc_score(yb, pred):.3f}  logistic {roc_auc_score(yb, prob):.3f}")


def bench_states(cases):
    out = []
    for c in cases.itertuples():
        out.append(np.load(cv.acts_file(c.model_id, c.group))["last_prompt"].astype(np.float64))
    return out


def cmd_bench(args):
    alpha = float(args[0]) if args else 1e3
    P = problems()
    pid_of = {norm(r["problem"]): k for k, r in P.items()}
    cases = cv.load_cases(list(cv.CASE_FILES))
    bench_pids = {pid_of.get(norm(t)) for t in cases.problem} - {None}
    cases = cases.copy()
    cols = {f"pc_l{j}": np.full(len(cases), np.nan) for j in range(3)}
    true_p = np.full(len(cases), np.nan)
    for m, safe in MODELS.items():
        rows = [r for r in table(m, P) if r[0] not in bench_pids]
        p = np.array([r[1] for r in rows]); X3 = np.stack([r[2] for r in rows]); fracs = rows[0][3]
        k = np.where(cases.model_id.to_numpy() == m)[0]
        S = samples(m)
        for i in k:
            pid = pid_of.get(norm(cases.problem.iloc[i]))
            if pid and (pid, "original", "0") in S and P[pid].get("answer"):
                true_p[i] = solve_rate(S[(pid, "original", "0")], P[pid]["answer"])
        A = bench_states(cases.iloc[k])
        L = A[0].shape[0]
        for j, fr in enumerate(fracs):
            r, mu = fit_ridge(X3[:, j], p, alpha)
            B = np.stack([a[wb.layer_of(L, fr)] for a in A])
            # centre benchmark states on their own mean (prompt template differs slightly from the effort stage)
            cols[f"pc_l{j}"][k] = r.predict(B - B.mean(0))
        print(f"{m}: trained on {len(p)} problems (excluded {len(bench_pids)} benchmark problems), scored {len(k)} cases")
    for c, v in cols.items():
        cases[c] = v
    cases["true_p"] = true_p
    y = cases.is_robust.astype(bool).to_numpy()
    out = DATA / "pcorrect"; out.mkdir(exist_ok=True)
    cases[["model_id", "group", "is_robust", "true_p", *cols]].to_csv(out / "scores.csv")
    auc = lambda s, yy: roc_auc_score(yy, s) if 0 < yy.sum() < len(yy) else np.nan  # noqa: E731
    sub = cases.model_id.isin(list(MODELS)).to_numpy()
    print("\nAUC for is_robust (higher pred = robust):")
    for m in list(MODELS) + ["pooled (z within model)"]:
        k = cases.model_id.to_numpy() == m if m in MODELS else sub
        line = f"  {m:42s} n={k.sum():3d} fragile={int((~y[k]).sum()):2d}"
        for c in cols:
            s = cases[c].to_numpy()
            if m not in MODELS:
                s = cv.zscore_within_model(pd.DataFrame({c: s}, index=cases.index), cases)[:, 0]
            line += f"  {c} {auc(s[k], y[k]):.3f}"
        tp = ~np.isnan(true_p) & k
        if tp.sum() > 3:
            line += f"  | true p (n={tp.sum()}) {auc(true_p[tp], y[tp]):.3f}  pc_l1 same rows {auc(cases.pc_l1.to_numpy()[tp], y[tp]):.3f}"
        print(line)


def cmd_selflabel(_):
    P = problems()
    S = samples("Skywork/Skywork-OR1-Math-7B")
    by = {}
    for (pid, typ, var), a in S.items():
        if P.get(pid, {}).get("answer"):
            by.setdefault(pid, {}).setdefault(typ, []).append(solve_rate(a, P[pid]["answer"]))
    rows = []
    for pid, d in by.items():
        if "original" not in d or len(d) < 2:
            continue
        base = np.mean(d["original"])
        drops = {t: base - np.mean(v) for t, v in d.items() if t != "original"}
        rows.append({"problem_id": pid, "base": base, "max_drop": max(drops.values()), "n_types": len(drops)})
    L = pd.DataFrame(rows).set_index("problem_id")
    L["is_robust"] = L.max_drop < 0.25
    full = L[L.n_types >= 6]
    print(f"labelled {len(L)} problems ({len(full)} with all 6 types); robust rate {L.is_robust.mean():.2f}"
          f" (all-6: {full.is_robust.mean():.2f})")
    pid_of = {norm(r["problem"]): k for k, r in P.items()}
    cases = cv.load_cases(list(cv.CASE_FILES))
    sk = cases[(cases.model_id == "Skywork/Skywork-OR1-Math-7B")]
    m = [(c.is_robust, c.max_drop, L.loc[pid_of[norm(c.problem)]]) for c in sk.itertuples()
         if pid_of.get(norm(c.problem)) in L.index]
    agree = np.mean([bool(a) == bool(r.is_robust) for a, _, r in m])
    corr = np.corrcoef([d for _, d, _ in m], [r.max_drop for *_, r in m])[0, 1]
    print(f"benchmark Skywork cases matched: {len(m)}; label agreement {agree:.2f}; max_drop corr {corr:.2f}")
    L.to_csv(DATA / "pcorrect" / "skywork_selflabels.csv")


if __name__ == "__main__":
    (DATA / "pcorrect").mkdir(exist_ok=True)
    {"cv": cmd_cv, "bench": cmd_bench, "selflabel": cmd_selflabel}[sys.argv[1]](sys.argv[2:])
