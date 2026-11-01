#!/usr/bin/env python3
"""Fickle-anchor test (2026-11-02, Hannah's thought-anchors hypothesis at scale).

Hypothesis: a non-robust problem is one whose solution hinges on a "fickle" step: the model's chance of
being right depends sharply on which path it happens to take, so P(correct | prefix of its own trace) swings
(or jumps late at one pivotal sentence) instead of rising smoothly and locking in early.

Cheap thought-anchors curve on every case: cut the case's greedy trace (data/cvbench/greedy) at sentence
boundaries nearest to fractions F of its length, resample N continuations from each prefix (T=1, top-p .95)
and record the final answers.

    python cvbench/fickle.py sample --model Skywork/Skywork-OR1-Math-7B   # vLLM env -> data/cvbench/fickle/
    python cvbench/fickle.py analyze                                        # pyine env: AUC + features csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cvbench"
OUT = DATA / "fickle"
FRACS = (0.0, 0.2, 0.4, 0.6, 0.8)
sys.path.insert(0, str(ROOT / "datagen"))
safe = lambda m: re.sub(r"[^A-Za-z0-9._-]+", "_", m)  # noqa: E731
norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()  # noqa: E731
SENT_END = re.compile(r"(?<=[.!?])\s+|\n+")


def read_jsonl(p):
    return [json.loads(l) for l in open(p)] if Path(p).is_file() else []


def cut(text: str, frac: float) -> str:
    """Prefix ending at the sentence boundary nearest frac * len(text)."""
    if frac <= 0:
        return ""
    target = frac * len(text)
    ends = [m.end() for m in SENT_END.finditer(text)] or [len(text)]
    return text[:min(ends, key=lambda e: abs(e - target))]


def cmd_sample(a):
    from vllm import LLM, SamplingParams
    from common import extract_answer, tokenizer_path
    from transformers import AutoTokenizer

    cases = [c for c in read_jsonl(DATA / "cases.jsonl") if c["model_id"] == a.model]
    greedy = {r["group"]: r["generated_text"] for r in read_jsonl(DATA / "greedy" / f"{safe(a.model)}.jsonl")}
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{safe(a.model)}.jsonl"
    done = {(r["group"], r["frac"]) for r in read_jsonl(out)}
    tok = AutoTokenizer.from_pretrained(tokenizer_path(a.model))
    jobs = []
    for c in cases:
        g = greedy.get(c["group"])
        if g is None:
            continue
        base = tok.apply_chat_template([{"role": "user", "content": c["problem"]}], tokenize=False,
                                       add_generation_prompt=True)
        for f in FRACS:
            if (c["group"], f) not in done:
                pre = cut(g, f)
                jobs.append((c["group"], f, base + pre, len(pre)))
    print(f"{a.model}: {len(cases)} cases, {len(jobs)} prefixes to sample -> {out}", flush=True)
    if not jobs:
        return
    llm = LLM(a.model, tokenizer=tokenizer_path(a.model), seed=0, gpu_memory_utilization=0.9,
              max_model_len=a.max_tokens + 6144, enable_prefix_caching=True)
    sp = SamplingParams(n=a.n, temperature=1.0, top_p=0.95, max_tokens=a.max_tokens, seed=0)
    for i in range(0, len(jobs), a.chunk):
        chunk = jobs[i:i + a.chunk]
        outs = llm.generate([j[2] for j in chunk], sp)
        with open(out, "a") as fh:
            for (g, f, _, nc), o in zip(chunk, outs):
                fh.write(json.dumps({"group": g, "frac": f, "prefix_chars": nc,
                                     "answers": [extract_answer(x.text) if x.finish_reason == "stop" else None
                                                 for x in o.outputs],
                                     "n_tokens": [len(x.token_ids) for x in o.outputs],
                                     "finish": [x.finish_reason for x in o.outputs]}) + "\n")
        print(f"  {min(i + a.chunk, len(jobs))}/{len(jobs)}", flush=True)


def features(rows, gold):
    """rows: list of {frac, answers} for one case. gold: reference answer or None (-> modal answer)."""
    import numpy as np
    rows = sorted(rows, key=lambda r: r["frac"])
    allans = [x for r in rows for x in r["answers"] if x is not None]
    ref = gold if gold is not None else (Counter(allans).most_common(1)[0][0] if allans else None)
    p = np.array([np.mean([x is not None and str(x) == str(ref) for x in r["answers"]]) for r in rows])

    def ent(ans):
        c = Counter("none" if x is None else str(x) for x in ans)
        q = np.array(list(c.values()), float) / len(ans)
        return float(-(q * np.log(q)).sum())
    d = np.diff(p)
    return {"fk_p0": p[0], "fk_pmean": p.mean(), "fk_pmin": p.min(), "fk_tv": np.abs(d).sum(),
            "fk_maxjump": d.max() if len(d) else 0.0, "fk_maxdrop": -d.min() if len(d) else 0.0,
            "fk_lock": p[-1] - p[0], "fk_ent": np.mean([ent(r["answers"]) for r in rows]),
            "fk_ent0": ent(rows[0]["answers"]), "fk_npts": len(rows),
            "fk_pvar": p.var(), "fk_none": np.mean([x is None for r in rows for x in r["answers"]])}


def gold_answers():
    P = read_jsonl(ROOT / "data" / "labelgen" / "problems.jsonl")
    out = {}
    for r in P:
        if r.get("answer") not in (None, ""):
            try:
                out[norm(r["problem"])] = int(float(str(r["answer"]).strip()))
            except ValueError:
                pass
    return out


def load_features(cases):
    """DataFrame of fk_* features indexed like cases (NaN where not sampled)."""
    import pandas as pd
    gold = gold_answers()
    feats = {}
    for m in cases.model_id.unique():
        by = {}
        for r in read_jsonl(OUT / f"{safe(m)}.jsonl"):
            by.setdefault(r["group"], []).append(r)
        for i, c in cases[cases.model_id == m].iterrows():
            rows = by.get(c.group)
            if rows and len({r["frac"] for r in rows}) >= 3:
                feats[i] = features(rows, gold.get(norm(c.problem)))
    return pd.DataFrame.from_dict(feats, orient="index").reindex(cases.index)


def cmd_analyze(_):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, str(ROOT / "cvbench"))
    import cv
    cases = cv.load_cases(list(cv.CASE_FILES))
    F = load_features(cases)
    ok = F.fk_p0.notna().to_numpy()
    y = cases.is_robust.astype(bool).to_numpy()
    print(f"cases with fickle curves: {ok.sum()} / {len(cases)}  (fragile {int((~y[ok]).sum())})")
    print("AUC for is_robust per feature (pooled z-within-model; >.5 = higher feature -> robust):")
    Z = cv.zscore_within_model(F.fillna(F.mean()), cases)
    for j, col in enumerate(F.columns):
        print(f"  {col:12s} pooled {roc_auc_score(y[ok], Z[ok, j]):.3f}  " + "  ".join(
            f"{m.split('/')[-1][:8]} {roc_auc_score(y[k], F[col].to_numpy()[k]):.2f}"
            for m in cases.model_id.unique()
            for k in [ok & (cases.model_id.to_numpy() == m)] if k.sum() > 3 and 0 < y[k].sum() < k.sum()))
    # partial: does fickleness add over plain solve stability (p at the empty prefix)?
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    cols = list(F.columns)
    for name, use in [("p0 only", ["fk_p0"]), ("p0+tv", ["fk_p0", "fk_tv"]), ("p0+maxdrop", ["fk_p0", "fk_maxdrop"]),
                      ("p0+lock", ["fk_p0", "fk_lock"]), ("all", cols)]:
        X = Z[ok][:, [cols.index(c) for c in use]]
        aucs = [roc_auc_score(y[ok], cross_val_predict(LogisticRegression(C=0.3), X, y[ok], method="decision_function",
                                                      cv=StratifiedKFold(5, shuffle=True, random_state=s))) for s in range(10)]
        print(f"  logistic {name:12s} cv AUC {np.mean(aucs):.3f}")
    F.assign(model_id=cases.model_id, is_robust=y).to_csv(OUT / "features.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "analyze"])
    ap.add_argument("--model")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=40)
    a = ap.parse_args()
    {"sample": cmd_sample, "analyze": cmd_analyze}[a.cmd](a)
