"""Codabench entry point for the transcript probe.

For each problem: sample ``N`` reasoning traces on the original problem only,
compute three gold-free transcript features (transcript_probe/features.py),
z-score them with the training statistics of the model, and apply a shared
logistic regression trained on 82 labelled small-model cases
(``research/train.py --export`` writes ``artifacts/model.json``).

The artifact holds one fit per thinking budget. Problems are sampled in
batches under a time limit of ``SECONDS_PER_CASE`` per problem; when a batch
stops early, its traces are scored at the largest trained budget it reached,
cut to exactly that many tokens. A model without training statistics (e.g.
gpt-oss-120b) is z-scored over its own batch. Falls back to robust when the
model cannot be loaded, the budget never reaches the smallest trained budget,
or the run's time is nearly used up.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from transcript_probe.features import FEATURES, transcript_features
from transcript_probe.sampler import chat_prompt, load_model, problems_per_batch, release, sample

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "model.json"
SECONDS_PER_CASE = 150   # 3600 s over about 21 cases, minus model loads
TOTAL_SECONDS = 3600
RESERVE_SECONDS = 120
_START = time.time()


def _scores(fit: dict, model_id: str, rows: list[dict]) -> np.ndarray:
    raw = np.array([[r[name] for name in FEATURES] for r in rows], dtype=float)
    stats = fit["stats"].get(model_id)
    if stats is None:  # unseen model: normalise over this batch (no labels needed)
        mu, sd = raw.mean(axis=0), raw.std(axis=0)
    else:
        mu, sd = np.array(stats["mean"]), np.array(stats["sd"])
    sd = np.where(sd > 0, sd, 1.0)
    return ((raw - mu) / sd) @ np.array(fit["coef"]) + fit["intercept"]


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    if not problems:
        return []
    artifact = json.loads(ARTIFACT.read_text())
    n = artifact["n"]
    budgets = sorted(int(b) for b in artifact["fits"])
    try:
        tokenizer, model = load_model(model_id)
    except Exception:
        return [True for _ in problems]

    rows: list[dict | None] = [None] * len(problems)
    budget_of: list[int | None] = [None] * len(problems)
    try:
        prompts = [chat_prompt(tokenizer, p, reasoning_effort) for p in problems]
        prompt_tokens = max(len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts)
        size = problems_per_batch(model, n, budgets[-1], prompt_tokens)
        for start in range(0, len(problems), size):
            left = TOTAL_SECONDS - RESERVE_SECONDS - (time.time() - _START)
            batch = list(range(start, min(start + size, len(problems))))
            if left < 30:
                break
            try:
                samples, steps = sample(model, tokenizer, [prompts[i] for i in batch], n, budgets[-1],
                                        max_time=min(left, SECONDS_PER_CASE * len(batch)))
            except Exception:
                continue
            if all(done for traces in samples for _, done in traces):
                budget = budgets[-1]  # every trace finished: nothing was cut short
            else:
                reached = [b for b in budgets if b <= steps]
                if not reached:
                    continue
                budget = reached[-1]
            for i, traces in zip(batch, samples):
                cut = [(ids[:budget], done and len(ids) <= budget) for ids, done in traces]
                texts = tokenizer.batch_decode([ids for ids, _ in cut], skip_special_tokens=True)
                rows[i] = transcript_features(problems[i], texts, [done for _, done in cut])
                budget_of[i] = budget
    finally:
        del model
        release()

    predictions = [True] * len(problems)
    for budget in sorted({b for b in budget_of if b is not None}):
        idx = [i for i, b in enumerate(budget_of) if b == budget]
        fit = artifact["fits"][str(budget)]
        for i, s in zip(idx, _scores(fit, model_id, [rows[i] for i in idx])):
            predictions[i] = bool(s >= 0.0) if math.isfinite(s) else True
    return predictions
