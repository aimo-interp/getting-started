#!/usr/bin/env python3
"""Generate a roughly balanced set of robustness labels per model through OpenRouter.

Labels follow aimo-interp/train-main-v2: for each (model, problem),

    max_drop  = max over perturbation types of (accuracy on the original - accuracy on that type)
    is_robust = True if max_drop <= 0.10, False if max_drop >= 0.25, else None (not counted)

Problems whose original the model solves less than half the time get no label.

Per model, problems are processed in batches:

1. Originals: ``--base-samples`` samples each; keep problems solved at least half the time.
2. Variants: every variant of a kept problem gets ``--variant-samples`` samples, then the
   problem is labelled. Once one class reaches its quota, problems likely to fall in the
   other class are tried first, using the original's response length (long reasoning
   predicted "not robust" in our gpt-oss data).

It stops when both classes reach ``--per-class`` (default 10, so about 20 labels) or the
problem pool runs out. Every sample's full response and reasoning (chain of thought) is
saved to ``<out-dir>/samples/<model>__<effort>.traces.jsonl``; graded samples go to the
matching ``.jsonl``, and labels to ``<out-dir>/labels.jsonl``. Reruns resume.

Problems and variants come from ``data/labelgen/`` (``build_problems.py``, ``perturb.py``).

    export OPENROUTER_API_KEY=...
    python datagen/balanced_labels.py --models qwen/qwen3.5-9b qwen/qwen3-14b
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DEFAULT_WORK_DIR, append_jsonl, label_for, read_jsonl, write_jsonl  # noqa: E402
from sample import openrouter_sample, result_row, samples_path, trace_row, traces_path  # noqa: E402

MIN_BASE_ACCURACY = 0.5


class Sampler:
    """Runs OpenRouter samples for whole prompts in parallel and appends results."""

    def __init__(self, args, model: str, problems: dict, out: Path, key: str):
        self.args = argparse.Namespace(model=model, openrouter_model=None, effort=args.effort,
                                       max_tokens=args.max_tokens, quantizations=args.quantizations)
        self.problems, self.out, self.key, self.workers = problems, out, key, args.workers
        self.lock = threading.Lock()

    def run(self, jobs: list[tuple]) -> None:
        """jobs: (problem_id, type, variant, text, n). Writes one row per finished job."""
        if not jobs:
            return

        def one(job):
            try:
                results = [openrouter_sample(self.args, self.key, job[3]) for _ in range(job[4])]
            except Exception as exc:  # leave it unwritten; a rerun retries it
                print(f"    gave up on {job[0]} {job[1]}#{job[2]}: {str(exc)[:100]}", flush=True)
                return
            row = result_row(self.args, self.problems, job, [r[0] for r in results],
                             [r[1] for r in results], [r[2] for r in results])
            trace = trace_row(job, [r[3] for r in results], [r[4] for r in results], [r[5] for r in results])
            with self.lock:
                append_jsonl(traces_path(self.out), [trace])
                append_jsonl(self.out, [row])

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(one, jobs))


def state(out: Path):
    """correct[pid][type] -> list of bools, tokens[pid] -> original token counts, done prompt keys."""
    correct, tokens, done = defaultdict(lambda: defaultdict(list)), {}, set()
    for row in read_jsonl(out):
        correct[row["problem_id"]][row["type"]] += row["correct"]
        done.add((row["problem_id"], row["type"], row["variant"]))
        if row["type"] == "original":
            tokens[row["problem_id"]] = mean(row["n_tokens"]) if row["n_tokens"] else 0.0
    return correct, tokens, done


def label(by_type: dict, n_variant_prompts: int, n_done_variants: int):
    """(is_robust, max_drop, base) once all variants are sampled, else None."""
    base = mean(by_type["original"])
    if base < MIN_BASE_ACCURACY:
        return ("unsolved", None, base)
    if n_done_variants < n_variant_prompts:
        return None
    drops = [base - mean(v) for t, v in by_type.items() if t != "original" and v]
    max_drop = max(drops) if drops else 0.0
    return (label_for(max_drop), round(max_drop, 4), base)


def run_model(args, model: str, problems: dict, variants: dict, key: str) -> list[dict]:
    out = samples_path(args.out_dir / "samples", model, args.effort)
    sampler = Sampler(args, model, problems, out, key)
    order = sorted(problems)
    random.Random(f"{args.seed}:{model}").shuffle(order)
    quota = args.per_class

    def tally():
        correct, tokens, done = state(out)
        labels = {}
        for pid in correct:
            if "original" not in correct[pid]:
                continue
            n_done = sum((pid, t, v) in done for t, v, _ in variants.get(pid, []))
            result = label(correct[pid], len(variants.get(pid, [])), n_done)
            if result is not None:
                labels[pid] = result
        return correct, tokens, done, labels

    cursor, skipped = 0, set()
    while True:
        correct, tokens, done, labels = tally()
        robust = sum(1 for v in labels.values() if v[0] is True)
        fragile = sum(1 for v in labels.values() if v[0] is False)
        print(f"  {model}: robust {robust}/{quota}, not robust {fragile}/{quota}, "
              f"unlabelled band {sum(1 for v in labels.values() if v[0] is None)}, "
              f"unsolved {sum(1 for v in labels.values() if v[0] == 'unsolved')}", flush=True)
        if robust >= quota and fragile >= quota:
            break

        # Problems with originals sampled, solved, and variants still to do.
        pending = [pid for pid in correct if "original" in correct[pid] and pid not in labels
                   and pid not in skipped and mean(correct[pid]["original"]) >= MIN_BASE_ACCURACY]
        if pending:
            # Long original reasoning predicted "not robust"; aim at the class still short.
            pending.sort(key=lambda p: tokens.get(p, 0.0), reverse=fragile < quota <= robust)
            if robust < quota and fragile < quota:
                pending.sort(key=lambda p: order.index(p))
            # Once one class is full, label only about as many problems as the other still needs.
            short = quota - min(robust, fragile)
            size = args.batch if (robust < quota and fragile < quota) else max(2, min(args.batch, short))
            batch = pending[:size]
            jobs = [(pid, t, v, text, args.variant_samples) for pid in batch
                    for t, v, text in variants.get(pid, []) if (pid, t, v) not in done]
            print(f"    sampling variants for {len(batch)} problems ({len(jobs)} prompts)", flush=True)
            sampler.run(jobs)
            if not jobs or len(state(out)[2]) == len(done):  # no progress: requests keep failing
                skipped.update(batch)
                print(f"    no progress on {batch}; skipping them this run", flush=True)
            continue

        # Need more candidates: sample originals for the next batch of unseen problems.
        fresh = []
        while cursor < len(order) and len(fresh) < args.batch:
            pid = order[cursor]
            cursor += 1
            if (pid, "original", 0) not in done:
                fresh.append(pid)
        if not fresh:
            print(f"  {model}: problem pool exhausted", flush=True)
            break
        print(f"    sampling originals for {len(fresh)} problems", flush=True)
        sampler.run([(pid, "original", 0, problems[pid]["problem"], args.base_samples) for pid in fresh])

    rows = []
    for pid, (is_robust, max_drop, base) in labels.items():
        if is_robust == "unsolved":
            continue
        rows.append({"model_id": model, "reasoning_effort": args.effort, "problem": problems[pid]["problem"],
                     "is_robust": is_robust, "max_drop": max_drop, "problem_id": pid,
                     "source": problems[pid].get("source"), "base_accuracy": base,
                     "type_accuracy": {t: mean(v) for t, v in correct[pid].items() if t != "original"}})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True, help="OpenRouter model ids.")
    parser.add_argument("--effort", default="default", choices=["default", "low", "medium", "high"])
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR, help="Holds problems.jsonl, variants.jsonl.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_WORK_DIR / "balanced")
    parser.add_argument("--per-class", type=int, default=10, help="Labels wanted per class per model.")
    parser.add_argument("--base-samples", type=int, default=8)
    parser.add_argument("--variant-samples", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--batch", type=int, default=8, help="Problems per round.")
    parser.add_argument("--workers", type=int, default=32, help="Parallel OpenRouter requests.")
    parser.add_argument("--quantizations", nargs="*", default=["bf16", "fp8"],
                        help="Allowed provider precisions; pass none to allow any.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY to an OpenRouter key")
    problems = {p["problem_id"]: p for p in read_jsonl(args.work_dir / "problems.jsonl")}
    if not problems:
        raise SystemExit(f"no problems in {args.work_dir}; run datagen/build_problems.py and perturb.py first")
    variants = defaultdict(list)
    for v in read_jsonl(args.work_dir / "variants.jsonl"):
        if v["problem_id"] in problems:
            variants[v["problem_id"]].append((v["type"], v["variant"], v["text"]))

    labels_path = args.out_dir / "labels.jsonl"
    kept = [r for r in read_jsonl(labels_path) if r["model_id"] not in args.models
            or r["reasoning_effort"] != args.effort]
    for model in args.models:
        print(f"== {model} ({args.effort})", flush=True)
        kept += run_model(args, model, problems, variants, key)
        write_jsonl(labels_path, kept)
    counts = defaultdict(lambda: [0, 0, 0])
    for r in kept:
        counts[(r["model_id"], r["reasoning_effort"])][{True: 0, False: 1, None: 2}[r["is_robust"]]] += 1
    for (m, e), (t, f, u) in sorted(counts.items()):
        print(f"{m} {e}: robust {t}, not robust {f}, unlabelled band {u}")
    print(f"wrote {labels_path}")


if __name__ == "__main__":
    main()
