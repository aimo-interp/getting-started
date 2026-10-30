#!/usr/bin/env python3
"""Robustness labels for a local model with vLLM, uncapped generations (loop 2026-10-31).

The earlier 8k runs truncated 63-96% of samples (truncated = wrong), so most problems looked unsolved
and labels were biased. This driver samples at ``--max-tokens`` 32,768 with the labelgen prompt
(problem + SOLVE_SUFFIX, the cvbench format), 8 samples per original and 4 per variant, in rounds:

1. originals for the next ``--batch`` unseen problems;
2. every variant of each problem solved at least half the time, then its label
   (``datagen/balanced_labels.label``: max drop over types; <= .10 robust, >= .25 not robust).

Stops at ``--target-labels`` labelled problems (robust + not robust) or when the pool runs out.
Rows go to ``<out-dir>/samples/<model>__default.jsonl`` (+ ``.traces.jsonl``) in ``sample.py``'s format
and labels to ``<out-dir>/labels_<model>.jsonl`` (train-main-v2 schema); reruns resume at round granularity.

    source venvs/vllm-q35/bin/activate
    python datagen/label_local.py --model Skywork/Skywork-OR1-Math-7B
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))

from balanced_labels import MIN_BASE_ACCURACY, label, state  # noqa: E402
from common import DEFAULT_WORK_DIR, SOLVE_SUFFIX, append_jsonl, extract_answer, read_jsonl, tokenizer_path, write_jsonl  # noqa: E402
from sample import result_row, samples_path, trace_row, traces_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_WORK_DIR / "l32k")
    ap.add_argument("--target-labels", type=int, default=100)
    ap.add_argument("--base-samples", type=int, default=8)
    ap.add_argument("--variant-samples", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=16, help="Problems whose originals are sampled per round.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.effort = "default"
    problems = {p["problem_id"]: p for p in read_jsonl(args.work_dir / "problems.jsonl")}
    variants = defaultdict(list)
    for v in read_jsonl(args.work_dir / "variants.jsonl"):
        if v["problem_id"] in problems:
            variants[v["problem_id"]].append((v["type"], v["variant"], v["text"]))
    out = samples_path(args.out_dir / "samples", args.model, args.effort)
    out.parent.mkdir(parents=True, exist_ok=True)
    order = sorted(problems)
    random.Random(f"{args.seed}:{args.model}").shuffle(order)

    from vllm import LLM, SamplingParams
    llm = LLM(args.model, tokenizer=tokenizer_path(args.model), seed=args.seed, max_model_len=args.max_tokens + 2048,
              gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=args.max_num_seqs)
    tokenizer = llm.get_tokenizer()

    def run(jobs: list[tuple]) -> None:
        """jobs: (problem_id, type, variant, text, n). One vLLM call; rows appended when all finish."""
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": text + SOLVE_SUFFIX}], tokenize=False,
                                                 add_generation_prompt=True) for _, _, _, text, _ in jobs]
        params = [SamplingParams(n=n, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                                 seed=args.seed) for *_, n in jobs]
        t0 = time.time()
        rows, traces, n_tok, n_cap = [], [], 0, 0
        for job, output in zip(jobs, llm.generate(prompts, params)):
            answers = [extract_answer(c.text) if c.finish_reason == "stop" else None for c in output.outputs]
            lens = [len(c.token_ids) for c in output.outputs]
            n_tok += sum(lens)
            n_cap += sum(c.finish_reason != "stop" for c in output.outputs)
            rows.append(result_row(args, problems, job, answers, lens))
            traces.append(trace_row(job, [c.text for c in output.outputs],
                                    finish_reasons=[c.finish_reason for c in output.outputs]))
        append_jsonl(traces_path(out), traces)
        append_jsonl(out, rows)
        dt = time.time() - t0
        print(f"    {len(jobs)} prompts, {sum(j[4] for j in jobs)} samples, {n_tok} tokens in {dt:.0f}s "
              f"({n_tok / max(dt, 1):.0f} tok/s), {n_cap} hit the cap", flush=True)

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
        return correct, done, labels

    cursor = 0
    while True:
        correct, done, labels = tally()
        robust = sum(1 for v in labels.values() if v[0] is True)
        fragile = sum(1 for v in labels.values() if v[0] is False)
        band = sum(1 for v in labels.values() if v[0] is None)
        unsolved = sum(1 for v in labels.values() if v[0] == "unsolved")
        print(f"  {args.model}: robust {robust}, not robust {fragile}, band {band}, unsolved {unsolved}, "
              f"originals sampled {sum('original' in c for c in correct.values())}/{len(problems)}", flush=True)
        if robust + fragile >= args.target_labels:
            print(f"  {args.model}: target reached", flush=True)
            break
        pending = [pid for pid in correct if "original" in correct[pid] and pid not in labels
                   and mean(correct[pid]["original"]) >= MIN_BASE_ACCURACY]
        jobs = [(pid, t, v, text, args.variant_samples) for pid in pending
                for t, v, text in variants.get(pid, []) if (pid, t, v) not in done]
        if jobs:
            print(f"    sampling variants for {len(pending)} problems ({len(jobs)} prompts)", flush=True)
            run(jobs)
            continue
        fresh = []
        while cursor < len(order) and len(fresh) < args.batch:
            pid = order[cursor]
            cursor += 1
            if (pid, "original", 0) not in done:
                fresh.append(pid)
        if not fresh:
            print(f"  {args.model}: problem pool exhausted", flush=True)
            break
        print(f"    sampling originals for {len(fresh)} problems", flush=True)
        run([(pid, "original", 0, problems[pid]["problem"], args.base_samples) for pid in fresh])

    correct, done, labels = tally()
    rows = [{"model_id": args.model, "reasoning_effort": args.effort, "problem": problems[pid]["problem"],
             "is_robust": is_robust, "max_drop": max_drop, "problem_id": pid, "source": problems[pid].get("source"),
             "base_accuracy": base,
             "type_accuracy": {t: mean(v) for t, v in correct[pid].items() if t != "original"}}
            for pid, (is_robust, max_drop, base) in labels.items() if is_robust != "unsolved"]
    path = args.out_dir / f"labels_{samples_path(Path('.'), args.model, args.effort).stem}.jsonl"
    write_jsonl(path, rows)
    print(f"wrote {len(rows)} labels to {path}", flush=True)


if __name__ == "__main__":
    main()
