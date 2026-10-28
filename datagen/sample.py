#!/usr/bin/env python3
"""Sample a target model on every original problem and variant, and grade each answer.

Originals get ``--base-samples`` samples and each variant gets
``--variant-samples``, at temperature 1.0 with the model's own chat template.
Truncated generations count as wrong. Results are appended per prompt to
``<work-dir>/samples/<model>__<effort>.jsonl``, so an interrupted run resumes.
The full responses (and, for OpenRouter, the separate reasoning text) go to
``<model>__<effort>.traces.jsonl`` next to it, one line per prompt.
``--limit-problems N`` keeps the first N problems of ``problems.jsonl``
(AIME first, then AMC, then Omni-MATH-2).

Two backends:

* ``vllm`` (default): local GPU, run in a vLLM environment::

      python datagen/sample.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

* ``openrouter``: hosted API, no GPU. Reads the key from ``OPENROUTER_API_KEY``
  (or ``OPENAI_API_KEY``). Each sample is one request; ``--workers`` run in
  parallel and the serving provider is recorded per sample::

      python datagen/sample.py --backend openrouter --model openai/gpt-oss-120b --effort low
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DEFAULT_WORK_DIR, SOLVE_SUFFIX, append_jsonl, extract_answer, read_jsonl  # noqa: E402

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def samples_path(samples_dir: Path, model: str, effort: str) -> Path:
    return samples_dir / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', model)}__{effort}.jsonl"


def traces_path(out: Path) -> Path:
    return out.with_suffix(".traces.jsonl")


def trace_row(job, texts, reasonings=None, finish_reasons=None) -> dict:
    pid, kind, variant, _, _ = job
    row = {"problem_id": pid, "type": kind, "variant": variant, "texts": texts}
    if reasonings is not None:
        row["reasoning"] = reasonings
    if finish_reasons is not None:
        row["finish_reasons"] = finish_reasons
    return row


def result_row(args, problems, job, answers, n_tokens, providers=None) -> dict:
    pid, kind, variant, _, _ = job
    gold = problems[pid]["answer"]
    row = {
        "model_id": args.model, "reasoning_effort": args.effort,
        "problem_id": pid, "type": kind, "variant": variant,
        "answers": answers,
        "correct": [answer == gold for answer in answers],
        "n_tokens": n_tokens,
    }
    if providers is not None:
        row["providers"] = providers
    return row


def run_vllm(args, problems, jobs, out) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(args.model, tensor_parallel_size=args.tensor_parallel_size, seed=args.seed,
              max_model_len=args.max_tokens + 2048, gpu_memory_utilization=args.gpu_memory_utilization)
    tokenizer = llm.get_tokenizer()
    template_kwargs = {"reasoning_effort": args.effort} if args.effort != "default" else {}

    for start in range(0, len(jobs), args.chunk_size):
        chunk = jobs[start : start + args.chunk_size]
        prompts = [
            tokenizer.apply_chat_template([{"role": "user", "content": text + SOLVE_SUFFIX}],
                                          tokenize=False, add_generation_prompt=True, **template_kwargs)
            for _, _, _, text, _ in chunk
        ]
        params = [
            SamplingParams(n=n, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=args.max_tokens,
                           seed=args.seed)
            for *_, n in chunk
        ]
        rows, traces = [], []
        for job, output in zip(chunk, llm.generate(prompts, params)):
            answers = [
                extract_answer(c.text) if c.finish_reason == "stop" else None for c in output.outputs
            ]
            rows.append(result_row(args, problems, job, answers, [len(c.token_ids) for c in output.outputs]))
            traces.append(trace_row(job, [c.text for c in output.outputs],
                                    finish_reasons=[c.finish_reason for c in output.outputs]))
        append_jsonl(traces_path(out), traces)
        append_jsonl(out, rows)
        print(f"  {min(start + args.chunk_size, len(jobs))}/{len(jobs)} prompts", flush=True)


def openrouter_sample(args, key: str, text: str) -> tuple[int | None, int, str, str, str, str]:
    """One sample: (answer, completion tokens, provider, content, reasoning, finish reason).

    Retries transient errors.
    """
    body = {
        "model": args.openrouter_model or args.model,
        "messages": [{"role": "user", "content": text + SOLVE_SUFFIX}],
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
    }
    if args.effort != "default":
        body["reasoning"] = {"effort": args.effort}
    if getattr(args, "quantizations", None):
        # Only route to providers serving these precisions (e.g. bf16, fp8).
        body["provider"] = {"quantizations": list(args.quantizations)}
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                reply = json.load(response)
            if "choices" not in reply:
                raise RuntimeError(str(reply.get("error", reply))[:200])
            choice = reply["choices"][0]
            content = choice["message"].get("content") or ""
            reasoning = choice["message"].get("reasoning") or ""
            finish = str(choice.get("finish_reason") or "")
            answer = extract_answer(content) if finish == "stop" else None
            tokens = int((reply.get("usage") or {}).get("completion_tokens") or 0)
            return answer, tokens, str(reply.get("provider", "")), content, reasoning, finish
        except (urllib.error.URLError, TimeoutError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
            if attempt == 5:
                raise
            print(f"  retry {attempt + 1} after error: {str(exc)[:120]}", flush=True)
            time.sleep(min(60, 2 ** attempt * 5))
    raise AssertionError("unreachable")


def run_openrouter(args, problems, jobs, out) -> None:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY (or OPENAI_API_KEY) to an OpenRouter key")
    lock, done, failed = threading.Lock(), [0], [0]

    def run_job(job) -> None:
        try:
            results = [openrouter_sample(args, key, job[3]) for _ in range(job[4])]
        except Exception as exc:  # leave the prompt unwritten so a rerun retries it
            with lock:
                failed[0] += 1
            print(f"  gave up on {job[0]} {job[1]}: {str(exc)[:120]}", flush=True)
            return
        row = result_row(args, problems, job, [r[0] for r in results], [r[1] for r in results],
                         [r[2] for r in results])
        trace = trace_row(job, [r[3] for r in results], [r[4] for r in results], [r[5] for r in results])
        with lock:
            append_jsonl(traces_path(out), [trace])
            append_jsonl(out, [row])
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == len(jobs):
                print(f"  {done[0]}/{len(jobs)} prompts", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_job, jobs))
    if failed[0]:
        print(f"{failed[0]} prompts failed; rerun the same command to retry them")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model id recorded in the labels.")
    parser.add_argument("--effort", default="default", choices=["default", "low", "medium", "high"])
    parser.add_argument("--backend", default="vllm", choices=["vllm", "openrouter"])
    parser.add_argument("--openrouter-model", help="OpenRouter model id if it differs from --model.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--samples-dir", type=Path, help="Where samples are written (default: <work-dir>/samples).")
    parser.add_argument("--limit-problems", type=int, help="Only the first N problems.")
    parser.add_argument("--base-samples", type=int, default=8)
    parser.add_argument("--variant-samples", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--chunk-size", type=int, default=64, help="Prompts per vLLM call (resume granularity).")
    parser.add_argument("--workers", type=int, default=32, help="Parallel OpenRouter prompts.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problems = {row["problem_id"]: row for row in read_jsonl(args.work_dir / "problems.jsonl")}
    if args.limit_problems:
        problems = dict(list(problems.items())[: args.limit_problems])
    jobs = [(pid, "original", 0, row["problem"], args.base_samples) for pid, row in problems.items()]
    jobs += [
        (row["problem_id"], row["type"], row["variant"], row["text"], args.variant_samples)
        for row in read_jsonl(args.work_dir / "variants.jsonl")
        if row["problem_id"] in problems
    ]
    out = samples_path(args.samples_dir or args.work_dir / "samples", args.model, args.effort)
    done = {(row["problem_id"], row["type"], row["variant"]) for row in read_jsonl(out)}
    jobs = [job for job in jobs if job[:3] not in done]
    print(f"{args.model} ({args.effort}, {args.backend}): {len(problems)} problems, "
          f"{len(jobs)} prompts to sample, {len(done)} already done -> {out}", flush=True)
    if jobs:
        (run_vllm if args.backend == "vllm" else run_openrouter)(args, problems, jobs, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
