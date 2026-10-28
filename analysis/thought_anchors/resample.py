#!/usr/bin/env python3
"""Thought Anchors and transplant resampling through OpenRouter.

For one problem X and one perturbed version dX (from ``data/labelgen``), take a
solved reasoning trace on X and a failed one on dX (from the saved traces),
split each into sentences, and at prefix points k resample continuations of
the model's own reasoning:

* ``X``          prompt X  + first k sentences of the X trace    -> P(correct | prefix)
* ``dX``         prompt dX + first k sentences of the dX trace   -> P(correct | prefix)
* ``transplant`` prompt dX + first k sentences of the X trace    -> P(correct | prefix)

Continuations use OpenRouter's raw completion endpoint: the prompt is the
model's own chat template up to the start of the reasoning channel, followed
by the prefix text, so the model keeps reasoning from exactly that point.
Answers are graded against the gold answer; cut-off continuations count as wrong
(as in the label construction).

    export OPENROUTER_API_KEY=...
    python analysis/thought_anchors/resample.py check --effort low
    python analysis/thought_anchors/resample.py run --problem-id 19327236af77 --variant typos#0 --effort low
    python analysis/thought_anchors/resample.py run --problem-id 12f7caa1a71e --variant typos#0 --effort low

Train-set cases (``cases.jsonl``: problem, gold answer, label, effort) have no
saved traces, so ``case`` first screens X and its 4 code-made perturbations
(``--screen-n`` samples each), picks the perturbation that hurts most, takes a
solved X trace and a failed dX trace from the screen, then runs the curves::

    python analysis/thought_anchors/resample.py case --case fragile_ninths
    python analysis/thought_anchors/resample.py case --case robust_erased

Outputs go to ``data/thought_anchors/<problem>__<variant>__<effort>/``:
``continuations.jsonl`` (every continuation's full text; reruns resume),
``curves.csv``, ``summary.json`` (the sentence at the largest jump of each
curve, earliest point reaching 0.8), and ``curves.png`` if matplotlib is
available.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "datagen"), str(ROOT / "solutions" / "perturb-probe")]

from common import SOLVE_SUFFIX, extract_answer, read_jsonl  # noqa: E402

COMPLETIONS_URL = "https://openrouter.ai/api/v1/completions"
LABELGEN = ROOT / "data" / "labelgen"
CASES = Path(__file__).resolve().parent / "cases.jsonl"
TYPE_ORDER = ("distract", "typos", "typos2", "rename", "rephrase")
OUT = ROOT / "data" / "thought_anchors"
TOKENIZERS = {"openai/gpt-oss-120b": "openai/gpt-oss-120b", "qwen/qwen3.5-9b": "Qwen/Qwen3.5-9B",
              "qwen/qwen3-14b": "Qwen/Qwen3-14B"}


# ---------------------------------------------------------------------------
# Prompt construction: the model's chat template, opened into its reasoning
# ---------------------------------------------------------------------------

def reasoning_prompt(tokenizer, problem_text: str, effort: str) -> str:
    """Chat-formatted prompt that ends where the model's reasoning text begins."""
    kwargs = {"reasoning_effort": effort} if effort in ("low", "medium", "high") else {}
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": problem_text + SOLVE_SUFFIX}],
                                           tokenize=False, add_generation_prompt=True, **kwargs)
    if "<|channel|>" in tokenizer.get_vocab():  # gpt-oss (harmony): open the analysis channel
        return prompt + "<|channel|>analysis<|message|>"
    if not prompt.rstrip().endswith("<think>"):
        prompt += "<think>\n"
    return prompt


def reasoning_of(completion: str) -> str:
    """The reasoning part of a raw completion started inside the reasoning channel."""
    for marker in ("<|end|>", "</think>"):
        if marker in completion:
            return completion.split(marker, 1)[0]
    # Providers often strip special tokens, leaving the harmony boundary as plain text.
    match = re.search(r"assistant\s*final", completion)
    return completion[: match.start()] if match else completion


def split_sentences(text: str) -> list[str]:
    """Sentence-ish chunks: split after . ! ? or blank lines; merge very short pieces."""
    pieces = re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip())
    out = []
    for piece in pieces:
        if out and len(piece) < 25:
            out[-1] += " " + piece
        elif piece:
            out.append(piece)
    return out


def prefix_points(n_sentences: int, points: int) -> list[int]:
    """Evenly spaced numbers of leading sentences, always including 0 and all."""
    if n_sentences <= points:
        return list(range(n_sentences + 1))
    return sorted({round(i * n_sentences / (points - 1)) for i in range(points)})


# ---------------------------------------------------------------------------
# OpenRouter raw completions
# ---------------------------------------------------------------------------

PROVIDER: list[str] = []  # set from --provider; empty means OpenRouter routes freely


def complete(key: str, model: str, prompt: str, max_tokens: int, quantizations) -> tuple[str, str, str]:
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 1.0, "top_p": 1.0}
    provider = {}
    if quantizations:
        provider["quantizations"] = list(quantizations)
    if PROVIDER:
        provider.update({"order": list(PROVIDER), "allow_fallbacks": False})
    if provider:
        body["provider"] = provider
    request = urllib.request.Request(COMPLETIONS_URL, data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                reply = json.load(response)
            if "choices" not in reply:
                raise RuntimeError(str(reply.get("error", reply))[:300])
            choice = reply["choices"][0]
            return choice.get("text") or "", str(choice.get("finish_reason") or ""), str(reply.get("provider", ""))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code in (400, 401, 403, 404) or attempt == 5:
                raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
            time.sleep(min(60, 2 ** attempt * 5))
        except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError):
            if attempt == 5:
                raise
            time.sleep(min(60, 2 ** attempt * 5))
    raise AssertionError("unreachable")


def grade(continuation: str, finish: str, gold: int) -> tuple[int | None, bool]:
    if finish not in ("stop", "eos", "end_turn", ""):  # cut off by max_tokens
        return None, False
    answer = extract_answer(continuation)
    return answer, answer == gold


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def load_tokenizer(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZERS.get(model, model))


def cmd_check(args, key: str) -> None:
    """One cheap continuation from a hand-written prefix, to see that raw prompts work."""
    tokenizer = load_tokenizer(args.model)
    prompt = reasoning_prompt(tokenizer, "What is 17 times 23?", args.effort) + "We need 17*23. 17*20 = 340, and 17*3 ="
    text, finish, provider = complete(key, args.model, prompt, 400, args.quantizations)
    print(f"provider {provider or '?'} | finish {finish}\n--- prompt tail ---\n{prompt[-200:]}\n--- continuation ---\n{text[:1200]}")
    answer = extract_answer(text)
    print(f"\nparsed answer: {answer} (expected 391)")
    if text.lstrip().startswith("51"):
        print("OK: the continuation picks up mid-sentence ('51 ...'), so the prefix is honoured.")
    else:
        print("NOT CONTINUED: the text should start with '51' (finishing '17*3 ='); this provider restarts instead.")


def pick_trace(effort: str, model: str, problem_id: str, kind: str, variant: int, want_correct: bool) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    samples = {(s["problem_id"], s["type"], s["variant"]): s
               for s in read_jsonl(LABELGEN / "samples" / f"{safe}__{effort}.jsonl")}
    traces = {(t["problem_id"], t["type"], t["variant"]): t
              for t in read_jsonl(LABELGEN / "samples" / f"{safe}__{effort}.traces.jsonl")}
    s, t = samples.get((problem_id, kind, variant)), traces.get((problem_id, kind, variant))
    if not s or not t:
        raise SystemExit(f"no saved trace for {problem_id} {kind}#{variant} at {effort}")
    for i, (ok, finish) in enumerate(zip(s["correct"], t["finish_reasons"])):
        if ok == want_correct and finish == "stop" and (want_correct or s["answers"][i] is not None):
            reasoning = (t.get("reasoning") or [""] * len(t["texts"]))[i]
            if reasoning.strip():
                return reasoning
    raise SystemExit(f"no {'correct' if want_correct else 'genuinely wrong'} finished trace for {problem_id} {kind}#{variant}")


def cmd_run(args, key: str) -> None:
    problems = {p["problem_id"]: p for p in read_jsonl(LABELGEN / "problems.jsonl")}
    variants = {(v["problem_id"], v["type"], v["variant"]): v["text"] for v in read_jsonl(LABELGEN / "variants.jsonl")}
    kind, number = args.variant.split("#")
    problem = problems[args.problem_id]
    x_text, dx_text, gold = problem["problem"], variants[(args.problem_id, kind, int(number))], problem["answer"]

    x_trace = pick_trace(args.effort, args.model, args.problem_id, "original", 0, True)
    try:
        dx_trace = pick_trace(args.effort, args.model, args.problem_id, kind, int(number), False)
    except SystemExit:  # robust case: no failed trace on dX, use a solved one
        dx_trace = pick_trace(args.effort, args.model, args.problem_id, kind, int(number), True)
    x_sent, dx_sent = split_sentences(x_trace), split_sentences(dx_trace)

    tokenizer = load_tokenizer(args.model)
    x_prompt, dx_prompt = reasoning_prompt(tokenizer, x_text, args.effort), reasoning_prompt(tokenizer, dx_text, args.effort)
    curves = {
        "X": (x_prompt, x_sent),
        "dX": (dx_prompt, dx_sent),
        "transplant": (dx_prompt, x_sent),
    }
    out_dir = args.out_dir / f"{args.problem_id}__{kind}{number}__{args.effort}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "case.json").write_text(json.dumps({"model": args.model, "effort": args.effort, "problem_id": args.problem_id,
                                                   "variant": args.variant, "gold": gold, "x_text": x_text, "dx_text": dx_text,
                                                   "x_sentences": x_sent, "dx_sentences": dx_sent}, indent=1))
    path = out_dir / "continuations.jsonl"
    done = defaultdict(int)
    for row in read_jsonl(path):
        done[(row["curve"], row["k"])] += 1
    jobs = []
    for curve, (prompt, sentences) in curves.items():
        for k in prefix_points(len(sentences), args.points):
            for _ in range(max(0, args.resamples - done[(curve, k)])):
                jobs.append((curve, k, prompt + " ".join(sentences[:k]) + (" " if k else "")))
    print(f"{len(x_sent)} sentences on X, {len(dx_sent)} on dX | {len(jobs)} continuations to run "
          f"(about {len(jobs) * args.max_tokens / 2e6:.1f}M tokens at most half the cap each)", flush=True)

    lock = threading.Lock()
    counter = [0]

    def one(job):
        curve, k, prompt = job
        try:
            text, finish, provider = complete(key, args.model, prompt, args.max_tokens, args.quantizations)
        except Exception as exc:
            print(f"  failed {curve} k={k}: {str(exc)[:150]}", flush=True)
            return
        answer, correct = grade(text, finish, gold)
        with lock:
            with path.open("a") as stream:
                stream.write(json.dumps({"curve": curve, "k": k, "answer": answer, "correct": correct,
                                         "finish": finish, "provider": provider, "text": text}) + "\n")
            counter[0] += 1
            if counter[0] % 25 == 0:
                print(f"  {counter[0]}/{len(jobs)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, jobs))
    summarise(out_dir, curves, gold)


def run_jobs(args, key, jobs, path: Path, gold: int, extra=lambda job: {}) -> None:
    """jobs: (curve, k, prompt). Appends graded continuations (full text) to path."""
    lock, counter = threading.Lock(), [0]

    def one(job):
        curve, k, prompt = job
        try:
            text, finish, provider = complete(key, args.model, prompt, args.max_tokens, args.quantizations)
        except Exception as exc:
            print(f"  failed {curve} k={k}: {str(exc)[:150]}", flush=True)
            return
        answer, correct = grade(text, finish, gold)
        with lock:
            with path.open("a") as stream:
                stream.write(json.dumps({"curve": curve, "k": k, "answer": answer, "correct": correct,
                                         "finish": finish, "provider": provider, "text": text, **extra(job)}) + "\n")
            counter[0] += 1
            if counter[0] % 25 == 0:
                print(f"  {counter[0]}/{len(jobs)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, jobs))


def cmd_case(args, key: str) -> None:
    """Train-set case: screen X and its 4 code perturbations, then run the curves."""
    from perturb_probe.perturbations import make_perturbations

    case = next((c for c in read_jsonl(args.cases) if c["case"] == args.case), None)
    if case is None:
        raise SystemExit(f"no case {args.case!r} in {args.cases}")
    args.model, args.effort, gold = case["model"], case["effort"], case["gold"]
    tokenizer = load_tokenizer(args.model)
    prompts = {"original": case["problem"], **{kind: text for kind, text in make_perturbations(case["problem"], 0)}}
    out_dir = args.out_dir / f"{args.case}__{args.effort}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Screen: full samples from an empty prefix on X and every perturbation.
    screen_path = out_dir / "screen.jsonl"
    have = defaultdict(int)
    for row in read_jsonl(screen_path):
        have[row["curve"]] += 1
    jobs = [(kind, 0, reasoning_prompt(tokenizer, text, args.effort))
            for kind, text in prompts.items() for _ in range(max(0, args.screen_n - have[kind]))]
    print(f"{args.case}: screening {len(jobs)} samples ({args.screen_n} per prompt, {len(prompts)} prompts)", flush=True)
    run_jobs(args, key, jobs, screen_path, gold)
    rows = list(read_jsonl(screen_path))
    acc = {kind: [r["correct"] for r in rows if r["curve"] == kind] for kind in prompts}
    for kind, values in acc.items():
        print(f"  {kind:9s} accuracy {sum(values)}/{len(values)}", flush=True)
    perturbed = [k for k in prompts if k != "original" and acc[k]]
    worst = min(perturbed, key=lambda k: (sum(acc[k]) / len(acc[k]), TYPE_ORDER.index(k) if k in TYPE_ORDER else 9))
    x_traces = [reasoning_of(r["text"]) for r in rows if r["curve"] == "original" and r["correct"]]
    wrong = [reasoning_of(r["text"]) for r in rows if r["curve"] == worst and not r["correct"]
             and r["answer"] is not None]
    right = [reasoning_of(r["text"]) for r in rows if r["curve"] == worst and r["correct"]]
    if not x_traces:
        raise SystemExit("no correct trace on X in the screen; raise --screen-n or check the gold answer")
    dx_trace, dx_kind = (wrong[0], "failed") if wrong else (right[0], "solved")
    print(f"  perturbation for the curves: {worst} (accuracy {sum(acc[worst])}/{len(acc[worst])}); "
          f"dX trace is a {dx_kind} one", flush=True)

    # 2. Curves from sentence prefixes.
    x_sent, dx_sent = split_sentences(x_traces[0]), split_sentences(dx_trace)
    x_prompt = reasoning_prompt(tokenizer, prompts["original"], args.effort)
    dx_prompt = reasoning_prompt(tokenizer, prompts[worst], args.effort)
    curves = {"X": (x_prompt, x_sent), "dX": (dx_prompt, dx_sent), "transplant": (dx_prompt, x_sent)}
    (out_dir / "case.json").write_text(json.dumps({**case, "perturbation": worst, "dx_text": prompts[worst],
                                                   "dx_trace_kind": dx_kind, "screen_accuracy": {k: sum(v) / len(v) for k, v in acc.items() if v},
                                                   "x_sentences": x_sent, "dx_sentences": dx_sent}, indent=1))
    path = out_dir / "continuations.jsonl"
    done = defaultdict(int)
    for row in read_jsonl(path):
        done[(row["curve"], row["k"])] += 1
    jobs = [(curve, k, prompt + " ".join(sentences[:k]) + (" " if k else ""))
            for curve, (prompt, sentences) in curves.items()
            for k in prefix_points(len(sentences), args.points)
            for _ in range(max(0, args.resamples - done[(curve, k)]))]
    print(f"curves: {len(x_sent)} sentences on X, {len(dx_sent)} on dX | {len(jobs)} continuations to run", flush=True)
    run_jobs(args, key, jobs, path, gold)
    summarise(out_dir, curves, gold)


def summarise(out_dir: Path, curves: dict, gold: int) -> None:
    acc = defaultdict(list)
    for row in read_jsonl(out_dir / "continuations.jsonl"):
        acc[(row["curve"], row["k"])].append(row["correct"])
    summary, table = {}, []
    for curve, (_, sentences) in curves.items():
        ks = sorted(k for c, k in acc if c == curve)
        if not ks:
            continue
        p = [sum(acc[(curve, k)]) / len(acc[(curve, k)]) for k in ks]
        for k, value in zip(ks, p):
            table.append({"curve": curve, "k": k, "fraction": round(k / max(len(sentences), 1), 3),
                          "p_correct": round(value, 3), "n": len(acc[(curve, k)])})
        jumps = [(p[i + 1] - p[i], ks[i], ks[i + 1]) for i in range(len(p) - 1)]
        biggest = max(jumps, key=lambda j: abs(j[0])) if jumps else (0.0, 0, 0)
        reach = next((k for k, v in zip(ks, p) if v >= 0.8), None)
        summary[curve] = {
            "p_correct": dict(zip(map(str, ks), [round(v, 3) for v in p])),
            "first_k_reaching_0.8": reach,
            "first_fraction_reaching_0.8": round(reach / len(sentences), 3) if reach is not None else None,
            "largest_jump": round(biggest[0], 3),
            "largest_jump_between_k": [biggest[1], biggest[2]],
            "sentences_in_largest_jump": sentences[biggest[1]:biggest[2]],
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    with (out_dir / "curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["curve", "k", "fraction", "p_correct", "n"])
        writer.writeheader()
        writer.writerows(table)
    for curve, s in summary.items():
        print(f"{curve:10s} P(correct) by k: {s['p_correct']}")
        print(f"{'':10s} reaches 0.8 at k={s['first_k_reaching_0.8']} | largest jump {s['largest_jump']:+.2f} "
              f"between k={s['largest_jump_between_k']}: {' '.join(s['sentences_in_largest_jump'])[:300]!r}")
    print(f"wrote {out_dir}/curves.csv, summary.json")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for curve in summary:
        rows = [r for r in table if r["curve"] == curve]
        ax.plot([r["fraction"] for r in rows], [r["p_correct"] for r in rows], marker="o", label=curve)
    ax.set(xlabel="fraction of the trace given as prefix", ylabel="P(correct)", ylim=(-0.03, 1.03),
           title=out_dir.name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=130)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["check", "run", "case"])
    parser.add_argument("--case", help="Case name in --cases (for the 'case' command).")
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--screen-n", type=int, default=8, help="Samples per prompt when screening a case.")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--effort", default="low", choices=["default", "low", "medium", "high"])
    parser.add_argument("--problem-id")
    parser.add_argument("--variant", help="TYPE#INDEX of the perturbed version, e.g. typos#0")
    parser.add_argument("--points", type=int, default=16, help="Prefix points per curve.")
    parser.add_argument("--resamples", type=int, default=10, help="Continuations per prefix point.")
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--quantizations", nargs="*", default=[])
    parser.add_argument("--provider", nargs="*", default=[],
                        help="Pin OpenRouter provider(s), e.g. DeepInfra, so every sample comes from one deployment.")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PROVIDER[:] = args.provider
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY")
    if args.command == "check":
        cmd_check(args, key)
    elif args.command == "case":
        if not args.case:
            raise SystemExit("case needs --case NAME")
        cmd_case(args, key)
    else:
        if not (args.problem_id and args.variant):
            raise SystemExit("run needs --problem-id and --variant")
        cmd_run(args, key)


if __name__ == "__main__":
    main()
