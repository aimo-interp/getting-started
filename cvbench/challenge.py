#!/usr/bin/env python3
"""Self-challenge / chain-of-verification signal (the model as its own investigator).

Per case, up to 4 of the transcript probe's 16k samples of the original problem
(finished ones first; a truncated one gets a budget-forced answer). Each sample
becomes a finished first assistant turn (its own reasoning + answer A), followed by
a second user turn, one of:

* ``challenge``: "I think your answer might be wrong. Please re-check ..."
* ``cove``: "Before finalizing, list 3 verification questions ..."

One continuation per (sample, variant), T=0.6, 4,096 new tokens (budget-forced
``\\boxed{`` if cut). Features: share of samples whose answer survives each
challenge (``kept_challenge``, ``kept_cove``) and ``lp_A``, the mean log p of A's
tokens teacher-forced as "The final answer is \\boxed{A}" right after the challenge
turn (vLLM prompt_logprobs; NaN for gpt-oss). Output:
``data/cvbench/challenge/<model>/<group>.json``. Resumable.

    python cvbench/challenge.py --model Skywork/Skywork-OR1-Math-7B      # vLLM env
    python cvbench/challenge.py --oss                                     # gpt-oss-120b via OpenRouter
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "datagen"), str(ROOT / "solutions" / "transcript-probe")]

from common import SOLVE_SUFFIX, tokenizer_path  # noqa: E402
from transcript_probe.features import extract_answer  # noqa: E402

DATA = ROOT / "data" / "cvbench"
OUT = DATA / "challenge"
WORK = ROOT / "data" / "anchors"
SCREENS = {"Qwen/Qwen3.5-4B": WORK, "Skywork/Skywork-OR1-Math-7B": WORK / "Skywork_Skywork-OR1-Math-7B",
           "allenai/Olmo-3-7B-Think": WORK / "allenai_Olmo-3-7B-Think",
           "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": WORK / "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B"}
OSS = ROOT / "data" / "transcript_probe" / "oss"
PROMPTS = {
    "challenge": "I think your answer might be wrong. Please re-check your solution carefully and give your final "
                 "answer in \\boxed{}.",
    "cove": "Before finalizing, list 3 verification questions that would expose an error in your solution, answer "
            "each independently, then give your final answer in \\boxed{}.",
}
K = 4                 # samples per case
MAX_NEW = 4096
TEMP = 0.6
FORCE = "\n</think>\n\nThe final answer is \\boxed{"
MARK = "ZZMARKZZ"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def safe(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)


def clean(text: str) -> str:
    return text.replace("Ġ", " ").replace("Ċ", "\n") if "Ġ" in text else text


def load_cases(model_id: str) -> list[dict]:
    """Cases of one model with its original-problem screen row attached."""
    if model_id.startswith("openai/gpt-oss-120b:"):
        effort = model_id.split(":")[1]
        cases = [c for c in read_jsonl(DATA / "cases_oss.jsonl") if c["model_id"] == model_id]
        screen = {r["group"]: r for r in read_jsonl(OSS / f"screen_{effort}.jsonl")}
    else:
        cases = [c for c in read_jsonl(DATA / "cases.jsonl") if c["model_id"] == model_id]
        screen = {r["group"]: r for r in read_jsonl(SCREENS[model_id] / "screen.jsonl") if r["variant"] == "X"}
    for c in cases:
        c["screen"] = screen[c["group"]]
    return cases


def pick(case: dict) -> list[dict]:
    """Up to K samples: finished with an integer answer first, then truncated ones (answer forced later)."""
    x = case["screen"]
    answers = x.get("answers") or [None] * len(x["texts"])
    done, cut = [], []
    for text, finish, a in zip(x["texts"], x["finish"], answers):
        text = clean(text)
        a = a if a is not None else (extract_answer(text) if finish == "stop" else None)
        if finish == "stop" and a is not None:
            done.append({"text": text, "A": int(a), "forced": False})
        elif finish != "stop":
            cut.append({"text": text, "A": None, "forced": True})
    return (done + cut)[:K]


def summarize(samples: list[dict]) -> dict:
    out = {"n": len(samples), "n_forced": sum(s["forced"] for s in samples), "samples": samples}
    for v in PROMPTS:
        kept = [s[v] == s["A"] for s in samples if s.get("A") is not None and v in s]
        out[f"kept_{v}"] = sum(kept) / len(kept) if kept else None
        lens = [s[f"len_{v}"] for s in samples if f"len_{v}" in s]
        out[f"len_{v}"] = sum(math.log1p(t) for t in lens) / len(lens) if lens else None
    lps = [s["lp_A"] for s in samples if s.get("lp_A") is not None]
    out["lp_A"] = sum(lps) / len(lps) if lps else None
    return out


# ---------------------------------------------------------------- vLLM (small models)
def run_vllm(args) -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    out_dir = OUT / safe(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [c for c in load_cases(args.model) if not (out_dir / f"{c['group']}.json").is_file()]
    print(f"[challenge] {args.model}: {len(cases)} cases to do", flush=True)
    if not cases:
        return
    kwargs = dict(seed=0, max_model_len=16384 + MAX_NEW + 2048, enable_prefix_caching=True,
                  gpu_memory_utilization=args.gpu_memory_utilization, limit_mm_per_prompt={"image": 0, "video": 0},
                  tokenizer=tokenizer_path(args.model))
    try:
        llm = LLM(args.model, **kwargs)
    except TypeError:
        kwargs.pop("limit_mm_per_prompt")
        llm = LLM(args.model, **kwargs)
    tok = llm.get_tokenizer()
    head = lambda problem: tok.apply_chat_template([{"role": "user", "content": problem + SOLVE_SUFFIX}],
                                                   tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template([{"role": "user", "content": "P"}, {"role": "assistant", "content": MARK},
                                    {"role": "user", "content": "QQQ"}], tokenize=False, add_generation_prompt=True)
    turn = full[full.index(MARK) + len(MARK):]          # end of assistant turn + user turn + generation prompt
    pre_q, post_q = turn.split("QQQ")
    think_open = "" if post_q.rstrip().endswith("<think>") else "<think>\n"
    force_params = SamplingParams(temperature=0.0, max_tokens=16, stop=["}"])
    gen_params = SamplingParams(temperature=TEMP, top_p=0.95, max_tokens=MAX_NEW, seed=0)
    lp_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

    def forced_answers(prompts: list[str]) -> list[int | None]:
        return [extract_answer("\\boxed{" + o.outputs[0].text + "}") for o in llm.generate(prompts, force_params, use_tqdm=False)]

    for start in range(0, len(cases), args.chunk):
        chunk = cases[start:start + args.chunk]
        t0 = time.time()
        items = [(c, s) for c in chunk for s in (c.setdefault("picked", pick(c)))]
        # 1. budget-force answers of truncated samples; the trace becomes a finished turn
        cut = [(c, s) for c, s in items if s["forced"]]
        for (c, s), a in zip(cut, forced_answers([head(c["problem"]) + s["text"] + FORCE for c, s in cut])):
            s["A"] = a
            s["text"] = s["text"] + FORCE + (str(a) if a is not None else "") + "}"
        items = [(c, s) for c, s in items if s["A"] is not None]
        for c in chunk:
            c["picked"] = [s for s in c["picked"] if s["A"] is not None]
        # 2. challenge / cove continuations
        jobs = [(c, s, v, head(c["problem"]) + s["text"] + pre_q + PROMPTS[v] + post_q)
                for c, s in items for v in PROMPTS]
        outs = llm.generate([p for *_, p in jobs], gen_params, use_tqdm=False)
        refill = []
        for (c, s, v, p), o in zip(jobs, outs):
            text = o.outputs[0].text
            s[f"len_{v}"] = len(o.outputs[0].token_ids)
            a = extract_answer(text) if o.outputs[0].finish_reason == "stop" else None
            if a is None and "\\boxed" in text:
                a = extract_answer(text)
            s[v] = a
            s[f"text_{v}"] = text[-1500:]
            if a is None:
                refill.append((s, v, p + text + ("\n\nThe final answer is \\boxed{" if "</think>" in text else FORCE)))
        for (s, v, _), a in zip(refill, forced_answers([p for *_, p in refill])):
            s[v] = a
        # 3. log p(A) right after the challenge turn
        if not args.no_lp:
            lp_jobs = []
            for c, s in items:
                h = head(c["problem"]) + s["text"] + pre_q + PROMPTS["challenge"] + post_q + think_open
                h += "\n</think>\n\nThe final answer is \\boxed{"
                ids_h = tok(h, add_special_tokens=True)["input_ids"]
                ids_f = tok(h + str(s["A"]) + "}", add_special_tokens=True)["input_ids"]
                if ids_f[:len(ids_h)] == ids_h and len(ids_f) > len(ids_h) + 1:
                    lp_jobs.append((s, len(ids_h), len(ids_f) - 1, ids_f))
            try:
                outs = llm.generate([{"prompt_token_ids": ids} for *_, ids in lp_jobs], lp_params, use_tqdm=False)
                for (s, lo, hi, ids), o in zip(lp_jobs, outs):
                    s["lp_A"] = float(sum(o.prompt_logprobs[i][ids[i]].logprob for i in range(lo, hi)))
            except Exception as exc:  # noqa: BLE001
                print(f"[challenge] lp_A failed: {exc!r}; continuing without it", flush=True)
                args.no_lp = True
        for c in chunk:
            for s in c["picked"]:
                s.pop("text", None)
            (out_dir / f"{c['group']}.json").write_text(json.dumps({"model_id": args.model, "group": c["group"],
                                                                    **summarize(c["picked"])}))
        print(f"  {min(start + args.chunk, len(cases))}/{len(cases)} cases, {len(jobs)} gens, "
              f"{time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- OpenRouter (gpt-oss-120b)
def harmony_turn(text: str) -> tuple[str, str]:
    """(reasoning, final) from an OpenRouter raw completion whose special tokens were stripped."""
    for marker in ("<|end|><|start|>assistant<|channel|>final<|message|>", "assistantfinal"):
        if marker in text:
            r, f = text.split(marker, 1)
            return r, f
    return text, ""


def run_oss(args) -> None:
    from transformers import AutoTokenizer

    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
    spent = {"usd": 0.0}

    def complete(prompt: str, max_tokens: int, temperature: float) -> tuple[str, str]:
        body = {"model": "openai/gpt-oss-120b", "prompt": prompt, "max_tokens": max_tokens,
                "temperature": temperature, "top_p": 0.95, "usage": {"include": True},
                "provider": {"order": args.provider, "allow_fallbacks": False}}
        req = urllib.request.Request("https://openrouter.ai/api/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=900) as r:
                    reply = json.load(r)
                if "choices" not in reply:
                    raise RuntimeError(str(reply.get("error", reply))[:300])
                spent["usd"] += float((reply.get("usage") or {}).get("cost") or 0.0)
                ch = reply["choices"][0]
                return ch.get("text") or "", str(ch.get("finish_reason") or "")
            except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
                if attempt == 4:
                    print(f"[oss] giving up: {exc!r}"[:300], flush=True)
                    return "", "error"
                time.sleep(5 * 2 ** attempt)

    for effort in ("low", "medium", "high"):
        model_id = f"openai/gpt-oss-120b:{effort}"
        out_dir = OUT / safe(model_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        cases = [c for c in load_cases(model_id) if not (out_dir / f"{c['group']}.json").is_file()]
        print(f"[oss] {effort}: {len(cases)} cases to do; spent so far ${spent['usd']:.3f}", flush=True)
        for c in cases:
            if spent["usd"] > args.cap:
                print(f"[oss] spend cap ${args.cap} reached; stopping", flush=True)
                return
            picked = [s for s in pick(c) if not s["forced"]]
            base = tok.apply_chat_template([{"role": "user", "content": c["problem"] + SOLVE_SUFFIX}], tokenize=False,
                                           add_generation_prompt=True, reasoning_effort=effort)
            jobs = []
            for s in picked:
                reasoning, final = harmony_turn(s["text"])
                hist = (base + "<|channel|>analysis<|message|>" + reasoning + "<|end|><|start|>assistant<|channel|>final"
                        "<|message|>" + final + "<|end|>")
                for v in PROMPTS:
                    jobs.append((s, v, hist + "<|start|>user<|message|>" + PROMPTS[v] + "<|end|><|start|>assistant"
                                 "<|channel|>analysis<|message|>"))
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                outs = list(pool.map(lambda j: complete(j[2], MAX_NEW, TEMP), jobs))
            for (s, v, _), (text, finish) in zip(jobs, outs):
                s[f"len_{v}"] = len(tok(text, add_special_tokens=False)["input_ids"])
                _, final = harmony_turn(text)
                s[v] = extract_answer(final or text) if finish != "error" else None
                s[f"text_{v}"] = text[-1500:]
            for s in picked:
                s.pop("text", None)
            (out_dir / f"{c['group']}.json").write_text(json.dumps({"model_id": model_id, "group": c["group"],
                                                                    **summarize(picked)}))
        print(f"[oss] {effort} done; spent ${spent['usd']:.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--oss", action="store_true")
    ap.add_argument("--chunk", type=int, default=6, help="Cases per vLLM call (resume granularity).")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--no-lp", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cap", type=float, default=3.0, help="gpt-oss: stop once OpenRouter spend passes this (USD).")
    ap.add_argument("--provider", nargs="*", default=["Crusoe"])
    args = ap.parse_args()
    if args.oss:
        run_oss(args)
    else:
        run_vllm(args)


if __name__ == "__main__":
    main()
