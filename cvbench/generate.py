#!/usr/bin/env python3
"""GPU stage: cache every model-side signal the CV methods need.

Stages, each resumable and written under ``data/cvbench/<stage>/<model>.*``:

* ``greedy``: the uncertainty-profiling baseline's own generation (greedy,
  2,048 prompt / 4,096 new tokens, user-only chat template) with its 14
  features, plus the generated text.
* ``acts``: one forward pass per problem over prompt + greedy response,
  recording hidden states at every layer for the last prompt token, the mean
  over the response, and the last prompt token of each code-made perturbation
  (``perturb_probe.perturbations``); plus the per-token entropy and selected
  log-prob trace of the greedy response.
* ``pool``: prompt-only forward passes for X and its 4 perturbations, saving at
  every layer the mean over the problem-text tokens and over the last 4 chat-template
  tokens (``prob_mean``/``tmpl_last`` and their ``pert_*`` counterparts).
* ``sample``: the forced-answer perturbation probe (X + 4 perturbations,
  ``--n`` traces each, ``--budget`` thinking tokens, forced answers and
  answer log-probs).

    uv run cvbench/generate.py --stages greedy acts
    uv run cvbench/generate.py --stages sample --n 4 --budget 2048
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "solutions" / "uncertainty-profiling"), str(ROOT / "solutions" / "perturb-probe")]

from perturb_probe.perturbations import make_perturbations  # noqa: E402
from perturb_probe.probe import probe_problem  # noqa: E402
from uncertainty_profile.config import GenerationConfidenceConfig  # noqa: E402
from uncertainty_profile.extraction import (  # noqa: E402
    format_problem,
    iter_generation_feature_batches,
    load_model_and_tokenizer,
    release_model,
)

DATA = ROOT / "data" / "cvbench"
MODELS = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
          "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]


def safe(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def run_greedy(model, tokenizer, model_id: str, todo: dict[str, str], batch_size: int) -> None:
    out = DATA / "greedy" / f"{safe(model_id)}.jsonl"
    done = {r["group"] for r in read_jsonl(out)}
    groups = [g for g in todo if g not in done]
    print(f"[greedy] {model_id}: {len(groups)} to do ({len(done)} done)", flush=True)
    config = GenerationConfidenceConfig(batch_size=batch_size)
    for start, rows in iter_generation_feature_batches(model=model, tokenizer=tokenizer,
                                                       problem_texts=[todo[g] for g in groups],
                                                       config=config, include_generated_text=True):
        append_jsonl(out, [{"group": g, "model_id": model_id, **row}
                           for g, row in zip(groups[start:start + len(rows)], rows)])


@torch.no_grad()
def hidden_last(model, tokenizer, texts: list[str]) -> np.ndarray:
    """Last-token hidden states at every layer, [len(texts), L+1, d]."""
    device = model.get_input_embeddings().weight.device
    enc = tokenizer([format_problem(tokenizer, t) for t in texts], return_tensors="pt", padding=True,
                    truncation=True, max_length=2048).to(device)
    hs = model(**enc, output_hidden_states=True).hidden_states
    return torch.stack([h[:, -1] for h in hs], dim=1).float().cpu().numpy()


@torch.no_grad()
def run_acts(model, tokenizer, model_id: str, todo: dict[str, str]) -> None:
    out_dir = DATA / "acts" / safe(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    greedy = {r["group"]: r for r in read_jsonl(DATA / "greedy" / f"{safe(model_id)}.jsonl")}
    groups = [g for g in todo if not (out_dir / f"{g}.npz").is_file()]
    print(f"[acts] {model_id}: {len(groups)} to do", flush=True)
    device = model.get_input_embeddings().weight.device
    for k, g in enumerate(groups, 1):
        start = time.time()
        prompt = format_problem(tokenizer, todo[g])
        prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=2048)["input_ids"]
        response = greedy.get(g, {}).get("generated_text", "")
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"][:4096]
        ids = torch.tensor([prompt_ids + response_ids], device=device)
        outputs = model(input_ids=ids, output_hidden_states=True)
        p = len(prompt_ids)
        hs = outputs.hidden_states
        last_prompt = torch.stack([h[0, p - 1] for h in hs]).float().cpu().numpy()
        if response_ids:
            resp_mean = torch.stack([h[0, p:].float().mean(0) for h in hs]).cpu().numpy()
            logp = torch.log_softmax(outputs.logits[0, p - 1:-1].float(), dim=-1)
            entropy = -(logp.exp() * logp).nan_to_num().sum(-1).cpu().numpy()
            selected = logp.gather(-1, ids[0, p:].unsqueeze(-1)).squeeze(-1).cpu().numpy()
        else:
            resp_mean = np.zeros_like(last_prompt)
            entropy = selected = np.zeros(0, dtype=np.float32)
        del outputs, hs
        variants = make_perturbations(todo[g], seed=0)
        pert = hidden_last(model, tokenizer, [t for _, t in variants])
        np.savez_compressed(out_dir / f"{g}.npz", last_prompt=last_prompt.astype(np.float16),
                            resp_mean=resp_mean.astype(np.float16), pert_last=pert.astype(np.float16),
                            pert_types=np.array([kind for kind, _ in variants]),
                            entropy=entropy.astype(np.float16), selected_logp=selected.astype(np.float16))
        torch.cuda.empty_cache()
        print(f"  {k}/{len(groups)} {g}: {len(response_ids)} resp tokens, {time.time() - start:.1f}s", flush=True)


@torch.no_grad()
def pooled(model, tokenizer, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per text: mean over the problem-text tokens and mean over the last 4 template tokens, [n, L+1, d] each."""
    device = model.get_input_embeddings().weight.device
    prob, tmpl = [], []
    for text in texts:
        prompt = format_problem(tokenizer, text)
        start = prompt.find(text)
        enc = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=2048,
                        return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0]
        span = ((offsets[:, 0] >= start) & (offsets[:, 1] <= start + len(text)) & (offsets[:, 1] > offsets[:, 0]))
        if start < 0 or not span.any():
            span = torch.ones(len(offsets), dtype=torch.bool)
        hs = model(**enc.to(device), output_hidden_states=True).hidden_states
        span = span.to(device)
        prob.append(torch.stack([h[0, span].float().mean(0) for h in hs]).cpu().numpy())
        tmpl.append(torch.stack([h[0, -4:].float().mean(0) for h in hs]).cpu().numpy())
    return np.stack(prob), np.stack(tmpl)


def run_pool(model, tokenizer, model_id: str, todo: dict[str, str]) -> None:
    out_dir = DATA / "pool" / safe(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = [g for g in todo if not (out_dir / f"{g}.npz").is_file()]
    print(f"[pool] {model_id}: {len(groups)} to do", flush=True)
    for k, g in enumerate(groups, 1):
        variants = make_perturbations(todo[g], seed=0)
        prob, tmpl = pooled(model, tokenizer, [todo[g]] + [t for _, t in variants])
        np.savez_compressed(out_dir / f"{g}.npz", prob_mean=prob[0].astype(np.float16), tmpl_last=tmpl[0].astype(np.float16),
                            pert_prob_mean=prob[1:].astype(np.float16), pert_tmpl_last=tmpl[1:].astype(np.float16),
                            pert_types=np.array([kind for kind, _ in variants]))
        print(f"  {k}/{len(groups)} {g}", flush=True)


def run_sample(model, tokenizer, model_id: str, todo: dict[str, str], n: int, budget: int) -> None:
    out = DATA / "sample" / f"{safe(model_id)}__n{n}_b{budget}.jsonl"
    done = {r["group"] for r in read_jsonl(out)}
    groups = [g for g in todo if g not in done]
    print(f"[sample] {model_id}: {len(groups)} to do ({len(done)} done)", flush=True)
    for k, g in enumerate(groups, 1):
        start = time.time()
        features = probe_problem(model, tokenizer, model_id, "default", todo[g], n=n, budget=budget, seed=0)
        features.pop("x_raw", None)
        append_jsonl(out, [{"group": g, "model_id": model_id, "n": n, "budget": budget,
                            "seconds": round(time.time() - start, 1), **features}])
        print(f"  {k}/{len(groups)} {g}: ref {features['reference']} agree {features.get('agree_mean')} "
              f"{time.time() - start:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", default=["greedy", "acts"], choices=["greedy", "acts", "sample", "pool"])
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--budget", type=int, default=2048)
    args = parser.parse_args()

    cases = read_jsonl(DATA / "cases.jsonl")
    for model_id in args.models:
        todo = {c["group"]: c["problem"] for c in cases if c["model_id"] == model_id}
        if not todo:
            continue
        model, tokenizer = load_model_and_tokenizer(model_id, local_files_only=True)
        try:
            for stage in args.stages:
                if stage == "greedy":
                    run_greedy(model, tokenizer, model_id, todo, args.batch_size)
                elif stage == "acts":
                    run_acts(model, tokenizer, model_id, todo)
                elif stage == "pool":
                    run_pool(model, tokenizer, model_id, todo)
                else:
                    run_sample(model, tokenizer, model_id, todo, args.n, args.budget)
        finally:
            del model, tokenizer
            release_model()


if __name__ == "__main__":
    main()
