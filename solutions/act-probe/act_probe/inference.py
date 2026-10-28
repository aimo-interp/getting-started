"""Diff-in-means probe on the mean residual stream of the model's own greedy CoT.

Per problem: greedy-decode up to 4,096 tokens with the user-only chat template,
run one forward pass over prompt + response, average the chosen layer's hidden
states over the response tokens, project onto the model's robust-minus-not-robust
direction, and threshold with the shared logistic regression. This mirrors
``cvbench/generate.py`` (greedy, acts) and ``cvbench/whitebox.py`` (act_probe),
where the artifact was cross-validated.

Anything that cannot be scored (model without a direction, load failure, a
failed batch, or the run's time budget running out) is predicted robust, the
majority class.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "act_probe.json"
# The 3,600 s limit covers every are_robust call together, so time from import.
START = time.monotonic()
BUDGET_S = float(os.environ.get("ACT_PROBE_BUDGET_S", "3300"))
FIRST_BATCH_GUESS_S = 420.0  # 16 x 4,096 greedy tokens took 1.5-3.5 min on an A100
_ARTIFACT: dict | None = None
_BATCH_S: list[float] = []


def log(msg: str) -> None:
    print(f"[act-probe {time.monotonic() - START:7.1f}s] {msg}", file=sys.stderr, flush=True)


def load_artifact() -> dict:
    global _ARTIFACT
    if _ARTIFACT is None:
        _ARTIFACT = json.loads(ARTIFACT.read_text())
        for entry in _ARTIFACT["models"].values():
            if "direction" in entry:
                entry["mean"] = np.asarray(entry["mean"], dtype=np.float32)
                entry["direction"] = np.asarray(entry["direction"], dtype=np.float32)
    return _ARTIFACT


def load_model_and_tokenizer(model_id: str):
    """Same loading as the uncertainty-profiling baseline that produced the training traces."""
    kwargs = {"local_files_only": True, "trust_remote_code": False}
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id, **kwargs)
    except (OSError, TypeError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="auto", **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, **kwargs)
    model.eval()
    return model, tokenizer


def format_problem(tokenizer, problem: str) -> str:
    try:
        text = tokenizer.apply_chat_template([{"role": "user", "content": problem}], tokenize=False,
                                             add_generation_prompt=True)
    except (AttributeError, TypeError, ValueError):
        return problem
    return text if isinstance(text, str) else problem


def eos_ids(model, tokenizer):
    value = getattr(model.generation_config, "eos_token_id", None)
    return tokenizer.eos_token_id if value is None else value


@torch.no_grad()
def greedy_responses(model, tokenizer, problems: list[str], gen: dict) -> list[str]:
    device = model.get_input_embeddings().weight.device
    enc = tokenizer([format_problem(tokenizer, p) for p in problems], padding=True, truncation=True,
                    max_length=gen["max_prompt_length"], return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=gen["max_new_tokens"], do_sample=False,
                         pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_ids(model, tokenizer),
                         use_cache=True, cache_implementation=gen["cache_implementation"])
    generated = out[:, enc["input_ids"].shape[1]:].cpu()
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


@torch.no_grad()
def response_mean(model, tokenizer, problem: str, response: str, layer: int, n_hidden: int, gen: dict) -> np.ndarray:
    """Mean hidden state at ``layer`` over the response tokens (re-tokenized, as in training)."""
    device = model.get_input_embeddings().weight.device
    prompt_ids = tokenizer(format_problem(tokenizer, problem), add_special_tokens=False, truncation=True,
                           max_length=gen["max_prompt_length"])["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"][:gen["max_new_tokens"]]
    if not response_ids:
        return None  # training stored zeros here; the projection of zeros is not meaningful
    ids = torch.tensor([prompt_ids + response_ids], device=device)
    try:
        hs = model(input_ids=ids, output_hidden_states=True, logits_to_keep=1).hidden_states
    except TypeError:
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states
    if len(hs) != n_hidden:
        raise RuntimeError(f"model returned {len(hs)} hidden states, artifact expects {n_hidden}")
    return hs[layer][0, len(prompt_ids):].float().mean(0).cpu().numpy()


def decide(art: dict, entry: dict, h: np.ndarray | None) -> bool:
    z = 0.0 if h is None else float((h - entry["mean"]) @ entry["direction"]) / entry["scale"]
    return art["logistic"]["coef"] * z + art["logistic"]["intercept"] >= 0.0


def predict_robustness(model_id: str, problems: list[str]) -> list[bool]:
    preds = [True] * len(problems)
    if not problems:
        return preds
    try:
        art = load_artifact()
    except Exception as exc:  # noqa: BLE001
        log(f"artifact unavailable ({exc!r}); predicting robust")
        return preds
    entry = art["models"].get(model_id)
    if entry is None or "direction" not in entry:
        log(f"{model_id}: no direction in artifact; predicting robust for {len(problems)} problems")
        return preds

    gen = art["generation"]
    try:
        model, tokenizer = load_model_and_tokenizer(model_id)
    except Exception as exc:  # noqa: BLE001
        log(f"{model_id}: load failed ({exc!r}); predicting robust")
        return preds
    log(f"{model_id}: {len(problems)} problems, layer {entry['layer']}/{entry['n_hidden_states'] - 1}")
    try:
        bs = int(gen["batch_size"])
        for start in range(0, len(problems), bs):
            need = max(_BATCH_S) if _BATCH_S else FIRST_BATCH_GUESS_S
            if time.monotonic() - START + need > BUDGET_S:
                log(f"{model_id}: time budget reached; problems {start + 1}-{len(problems)} predicted robust")
                break
            t0 = time.monotonic()
            batch = problems[start:start + bs]
            try:
                responses = greedy_responses(model, tokenizer, batch, gen)
                for k, (problem, response) in enumerate(zip(batch, responses)):
                    h = response_mean(model, tokenizer, problem, response, entry["layer"],
                                      entry["n_hidden_states"], gen)
                    preds[start + k] = bool(decide(art, entry, h))
            except Exception as exc:  # noqa: BLE001
                log(f"{model_id}: batch at {start} failed ({exc!r}); predicting robust for it")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            _BATCH_S.append(time.monotonic() - t0)
            log(f"{model_id}: batch {start // bs + 1}/{-(-len(problems) // bs)} in {_BATCH_S[-1]:.0f}s")
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log(f"{model_id}: {sum(preds)}/{len(preds)} predicted robust")
    return [bool(p) for p in preds]
