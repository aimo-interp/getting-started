"""Diff-in-means probes on windows of the residual stream over the model's own greedy CoT.

Per problem: greedy-decode up to 4,096 tokens with the user-only chat template,
run one forward pass over prompt + response, cut the response into 10 equal token
segments and average the chosen layer's hidden states within each, then average
segments 0-4 and 5-8 into two windows (the CoT halves, without its final 10%).
Each window is projected onto the model's robust-minus-not-robust direction for
that window, and the shared logistic regression on both projections gives the
label. This mirrors ``cvbench/generate.py`` (greedy, segs) and
``cvbench/whitebox.py`` (seg_probe), where the artifact was cross-validated.

gpt-oss-120b is keyed by reasoning effort (``openai/gpt-oss-120b:medium``), as in
``cvbench/cases_oss.jsonl``: its effort goes into the chat template and selects
the directions. ``"default"`` means the template's own default, medium.

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

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "seg_probe.json"
# The 3,600 s limit covers every are_robust call together, so time from import.
START = time.monotonic()
BUDGET_S = float(os.environ.get("SEG_PROBE_BUDGET_S", "3300"))
FIRST_BATCH_GUESS_S = 420.0  # 16 x 4,096 greedy tokens took 1.5-3.5 min on an A100
FIRST_BATCH_GUESS_OSS_S = 900.0  # gpt-oss-120b at medium effort: ~11 min for 15 on two A100s
DEFAULT_EFFORT = "medium"  # gpt-oss's chat template default
_ARTIFACT: dict | None = None
_BATCH_S: dict[str, list[float]] = {}  # per artifact key: a small model's batch times say nothing about gpt-oss


def log(msg: str) -> None:
    print(f"[seg-probe {time.monotonic() - START:7.1f}s] {msg}", file=sys.stderr, flush=True)


def load_artifact() -> dict:
    global _ARTIFACT
    if _ARTIFACT is None:
        _ARTIFACT = json.loads(ARTIFACT.read_text())
        for entry in _ARTIFACT["models"].values():
            for w in entry.get("windows", []):
                w["mean"] = np.asarray(w["mean"], dtype=np.float32)
                w["direction"] = np.asarray(w["direction"], dtype=np.float32)
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


def artifact_key(art: dict, model_id: str, reasoning_effort: str) -> tuple[str, str | None]:
    """The artifact entry for this call and the effort to put in the chat template (None: leave it out)."""
    if f"{model_id}:{DEFAULT_EFFORT}" not in art["models"]:
        return model_id, None  # trained without efforts: the template is used as is
    effort = DEFAULT_EFFORT if reasoning_effort in ("", "default", None) else reasoning_effort
    return f"{model_id}:{effort}", effort


def format_problem(tokenizer, problem: str, effort: str | None = None) -> str:
    kwargs = {} if effort is None else {"reasoning_effort": effort}
    try:
        text = tokenizer.apply_chat_template([{"role": "user", "content": problem}], tokenize=False,
                                             add_generation_prompt=True, **kwargs)
    except (AttributeError, TypeError, ValueError):
        return problem
    return text if isinstance(text, str) else problem


def eos_ids(model, tokenizer):
    value = getattr(model.generation_config, "eos_token_id", None)
    return tokenizer.eos_token_id if value is None else value


@torch.no_grad()
def greedy_responses(model, tokenizer, problems: list[str], gen: dict, effort: str | None) -> list[str]:
    device = model.get_input_embeddings().weight.device
    enc = tokenizer([format_problem(tokenizer, p, effort) for p in problems], padding=True, truncation=True,
                    max_length=gen["max_prompt_length"], return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=gen["max_new_tokens"], do_sample=False,
                         pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_ids(model, tokenizer),
                         use_cache=True, cache_implementation=gen["cache_implementation"])
    generated = out[:, enc["input_ids"].shape[1]:].cpu()
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def greedy_split_on_oom(model, tokenizer, problems: list[str], gen: dict, effort: str | None) -> list[str]:
    """``greedy_responses``, halving the batch on CUDA OOM (the evaluation GPU is shared)."""
    try:
        return greedy_responses(model, tokenizer, problems, gen, effort)
    except torch.cuda.OutOfMemoryError:
        if len(problems) == 1:
            raise
        torch.cuda.empty_cache()
        half = len(problems) // 2
        log(f"OOM at batch {len(problems)}; retrying as {half} + {len(problems) - half}")
        return (greedy_split_on_oom(model, tokenizer, problems[:half], gen, effort)
                + greedy_split_on_oom(model, tokenizer, problems[half:], gen, effort))


@torch.no_grad()
def window_means(model, tokenizer, problem: str, response: str, layer: int, n_hidden: int, gen: dict,
                 segments: int, windows: list, effort: str | None) -> list[np.ndarray] | None:
    """Per window ``[a, b)``: mean over segments a..b-1 of the per-segment mean hidden state at ``layer``.

    The response (re-tokenized, as in training) is cut into ``segments`` equal token
    segments; under ``segments`` tokens every window is the full response mean, as
    ``whitebox.load_acts`` does for cases without cached segments.
    """
    device = model.get_input_embeddings().weight.device
    prompt_ids = tokenizer(format_problem(tokenizer, problem, effort), add_special_tokens=False, truncation=True,
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
    resp = hs[layer][0, len(prompt_ids):].float()
    n = len(response_ids)
    if n < segments:
        return [resp.mean(0).cpu().numpy()] * len(windows)
    bounds = np.linspace(0, n, segments + 1).round().astype(int)  # same cut as generate.py run_segs
    seg = torch.stack([resp[a:b].mean(0) for a, b in zip(bounds[:-1], bounds[1:])])
    return [seg[a:b].mean(0).cpu().numpy() for a, b in windows]


def decide(art: dict, entry: dict, hs: list[np.ndarray] | None) -> bool:
    logit = art["logistic"]["intercept"]
    if hs is not None:
        for coef, w, h in zip(art["logistic"]["coef"], entry["windows"], hs):
            logit += coef * float((h - w["mean"]) @ w["direction"]) / w["scale"]
    return logit >= 0.0


def predict_robustness(model_id: str, problems: list[str], reasoning_effort: str = "default") -> list[bool]:
    preds = [True] * len(problems)
    if not problems:
        return preds
    try:
        art = load_artifact()
    except Exception as exc:  # noqa: BLE001
        log(f"artifact unavailable ({exc!r}); predicting robust")
        return preds
    key, effort = artifact_key(art, model_id, reasoning_effort)
    entry = art["models"].get(key)
    if entry is None or "windows" not in entry:
        log(f"{key}: no direction in artifact; predicting robust for {len(problems)} problems")
        return preds

    gen = art["generation"]
    try:
        model, tokenizer = load_model_and_tokenizer(model_id)
    except Exception as exc:  # noqa: BLE001
        log(f"{model_id}: load failed ({exc!r}); predicting robust")
        return preds
    log(f"{key}: {len(problems)} problems, layer {entry['layer']}/{entry['n_hidden_states'] - 1}")
    try:
        bs = int(gen["batch_size"])
        for start in range(0, len(problems), bs):
            times = _BATCH_S.setdefault(key, [])
            need = max(times) if times else (FIRST_BATCH_GUESS_OSS_S if effort else FIRST_BATCH_GUESS_S)
            if time.monotonic() - START + need > BUDGET_S:
                log(f"{key}: time budget reached; problems {start + 1}-{len(problems)} predicted robust")
                break
            t0 = time.monotonic()
            batch = problems[start:start + bs]
            try:
                responses = greedy_split_on_oom(model, tokenizer, batch, gen, effort)
                for k, (problem, response) in enumerate(zip(batch, responses)):
                    hs = window_means(model, tokenizer, problem, response, entry["layer"],
                                      entry["n_hidden_states"], gen, art["segments"], art["windows"], effort)
                    preds[start + k] = bool(decide(art, entry, hs))
            except Exception as exc:  # noqa: BLE001
                log(f"{key}: batch at {start} failed ({exc!r}); predicting robust for it")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            times.append(time.monotonic() - t0)
            log(f"{key}: batch {start // bs + 1}/{-(-len(problems) // bs)} in {times[-1]:.0f}s")
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log(f"{key}: {sum(preds)}/{len(preds)} predicted robust")
    return [bool(p) for p in preds]
