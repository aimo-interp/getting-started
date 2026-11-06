"""Shared runtime for the cvbench heads: one greedy CoT per problem, the signals cvbench cached, a head.

Per ``are_robust`` call (one model and reasoning effort):

1. Greedy-decode each problem with the uncertainty-profiling baseline's own generator
   (``uncertainty_profile``: user-only chat template, 2,048 prompt / 4,096 new tokens, static
   cache), which also returns its 14 uncertainty features. This is the generation behind
   every ``data/cvbench/greedy`` trace the heads were trained on (``cvbench/generate.py greedy``).
2. If the head needs activations, one forward pass over prompt + response, keeping at every
   layer the last prompt token, the response mean and the means over 10 equal response
   segments (``generate.py`` acts and segs).
3. The head (``heads.py``) turns those signals into its raw features; they are z-scored with
   the model's training statistics and a shared logistic regression gives the label.

``introspect`` skips generation: three prompt-only passes per problem (``generate.py introspect``).

gpt-oss-120b is keyed by reasoning effort (``openai/gpt-oss-120b:medium``), as in
``cvbench/cases_oss.jsonl``; ``"default"`` is the template's own default, medium. Anything that
cannot be scored (model or effort not in the artifact, load failure, a failed batch, the run's
time budget) is predicted robust, the majority class.
"""

from __future__ import annotations

import dataclasses
import functools
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from trace_engine import heads
from uncertainty_profile.config import GenerationConfidenceConfig
from uncertainty_profile.extraction import format_problem, iter_generation_feature_batches, load_model_and_tokenizer

# The 3,600 s limit covers every are_robust call together, so time from import.
START = time.monotonic()
BUDGET_S = float(os.environ.get("TRACE_BUDGET_S", "3300"))
FIRST_BATCH_GUESS_S = 420.0  # 16 x 4,096 greedy tokens took 1.5-3.5 min on an A100
FIRST_BATCH_GUESS_OSS_S = 900.0  # gpt-oss-120b at medium effort: ~11 min for 15 on two A100s
DEFAULT_EFFORT = "medium"  # gpt-oss's chat template default
_BATCH_S: dict[str, list[float]] = {}  # per artifact key: a small model's batch times say nothing about gpt-oss


def log(msg: str) -> None:
    print(f"[trace-heads {time.monotonic() - START:7.1f}s] {msg}", file=sys.stderr, flush=True)


@functools.lru_cache(maxsize=1)
def load_artifact(path: str) -> dict:
    return json.loads(Path(path).read_text())


def artifact_key(models: dict, model_id: str, reasoning_effort: str) -> tuple[str, str | None]:
    """The artifact entry for this call and the effort to put in the chat template (None: leave it out)."""
    if f"{model_id}:{DEFAULT_EFFORT}" not in models:
        return model_id, None
    effort = DEFAULT_EFFORT if reasoning_effort in ("", "default", None) else reasoning_effort
    return f"{model_id}:{effort}", effort


@torch.no_grad()
def hidden_signals(model, tokenizer, problem: str, response: str, segments: int, max_new_tokens: int) -> dict:
    """``generate.py`` run_acts / run_segs: last prompt token, response mean, segment means, every layer."""
    device = model.get_input_embeddings().weight.device
    prompt_ids = tokenizer(format_problem(tokenizer, problem), add_special_tokens=False, truncation=True,
                           max_length=2048)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"][:max_new_tokens]
    try:
        hs = model(input_ids=torch.tensor([prompt_ids + response_ids], device=device), output_hidden_states=True,
                   logits_to_keep=1).hidden_states
    except TypeError:
        hs = model(input_ids=torch.tensor([prompt_ids + response_ids], device=device),
                   output_hidden_states=True).hidden_states
    p, n = len(prompt_ids), len(response_ids)
    out = {"last_prompt": torch.stack([h[0, p - 1] for h in hs]).float().cpu().numpy()}
    if n:
        out["resp_mean"] = torch.stack([h[0, p:].float().mean(0) for h in hs]).cpu().numpy()
    else:  # as training: zeros
        out["resp_mean"] = np.zeros_like(out["last_prompt"])
    if n >= segments:
        bounds = np.linspace(0, n, segments + 1).round().astype(int)
        out["seg_mean"] = torch.stack([torch.stack([h[0, p + a:p + b].float().mean(0)
                                                    for a, b in zip(bounds[:-1], bounds[1:])]) for h in hs],
                                      1).cpu().numpy()  # [S, L+1, d]
    del hs
    return out


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_robustness(artifact_path: Path, model_id: str, problems: list[str],
                       reasoning_effort: str = "default") -> list[bool]:
    preds = [True] * len(problems)
    if not problems:
        return preds
    try:
        art = load_artifact(str(artifact_path))
        head = heads.HEADS[art["head"]]
    except Exception as exc:  # noqa: BLE001
        log(f"artifact unavailable ({exc!r}); predicting robust")
        return preds
    key, effort = artifact_key(art["models"], model_id, reasoning_effort)
    entry = art["models"].get(key)
    if entry is None:
        log(f"{key}: not in artifact; predicting robust for {len(problems)} problems")
        return preds
    if time.monotonic() - START > BUDGET_S:
        log(f"{key}: time budget used up; predicting robust")
        return preds
    try:
        model, tokenizer = load_model_and_tokenizer(model_id, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        log(f"{model_id}: load failed ({exc!r}); predicting robust")
        return preds
    if effort:  # gpt-oss reads the effort from its chat template, which every step formats through
        tokenizer.apply_chat_template = functools.partial(tokenizer.apply_chat_template, reasoning_effort=effort)
    log(f"{key}: {len(problems)} problems, head {art['head']}")
    try:
        if art["head"] == "introspect":
            for i, problem in enumerate(problems):
                if time.monotonic() - START > BUDGET_S:
                    log(f"{key}: time budget reached; problems {i + 1}-{len(problems)} predicted robust")
                    break
                try:
                    preds[i] = heads.decide(art, entry, heads.introspect_features(model, tokenizer, problem))
                except Exception as exc:  # noqa: BLE001
                    log(f"{key}: problem {i} failed ({exc!r}); predicting robust")
        else:
            _greedy_and_score(art, entry, key, effort, model, tokenizer, problems, preds)
    finally:
        del model, tokenizer
        _release()
    log(f"{key}: {sum(preds)}/{len(preds)} predicted robust")
    return [bool(p) for p in preds]


def _greedy_and_score(art, entry, key, effort, model, tokenizer, problems, preds) -> None:
    gen = art["generation"]
    config = GenerationConfidenceConfig(max_prompt_length=gen["max_prompt_length"],
                                        max_new_tokens=gen["max_new_tokens"], batch_size=1)
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
            for k, (problem, row) in enumerate(zip(batch, _generate(model, tokenizer, batch, config))):
                signals = {"text": row.get("generated_text") or "", "n_tokens": int(row["generation_num_tokens"]),
                           "uncertainty": row}
                if art["needs_hidden"]:
                    signals.update(hidden_signals(model, tokenizer, problem, signals["text"], art.get("segments", 10),
                                                  gen["max_new_tokens"]))
                preds[start + k] = heads.decide(art, entry, signals)
        except Exception as exc:  # noqa: BLE001
            log(f"{key}: batch at {start} failed ({exc!r}); predicting robust for it")
        finally:
            _release()
        times.append(time.monotonic() - t0)
        log(f"{key}: batch {start // bs + 1}/{-(-len(problems) // bs)} in {times[-1]:.0f}s")


def _generate(model, tokenizer, problems: list[str], config: GenerationConfidenceConfig) -> list[dict]:
    """The baseline's greedy rows for one batch, halving it on CUDA OOM (the evaluation GPU is shared)."""
    try:
        cfg = dataclasses.replace(config, batch_size=len(problems))
        rows = []
        for _, batch_rows in iter_generation_feature_batches(model=model, tokenizer=tokenizer, problem_texts=problems,
                                                             config=cfg, include_generated_text=True):
            rows += batch_rows
        return rows
    except torch.cuda.OutOfMemoryError:
        if len(problems) == 1:
            raise
        _release()
        half = len(problems) // 2
        log(f"OOM at batch {len(problems)}; retrying as {half} + {len(problems) - half}")
        return _generate(model, tokenizer, problems[:half], config) + _generate(model, tokenizer, problems[half:], config)
