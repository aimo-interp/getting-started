"""Runtime of the per-model stack: one decode per case gives the greedy trace and 8 samples.

Per ``are_robust`` call (one model and effort, usually one problem on Codabench):

1. Time: the run's clock lives in a state file (Codabench re-imports the module per case). The call's
   share of the time left is weighted by the model's expected cost (load time + tokens / decode speed,
   ``COST``), so slow models (Olmo, gpt-oss) get more seconds and every case reaches about the same
   thinking budget. Unused time rolls over; a call may borrow up to ``BORROW`` shares if later problems
   keep ``FLOOR`` of theirs. The thinking budget is snapped to the largest trained budget the share can
   reach at the model's speed (prior, then measured on this run); decoding stops by time.
2. Decode: per problem a batch of 1 + n streams. Stream 0 is the greedy chain of thought (plain chat
   template, forced argmax, stopped at ``greedy_tokens``), streams 1..n sample at T=1 on the problem +
   SOLVE suffix (the transcript probe's prompt). Decoding is memory-bound, so the extra stream is cheap.
3. Features: one forward pass over prompt + greedy trace gives the segment means (seg probe windows and
   trajectory geometry, as ``trace_engine``); the samples give the transcript features at the budget
   reached. The model's option (block subset, chosen offline) is restricted to the blocks available and
   the matching logistic regression decides.

Anything that cannot be scored is predicted robust (the majority class).
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from trace_engine import heads
from trace_engine.engine import hidden_signals
from transcript_probe.features import transcript_features
from transcript_probe.sampler import _eos_ids, problems_per_batch, release
from uncertainty_profile.extraction import format_problem, load_model_and_tokenizer

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "stack.json"
SOLVE = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
EXPECTED_PROBLEMS = 21
DEADLINE_SECONDS = 3950
CALL_RESERVE_SECONDS = 12   # after decoding: hidden-state pass, features, release
MIN_SECONDS = 40
BORROW = 1.5
FLOOR = 0.85
SPEED_MARGIN = 0.9
DEFAULT_EFFORT = "medium"
# Per-stream decode speed (tok/s) with 8 streams on the Codabench GPU (2026-10-29 log) and load time (s).
SPEED_PRIOR = {"Qwen/Qwen3.5-4B": 54.0, "Skywork/Skywork-OR1-Math-7B": 58.0,
               "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": 42.0, "allenai/Olmo-3-7B-Think": 25.0,
               "openai/gpt-oss-120b": 25.0}
LOAD_PRIOR = {"openai/gpt-oss-120b": 90.0}
DEFAULT_LOAD = 25.0
COST_TOKENS = 8192  # the budget the cost weights are computed at
_STATE = Path(tempfile.gettempdir()) / (
    "stack_probe_clock_" + hashlib.sha1(str(ARTIFACT.parent.parent).encode()).hexdigest()[:12] + ".json")


# ------------------------------------------------------------------------------------------ run state
def _load_state() -> dict:
    try:
        state = json.loads(_STATE.read_text())
        if 0 <= time.time() - state["start"] < DEADLINE_SECONDS + 600:
            return state
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return {"start": time.time(), "seen": 0, "speed": {}, "load": {}}


def _save_state(state: dict) -> None:
    try:
        _STATE.write_text(json.dumps(state))
    except OSError:
        pass


_state = _load_state()
_save_state(_state)


def _log(msg: str) -> None:
    print(f"[stack-probe {time.time() - _state['start']:7.1f}s] {msg}", file=sys.stderr, flush=True)


def _base(model_id: str) -> str:
    return model_id.split(":")[0]


def _cost(model_id: str, state: dict) -> float:
    b = _base(model_id)
    speed = state.get("speed", {}).get(b) or SPEED_PRIOR.get(b) or 40.0
    load = state.get("load", {}).get(b) or LOAD_PRIOR.get(b, DEFAULT_LOAD)
    return load + COST_TOKENS / speed


def _call_cap(left: float, n: int, seen: int, cost: float, mean_cost: float) -> float:
    """Seconds for this call: the time left split in proportion to expected cost, up to BORROW shares as
    long as every later problem keeps FLOOR of an equal share. No borrowing on the first call (if the
    state file were lost, every call would look first)."""
    remaining = max(n, EXPECTED_PROBLEMS - seen)
    share = left * cost * n / (cost * n + (remaining - n) * mean_cost)
    if seen == 0:
        return min(share, left)
    later = remaining - n
    return max(min(BORROW * share, left - FLOOR * (left / remaining) * later), min(share, left))


def _target_budget(budgets: list[int], seconds: float, speed: float | None) -> int:
    if not speed:
        return budgets[-1]
    fits = [b for b in budgets if b <= SPEED_MARGIN * speed * seconds]
    return fits[-1] if fits else budgets[0]


# ------------------------------------------------------------------------------------------ decoding
class GreedyStreams(LogitsProcessor):
    """Rows in ``rows`` take the argmax token (sampling made deterministic) and are closed with EOS once
    they have generated ``limit`` tokens."""

    def __init__(self, rows: list[int], prompt_len: int, limit: int, eos: int):
        self.rows, self.prompt_len, self.limit, self.eos = rows, prompt_len, limit, eos

    def __call__(self, input_ids, scores):
        generated = input_ids.shape[1] - self.prompt_len
        for r in self.rows:
            row = scores[r]
            top = int(torch.argmax(row)) if generated < self.limit else self.eos
            row.fill_(-float("inf"))
            row[top] = 0.0
        return scores


@torch.no_grad()
def decode(model, tokenizer, greedy_prompts, sample_prompts, n, budget, greedy_limit, max_time, seed=0):
    """Per problem: (greedy ids, greedy finished, [(ids, finished)] * n). Also the decoding steps run."""
    device = model.get_input_embeddings().weight.device
    texts = []
    for g, s in zip(greedy_prompts, sample_prompts):
        texts += [g] + [s] * n
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    eos = _eos_ids(tokenizer, model)
    prompt_len = enc["input_ids"].shape[1]
    greedy_rows = list(range(0, len(texts), n + 1))
    torch.manual_seed(seed)
    out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=budget,
                         max_time=max_time, pad_token_id=tokenizer.pad_token_id, eos_token_id=eos,
                         logits_processor=LogitsProcessorList([GreedyStreams(greedy_rows, prompt_len, greedy_limit, eos[0])]))
    gen = out[:, prompt_len:].tolist()
    steps = len(gen[0]) if gen else 0
    eos_set = set(eos)
    rows = []
    for ids in gen:
        end = next((i for i, t in enumerate(ids) if t in eos_set), None)
        rows.append((ids if end is None else ids[:end], end is not None))
    per = [rows[i * (n + 1):(i + 1) * (n + 1)] for i in range(len(greedy_prompts))]
    return [(p[0][0], p[0][1], p[1:]) for p in per], steps


# ------------------------------------------------------------------------------------------ features
def _z(raw: list[float], stats: dict) -> np.ndarray:
    x = np.asarray(raw, dtype=np.float64)
    x = np.where(np.isfinite(x), x, np.asarray(stats["median"]))
    return (x - np.asarray(stats["mean"])) / np.asarray(stats["sd"])


def trace_blocks(art, entry, model, tokenizer, problem: str, greedy_text: str) -> dict:
    """seg window scores and z-scored trajectory features from the greedy trace (empty when unavailable)."""
    out = {}
    if not greedy_text:
        return out
    s = hidden_signals(model, tokenizer, problem, greedy_text, art["segments"], art["generation"]["greedy_tokens"])
    if "windows" in entry:
        w = entry.get("seg_weight", [1.0] * len(entry["windows"]))
        out["seg"] = [v * wj for v, wj in zip(heads._diffmean(art, entry, s), w)]
    seg = s.get("seg_mean")
    t = entry["traj"]
    if seg is None:
        out["traj"] = list((np.asarray(t["median"]) - np.asarray(t["mean"])) / np.asarray(t["sd"]))
    else:
        layer = int(round(t["frac"] * (seg.shape[1] - 1)))
        feats = heads.traj_features(seg[:t["end"], layer])
        out["traj"] = list(_z([feats[f] for f in art["blocks"]["traj"]], t))
    return out


def tx_block(art, entry, tokenizer, problem: str, samples, budget: int) -> list[float]:
    cut = [(ids[:budget], done and len(ids) <= budget) for ids, done in samples]
    texts = tokenizer.batch_decode([ids for ids, _ in cut], skip_special_tokens=True)
    feats = transcript_features(problem, texts, [done for _, done in cut])
    feats["length"] = float(np.mean([math.log(len(ids) + 1) for ids, _ in cut]))
    return list(_z([feats[f] for f in art["blocks"]["tx"]], entry["tx"][str(budget)]))


def decide(art, entry, blocks: dict, budget: int | None) -> bool:
    """The model's option restricted to the available blocks, else the first fallback that is."""
    want = art["options"][entry["option"]]
    have = [b for b in want if b in blocks]
    name = next((k for k, v in art["options"].items() if v == have), None)
    if name is None:
        name = next((k for k in art["fallback"] if all(b in blocks for b in art["options"][k])), None)
    if name is None:
        return True
    fit = art["logistic"][name][str(budget if "tx" in art["options"][name] else 0)]
    x = np.concatenate([np.asarray(blocks[b], dtype=np.float64) for b in art["options"][name]])
    logit = float(np.asarray(fit["coef"]) @ x + fit["intercept"])
    return bool(logit >= 0.0) if math.isfinite(logit) else True


# ------------------------------------------------------------------------------------------ entry point
def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    global _state
    preds = [True] * len(problems)
    if not problems:
        return preds
    art = json.loads(ARTIFACT.read_text())
    effort = DEFAULT_EFFORT if reasoning_effort in ("", "default", None) else reasoning_effort
    key = f"{model_id}:{effort}" if f"{model_id}:{DEFAULT_EFFORT}" in art["models"] else model_id
    entry = art["models"].get(key)
    _state = _load_state()
    now = time.time()
    left = DEADLINE_SECONDS - (now - _state["start"])
    n = len(problems)
    mean_cost = float(np.mean([_cost(m, _state) for m in art["models"]]))
    cap = _call_cap(left, n, _state["seen"], _cost(model_id, _state), mean_cost)
    first_call = _state["seen"] == 0
    _state["seen"] += n
    _save_state(_state)
    _log(f"{key}: problem {_state['seen']}/{EXPECTED_PROBLEMS}, {left:.0f}s left, cap {cap:.0f}s for {n} "
         f"(cost weight {_cost(model_id, _state) / mean_cost:.2f})")
    if entry is None:
        _log(f"{key}: not in artifact; predicting robust")
        return preds
    if left < MIN_SECONDS:
        _log(f"{key}: {left:.0f}s left; predicting robust")
        return preds
    call_end = now + cap
    t_load = time.time()
    try:
        model, tokenizer = load_model_and_tokenizer(model_id, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        _log(f"{model_id}: load failed ({exc!r}); predicting robust")
        return preds
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if key != model_id:  # gpt-oss: effort through the chat template, every formatting goes through it
        tokenizer.apply_chat_template = functools.partial(tokenizer.apply_chat_template, reasoning_effort=effort)
    _state = _load_state()
    _state.setdefault("load", {})[_base(model_id)] = time.time() - t_load
    _save_state(_state)

    gen = art["generation"]
    budgets, n_samples, greedy_limit = sorted(gen["budgets"]), int(gen["n_samples"]), int(gen["greedy_tokens"])
    try:
        greedy_prompts = [format_problem(tokenizer, p) for p in problems]
        sample_prompts = [format_problem(tokenizer, p + SOLVE) for p in problems]
        prompt_tokens = max(len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in sample_prompts)
        size = problems_per_batch(model, n_samples + 1, budgets[-1], prompt_tokens)
        for start in range(0, n, size):
            batch = list(range(start, min(start + size, n)))
            part = len(batch) / (n - start)
            secs = (call_end - CALL_RESERVE_SECONDS - time.time()) * part
            if secs < MIN_SECONDS / 2:
                _log(f"{key}: out of time; problems {start + 1}-{n} predicted robust")
                break
            speed = None if first_call else _state.get("speed", {}).get(_base(model_id)) or SPEED_PRIOR.get(_base(model_id))
            target = _target_budget(budgets, secs, speed)
            max_time = min(secs, max(target, greedy_limit) / (SPEED_MARGIN * speed)) if speed else secs
            t0 = time.time()
            try:
                rows, steps = decode(model, tokenizer, [greedy_prompts[i] for i in batch],
                                     [sample_prompts[i] for i in batch], n_samples, max(target, greedy_limit),
                                     greedy_limit, max_time)
            except Exception as exc:  # noqa: BLE001
                _log(f"{key}: decode failed at {start} ({exc!r}); predicted robust")
                release()
                continue
            dt = time.time() - t0
            if steps >= 256:
                _state = _load_state()
                _state.setdefault("speed", {})[_base(model_id)] = steps / max(dt, 1e-3)
                _save_state(_state)
            for i, (g_ids, g_done, samples) in zip(batch, rows):
                blocks = {}
                greedy_text = tokenizer.decode(g_ids, skip_special_tokens=True)
                try:
                    blocks.update(trace_blocks(art, entry, model, tokenizer, problems[i], greedy_text))
                except Exception as exc:  # noqa: BLE001
                    _log(f"{key}: hidden states failed ({exc!r}); trace blocks dropped")
                if all(done for _, done in samples):
                    budget = budgets[-1]
                else:
                    reached = [b for b in budgets if b <= steps]
                    budget = reached[-1] if reached else None
                if budget is not None:
                    blocks["tx"] = tx_block(art, entry, tokenizer, problems[i], samples, budget)
                preds[i] = decide(art, entry, blocks, budget)
                _log(f"{key}: problem {i}: greedy {len(g_ids)} tok ({'done' if g_done else 'cut'}), "
                     f"{sum(d for _, d in samples)}/{n_samples} samples finished, {steps} steps in {dt:.0f}s "
                     f"({steps / max(dt, 1e-3):.0f} tok/s), tx budget {budget}, blocks {sorted(blocks)} -> "
                     f"{'robust' if preds[i] else 'not robust'}")
            release()
    finally:
        del model, tokenizer
        release()
    return [bool(p) for p in preds]
