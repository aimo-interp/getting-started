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
import os
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
# Decode steps/s of the joint greedy + 8-sample batch on the Codabench GPU (2026-10-31 stack-probe log)
# and load time (s; README: ~2 s for 4-8B, ~4 s for gpt-oss from the warm cache).
SPEED_PRIOR = {"Qwen/Qwen3.5-4B": 47.0, "Skywork/Skywork-OR1-Math-7B": 38.0,
               "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": 28.0, "allenai/Olmo-3-7B-Think": 23.0,
               "openai/gpt-oss-120b": 24.0}
LOAD_PRIOR = {"openai/gpt-oss-120b": 6.0}
DEFAULT_LOAD = 3.0
# Thinking budget a model's call is given time to reach regardless of its cost share (transcript-probe
# reached 8192 on Skywork; the joint decode needs ~255 s for it).
MIN_TOKENS = {"Skywork/Skywork-OR1-Math-7B": 8192}
COST_TOKENS = 8192  # the budget the cost weights are computed at
_STATE_NAME = "stack_probe_clock_" + hashlib.sha1(str(ARTIFACT.parent.parent).encode()).hexdigest()[:12] + ".json"
_STASH = "_stack_probe_clock"   # in-process copy (attribute on sys): survives re-imports of this module


def _state_paths() -> list[Path]:
    """Every place the clock is written; the 2026-11-02 Codabench log shows the temp-dir file alone did not
    survive between calls (every call logged problem 1/21, 3950s left)."""
    dirs = [tempfile.gettempdir(), "/dev/shm", "/tmp", os.path.expanduser("~"), os.getcwd(), str(ARTIFACT.parent)]
    out, seen = [], set()
    for d in dirs:
        try:
            q = Path(d).resolve() / _STATE_NAME
        except (OSError, RuntimeError):
            continue
        if str(q) not in seen:
            seen.add(str(q))
            out.append(q)
    return out


# ------------------------------------------------------------------------------------------ run state
def _valid(state) -> bool:
    try:
        return 0 <= time.time() - float(state["start"]) < DEADLINE_SECONDS + 600
    except (KeyError, TypeError, ValueError):
        return False


def _load_state() -> dict:
    found = []
    stash = getattr(sys, _STASH, None)
    if isinstance(stash, dict) and _valid(stash):
        found.append(stash)
    for q in _state_paths():
        try:
            state = json.loads(q.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(state, dict) and _valid(state):
            found.append(state)
    if not found:
        return {"start": time.time(), "seen": 0, "speed": {}, "load": {}}
    # the earliest start is the run's start; among copies of it, the most advanced one
    return dict(min(found, key=lambda st: (round(float(st["start"]), 3), -int(st.get("seen", 0)))))


def _save_state(state: dict) -> None:
    setattr(sys, _STASH, dict(state))
    text = json.dumps(state)
    for q in _state_paths():
        try:
            q.write_text(text)
        except OSError:
            pass


def _cuda_ok() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


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
    if "attn" in entry and "attention" in art:
        try:
            raw = attention_signals(model, tokenizer, problem, greedy_text, {**art["attention"], **entry.get("attn_layers", {})})
            if raw is not None:
                out["attn"] = list(_z([raw[f] for f in art["blocks"]["attn"]], entry["attn"]))
                _log(f"attention: raw {raw} -> z {[round(v, 3) for v in out['attn']]}")
        except Exception as exc:  # noqa: BLE001
            _log(f"attention pass failed ({exc!r}); attn block dropped")
            release()
    seg = s.get("seg_mean")
    t = entry["traj"]
    if seg is None:
        out["traj"] = list((np.asarray(t["median"]) - np.asarray(t["mean"])) / np.asarray(t["sd"]))
    else:
        layer = int(round(t["frac"] * (seg.shape[1] - 1)))
        feats = heads.traj_features(seg[:t["end"], layer])
        out["traj"] = list(_z([feats[f] for f in art["blocks"]["traj"]], t))
    return out



# ------------------------------------------------------------------------------------------ attention block
class _StopForward(Exception):
    pass


def _decoder_layers(model):
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers", "model.model.layers"):
        obj = model
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            return obj
        except AttributeError:
            continue
    raise AttributeError("decoder layers not found")


def attention_signals(model, tokenizer, problem: str, response: str, cfg: dict) -> dict | None:
    """Attention look-back of the greedy trace (cvbench/circuit_attr.py attn + stack.attn_feats), one eager
    pass over prompt + trace stopped after layer ``cfg['last_layer']``: per response token and head, the
    attention mass on the prompt outside the problem text (BOS sink + chat template) and on CoT tokens more
    than ``recent`` back; decile means, averaged over heads of layers ``first_layer``..``last_layer`` and the
    deciles in ``sink_deciles`` / ``far_deciles``."""
    segments, recent = int(cfg["segments"]), int(cfg["recent"])
    lo, hi = int(cfg["first_layer"]), int(cfg["last_layer"])
    prompt = format_problem(tokenizer, problem)
    enc = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=2048, return_offsets_mapping=True)
    prompt_ids, offs = enc["input_ids"], enc["offset_mapping"]
    c0 = prompt.find(problem)
    if c0 < 0:
        return None
    c1 = c0 + len(problem)
    inside = [j for j, (a, b) in enumerate(offs) if a >= c0 and b <= c1 and b > a]
    if not inside:
        return None
    ps, pe = inside[0], inside[-1] + 1
    resp_ids = tokenizer(response, add_special_tokens=False)["input_ids"][:int(cfg["max_tokens"])]
    p, n = len(prompt_ids), len(resp_ids)
    if n < segments:
        return None
    bounds = np.linspace(0, n, segments + 1).round().astype(int)
    layers = _decoder_layers(model)
    reg = {}

    def hook(l):
        def f(mod, inp, out):
            A = out[1] if isinstance(out, tuple) and len(out) > 1 else None
            if A is None:   # linear-attention layer (Qwen3.5) or no weights: skipped
                return
            Q = A[0, :, p:p + n].float()                                   # [H, n, T]
            pre = Q[..., :p].sum(-1) - Q[..., ps:pe].sum(-1)              # sink + template
            C = Q[..., p:].cumsum(-1)                                      # [H, n, n]
            i = torch.arange(n, device=Q.device)
            far = torch.where(i - recent - 1 >= 0, C[:, i, (i - recent - 1).clamp(min=0)], torch.zeros_like(pre))
            reg[l] = (torch.stack([pre[:, a:b].mean(1) for a, b in zip(bounds[:-1], bounds[1:])], 1).cpu().numpy(),
                      torch.stack([far[:, a:b].mean(1) for a, b in zip(bounds[:-1], bounds[1:])], 1).cpu().numpy())
            del Q, C
        return f

    def stop(mod, inp, out):
        raise _StopForward

    impl = getattr(model.config, "_attn_implementation", None)
    hooks = [layers[l].self_attn.register_forward_hook(hook(l)) for l in range(lo, hi + 1)
             if hasattr(layers[l], "self_attn")]
    hooks.append(layers[hi].register_forward_hook(stop))
    try:
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("eager")
        else:
            model.config._attn_implementation = "eager"
        device = model.get_input_embeddings().weight.device
        with torch.no_grad():
            try:
                model(input_ids=torch.tensor([prompt_ids + resp_ids], device=device))
            except _StopForward:
                pass
    finally:
        for h in hooks:
            h.remove()
        if impl is not None:
            if hasattr(model, "set_attn_implementation"):
                model.set_attn_implementation(impl)
            else:
                model.config._attn_implementation = impl
    if not reg:
        return None
    pre = np.stack([reg[l][0] for l in sorted(reg)])                       # [layers with weights, H, S]
    far = np.stack([reg[l][1] for l in sorted(reg)])
    a, b = cfg["sink_deciles"]
    c, d = cfg["far_deciles"]
    return {"attn_sink": float(pre[:, :, a:b].mean()), "attn_far": float(far[:, :, c:d].mean())}


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
    if _base(model_id) in MIN_TOKENS:
        b = _base(model_id)
        floor = n * (MIN_TOKENS[b] / (SPEED_MARGIN * SPEED_PRIOR[b]) + CALL_RESERVE_SECONDS) \
            + LOAD_PRIOR.get(b, DEFAULT_LOAD)
        cap = max(cap, min(floor, left - MIN_SECONDS))
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
    if _state.get("cuda_dead") or not _cuda_ok():
        # a dead GPU (2026-11-02: "CUDA error: unknown error" mid-run) makes every later call load gpt-oss on
        # the CPU for ~3.5 min and still run out of time; answer at once
        _log(f"{key}: CUDA unavailable; predicting robust")
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
                if "cuda" in repr(exc).lower() or "accelerator" in repr(exc).lower() or not _cuda_ok():
                    _state = _load_state()
                    _state["cuda_dead"] = True
                    _save_state(_state)
                    _log(f"{key}: GPU error; remaining problems predicted robust")
                    break
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
