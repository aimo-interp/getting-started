"""Codabench entry point for the transcript probe.

For each problem: sample ``N`` reasoning traces on the original problem only,
compute three gold-free transcript features (transcript_probe/features.py),
z-score them with the training statistics of the model, and apply a shared
logistic regression trained on 82 labelled small-model cases
(``research/train.py --export`` writes ``artifacts/model.json``).

The artifact holds one fit per thinking budget. Time is balanced across problems:

* Rollover: the time left before the deadline is split over the problems not yet seen (out of
  ``EXPECTED_PROBLEMS``) in proportion to each model's expected cost (load time + ``COST_TOKENS`` at its decode
  speed), so slow models (Olmo, gpt-oss-120b) get more seconds and every case reaches about the same thinking
  budget. Time a problem leaves unused raises every later share.
* Borrowing: a call may take up to ``BORROW`` fair shares, as long as every later problem keeps
  ``FLOOR`` x the run's base share (``DEADLINE_SECONDS / EXPECTED_PROBLEMS``). Not on the first call,
  which is all any call would see if the state file were lost between problems.
* Snapping to a trained budget: tokens past a trained budget are thrown away at scoring, so each
  batch picks the largest trained budget its time can reach at the model's decode speed
  (``SPEED_PRIOR``, then the speed measured on this run) and stops once that much time has passed,
  banking the rest. It stops by time, not tokens, so traces that finish just past the budget still
  count as finished. Not on the first call (see borrowing).
* Easy problems end early: generation stops once all traces finish.

Codabench re-imports this module for every problem, so the run's clock, problem count and measured
speeds live in a state file (``_STATE``), not in module globals; all calls share one clock (4,200 s
job limit minus its 180 s margin). When a batch stops early, its traces are scored at the largest
trained budget it reached, cut to exactly that many tokens. Statistics are looked up per "model:effort"
(gpt-oss-120b, when trained with ``--oss``) and then per model; a model without
either is z-scored over its own batch. Falls back to robust when the
model cannot be loaded, the budget never reaches the smallest trained budget,
or the run's time is nearly used up.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from transcript_probe.features import FEATURES, transcript_features
from transcript_probe.sampler import chat_prompt, load_model, problems_per_batch, release, sample

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "model.json"
EXPECTED_PROBLEMS = 21   # Codabench competition phase: 21 cases, one are_robust call each
DEADLINE_SECONDS = 3950  # from import; the server stops at 4,020 s (4,200 job limit - 180 margin)
CALL_RESERVE_SECONDS = 10  # per call, after sampling stops: decode, features, release
MIN_SECONDS = 30         # below this, predict robust without loading the model
BORROW = 2.0             # a call may take up to this many fair shares...
FLOOR = 0.85             # ...as long as every later problem keeps this share of DEADLINE / EXPECTED
SPEED_MARGIN = 0.9       # plan on 90% of the expected decode speed
# Per-stream decode speed with 8 samples on the Codabench GPU (tok/s, from the 2026-10-29 LB log); a speed
# measured on this run replaces it. Unknown models get no snapping: they sample up to the largest budget.
SPEED_PRIOR = {"Qwen/Qwen3.5-4B": 54.0, "Skywork/Skywork-OR1-Math-7B": 58.0,
               "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": 42.0, "allenai/Olmo-3-7B-Think": 25.0,
               "openai/gpt-oss-120b": 25.0}
LOAD_PRIOR = {"openai/gpt-oss-120b": 90.0}  # seconds to load, measured on this run once seen
DEFAULT_LOAD = 25.0
COST_TOKENS = 8192  # the budget the cost weights are computed at
# Run state shared across re-imports and processes: one file per unpacked submission directory (a state
# older than one run counts as a new run). Not keyed by process: each case may run in a fresh process.
_STATE = Path(tempfile.gettempdir()) / (
    "transcript_probe_clock_" + hashlib.sha1(str(Path(__file__).resolve().parent).encode()).hexdigest()[:12] + ".json")


def _load_state() -> dict:
    try:
        state = json.loads(_STATE.read_text())
        if 0 <= time.time() - state["start"] < DEADLINE_SECONDS + 600:
            return state
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return {"start": time.time(), "seen": 0, "speed": {}, "load": {}}


def _save_state() -> None:
    try:
        _STATE.write_text(json.dumps(_state))
    except OSError:
        pass  # unwritable temp dir: the module globals still work if Codabench does not re-import


_state = _load_state()
_save_state()


def _cost(model_key: str, state: dict) -> float:
    """Expected seconds of one problem of this model: load + COST_TOKENS at its per-stream speed."""
    speed = state.get("speed", {}).get(model_key) or SPEED_PRIOR.get(model_key) or 40.0
    load = state.get("load", {}).get(model_key) or LOAD_PRIOR.get(model_key, DEFAULT_LOAD)
    return load + COST_TOKENS / speed


def _fair_share(left: float, n_problems: int, seen: int, cost: float, mean_cost: float) -> float:
    """Seconds for this call's problems: the time left split in proportion to expected cost, the other
    remaining problems counted at the mean cost over the trained models."""
    remaining = max(n_problems, EXPECTED_PROBLEMS - seen)
    return left * cost * n_problems / (cost * n_problems + (remaining - n_problems) * mean_cost)


def _call_cap(left: float, n_problems: int, seen: int, share: float) -> float:
    """Seconds this call may use: up to BORROW fair shares, keeping FLOOR of an equal share for every later
    problem. No borrowing on the first call: if the state file did not survive between problems, every call
    would look like the first and borrowing would overrun the deadline; a fair share each is the safe fallback."""
    remaining = max(n_problems, EXPECTED_PROBLEMS - seen)
    if seen == 0:
        return min(share, left)
    later = remaining - n_problems
    return max(min(BORROW * share, left - FLOOR * (left / remaining) * later), min(share, left))


def _target_budget(budgets: list[int], seconds: float, speed: float | None, fair_seconds: float) -> int:
    """Largest trained budget reachable in ``seconds`` at ``speed``. Borrowed time is only worth it when it
    reaches a larger budget than the fair share does; otherwise stay at the fair-share budget and bank the rest.
    Without a speed, ask for the largest budget and let the time limit stop it."""
    if not speed:
        return budgets[-1]

    def reach(t):
        fits = [b for b in budgets if b <= SPEED_MARGIN * speed * t]
        return fits[-1] if fits else budgets[0]
    return max(reach(fair_seconds), reach(seconds))


def _log(msg: str) -> None:
    print(f"[transcript-probe {time.time() - _state['start']:7.1f}s] {msg}", file=sys.stderr, flush=True)


def _scores(fit: dict, model_id: str, effort: str, rows: list[dict]) -> np.ndarray:
    raw = np.array([[r[name] for name in FEATURES] for r in rows], dtype=float)
    effort = "medium" if effort in ("", "default", None) else effort  # gpt-oss's template default
    stats = fit["stats"].get(f"{model_id}:{effort}") or fit["stats"].get(model_id)
    if stats is None:  # unseen model: normalise over this batch (no labels needed)
        mu, sd = raw.mean(axis=0), raw.std(axis=0)
    else:
        mu, sd = np.array(stats["mean"]), np.array(stats["sd"])
    sd = np.where(sd > 0, sd, 1.0)
    return ((raw - mu) / sd) @ np.array(fit["coef"]) + fit["intercept"]


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    if not problems:
        return []
    global _state
    _state = _load_state()  # another import may have advanced it
    now = time.time()
    left = DEADLINE_SECONDS - (now - _state["start"])
    speed_key = model_id.split(":")[0]
    artifact = json.loads(ARTIFACT.read_text())
    trained = sorted({k.split(":")[0] for k in artifact["fits"][next(iter(artifact["fits"]))]["stats"]})
    mean_cost = sum(_cost(k, _state) for k in trained) / max(len(trained), 1)
    share = _fair_share(left, len(problems), _state["seen"], _cost(speed_key, _state), mean_cost)
    first_call = _state["seen"] == 0
    cap = _call_cap(left, len(problems), _state["seen"], share)
    _state["seen"] += len(problems)
    _save_state()
    call_end, fair_end = now + cap, now + share
    _log(f"{model_id} ({reasoning_effort}): problem {_state['seen']}/{EXPECTED_PROBLEMS}, {left:.0f}s left, "
         f"fair share {share:.0f}s (cost weight {_cost(speed_key, _state) / mean_cost:.2f}), cap {cap:.0f}s for {len(problems)}")
    if left < MIN_SECONDS:
        _log(f"{model_id} ({reasoning_effort}): {left:.0f}s left; {len(problems)} problems predicted robust")
        return [True for _ in problems]
    n = artifact["n"]
    budgets = sorted(int(b) for b in artifact["fits"])
    t_load = time.time()
    try:
        tokenizer, model = load_model(model_id)
    except Exception:
        return [True for _ in problems]
    _state = _load_state()
    _state.setdefault("load", {})[speed_key] = time.time() - t_load
    _save_state()

    rows: list[dict | None] = [None] * len(problems)
    budget_of: list[int | None] = [None] * len(problems)
    try:
        prompts = [chat_prompt(tokenizer, p, reasoning_effort) for p in problems]
        prompt_tokens = max(len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts)
        size = problems_per_batch(model, n, budgets[-1], prompt_tokens)
        for start in range(0, len(problems), size):
            batch = list(range(start, min(start + size, len(problems))))
            part = len(batch) / (len(problems) - start)  # this batch's share of what the call has left
            left = (call_end - CALL_RESERVE_SECONDS - time.time()) * part
            fair = (fair_end - CALL_RESERVE_SECONDS - time.time()) * part
            if left < MIN_SECONDS:
                break
            speed = None if first_call else _state.get("speed", {}).get(speed_key) or SPEED_PRIOR.get(speed_key)
            target = _target_budget(budgets, left, speed, fair)
            max_time = min(left, target / (SPEED_MARGIN * speed)) if speed else left
            t0 = time.time()
            try:
                samples, steps = sample(model, tokenizer, [prompts[i] for i in batch], n, budgets[-1],
                                        max_time=max_time)
            except Exception as exc:  # noqa: BLE001
                _log(f"{model_id} ({reasoning_effort}): batch at {start} failed ({exc!r}); predicted robust")
                continue
            if steps >= 256:  # measured per-stream speed replaces the prior for later problems of this model
                _state = _load_state()
                _state.setdefault("speed", {})[speed_key] = steps / max(time.time() - t0, 1e-3)
                _save_state()
            if all(done for traces in samples for _, done in traces):
                budget = budgets[-1]  # every trace finished: nothing was cut short, valid at any budget
            else:
                reached = [b for b in budgets if b <= steps]
                if not reached:
                    _log(f"{model_id} ({reasoning_effort}): stopped at {steps} tokens, under the smallest budget "
                         f"{budgets[0]}; {len(batch)} problems predicted robust")
                    continue
                budget = reached[-1]
            for i, traces in zip(batch, samples):
                cut = [(ids[:budget], done and len(ids) <= budget) for ids, done in traces]
                texts = tokenizer.batch_decode([ids for ids, _ in cut], skip_special_tokens=True)
                rows[i] = transcript_features(problems[i], texts, [done for _, done in cut])
                budget_of[i] = budget
            finished = sum(done and len(ids) <= budget for traces in samples for ids, done in traces)
            _log(f"{model_id} ({reasoning_effort}): {len(batch)} problems x {n} samples, decoded {steps} tokens "
                 f"(target {target}, {max_time:.0f}s of {left:.0f}s, {steps / max(time.time() - t0, 1e-3):.0f} tok/s), scored at budget {budget}; {finished}/{n * len(batch)} finished within it")
    finally:
        del model
        release()

    predictions = [True] * len(problems)
    for budget in sorted({b for b in budget_of if b is not None}):
        idx = [i for i, b in enumerate(budget_of) if b == budget]
        fit = artifact["fits"][str(budget)]
        for i, s in zip(idx, _scores(fit, model_id, reasoning_effort, [rows[i] for i in idx])):
            predictions[i] = bool(s >= 0.0) if math.isfinite(s) else True
    return predictions
