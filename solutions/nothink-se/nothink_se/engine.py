"""No-reasoning semantic entropy.

Per problem: the thinking block is closed and the answer is prefilled up to ``\\boxed{`` (the same prompt as
``cvbench/generate.py --stages nothink --nt-mode direct``), then ``n`` answers are sampled at T=1 (~10 tokens
each). Features: discrete semantic entropy over the answer clusters (no answer = one cluster), the majority
share, the no-answer rate and the entropy of the first answer token. They are z-scored with the model's
training statistics and a pooled logistic regression decides; per model the option ("nt" or "prior",
always robust) was chosen by inner cross-validation. Anything that cannot be scored is predicted robust.
"""

from __future__ import annotations

import functools
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from uncertainty_profile.extraction import format_problem, load_model_and_tokenizer, release_model

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "model.json"
SOLVE_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
PREFILL = "The final answer is $\\boxed{"
DEFAULT_EFFORT = "medium"
DEADLINE_SECONDS = 3600
_START = time.time()


def _log(msg: str) -> None:
    print(f"[nothink-se {time.time() - _START:7.1f}s] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------ answer parsing (datagen/common.py)
def parse_integer(text):
    if text is None:
        return None
    cleaned = text.strip().strip("$").replace("\\!", "").replace("{,}", ",").strip()
    if "," in cleaned:
        if not re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.0+)?", cleaned):
            return None
        cleaned = cleaned.replace(",", "")
    cleaned = re.sub(r"^\\text\{(.*)\}$", r"\1", cleaned).strip()
    match = re.fullmatch(r"([+-]?\d+)(?:\.0+)?", cleaned)
    return int(match.group(1)) if match else None


def parse_boxed(content: str):
    content = re.sub(r"\\[,;: ]|~", "", content).strip()
    value = parse_integer(content)
    if value is None and "=" in content:
        value = parse_integer(content.rsplit("=", 1)[1])
    return value


def no_think_prefix(prompt: str, tokenizer) -> str:
    if "<|channel|>" in tokenizer.get_vocab():  # gpt-oss (harmony): empty analysis, then the final channel
        return prompt + "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
    if prompt.rstrip().endswith("<think>"):
        return prompt.rstrip() + "\n\n</think>\n\n"
    return prompt + "<think>\n\n</think>\n\n" if "<think>" in tokenizer.get_vocab() else prompt


def entropy(answers: list) -> float:
    counts = np.unique(np.array(["NA" if a is None else str(a) for a in answers]), return_counts=True)[1]
    q = counts / counts.sum()
    return float(-(q * np.log(q)).sum())


# ------------------------------------------------------------------------------ features
@torch.no_grad()
def features(model, tokenizer, problem: str, n: int, max_new: int) -> dict:
    device = model.get_input_embeddings().weight.device
    prompt = no_think_prefix(format_problem(tokenizer, problem + SOLVE_SUFFIX), tokenizer) + PREFILL
    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    p = enc.input_ids.shape[1]
    lp = torch.log_softmax(model(**enc, logits_to_keep=1).logits[0, -1].float(), -1)
    tok_ent = float(-(lp.exp() * lp).sum())
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=max_new,
                         num_return_sequences=n, pad_token_id=pad)[:, p:]
    answers = []
    for row in gen.tolist():
        ids = [t for t in row if t not in (tokenizer.pad_token_id, tokenizer.eos_token_id)]
        t = tokenizer.decode(ids, skip_special_tokens=True)
        answers.append(parse_boxed(t.split("}", 1)[0]) if "}" in t else parse_boxed(t))
    given = [a for a in answers if a is not None]
    top = np.unique(np.array([str(a) for a in given]), return_counts=True)[1].max() / n if given else 0.0
    return {"se": entropy(answers), "top_share": float(top), "na_rate": 1 - len(given) / n,
            "first_tok_entropy": tok_ent}


def decide(art, entry, feats: dict) -> bool:
    if entry["option"] == "prior":
        return True
    fit = art["logistic"]
    x = np.array([feats[f] for f in art["features"]], dtype=np.float64)
    x = np.where(np.isfinite(x), x, np.asarray(entry["median"]))
    z = (x - np.asarray(entry["mean"])) / np.asarray(entry["sd"])
    logit = float(np.asarray(fit["coef"]) @ z + fit["intercept"])
    return bool(logit >= 0.0) if math.isfinite(logit) else True


# ------------------------------------------------------------------------------ entry point
def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    preds = [True] * len(problems)
    if not problems:
        return preds
    art = json.loads(ARTIFACT.read_text())
    effort = DEFAULT_EFFORT if reasoning_effort in ("", "default", None) else reasoning_effort
    key = f"{model_id}:{effort}" if f"{model_id}:{DEFAULT_EFFORT}" in art["models"] else model_id
    entry = art["models"].get(key)
    if entry is None or entry["option"] == "prior":
        _log(f"{key}: {'not in artifact' if entry is None else 'prior option'}; predicting robust")
        return preds
    try:
        model, tokenizer = load_model_and_tokenizer(model_id, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        _log(f"{model_id}: load failed ({exc!r}); predicting robust")
        return preds
    if key != model_id:  # gpt-oss: effort through the chat template
        tokenizer.apply_chat_template = functools.partial(tokenizer.apply_chat_template, reasoning_effort=effort)
    torch.manual_seed(0)
    try:
        for i, problem in enumerate(problems):
            if time.time() - _START > DEADLINE_SECONDS:
                _log(f"{key}: past the deadline; problem {i} predicted robust")
                break
            try:
                f = features(model, tokenizer, problem, int(art["n"]), int(art["max_new"]))
                preds[i] = decide(art, entry, f)
                _log(f"{key}: problem {i}: {', '.join(f'{k} {v:.3f}' for k, v in f.items())} -> "
                     f"{'robust' if preds[i] else 'not robust'}")
            except Exception as exc:  # noqa: BLE001
                _log(f"{key}: problem {i} failed ({exc!r}); predicted robust")
            torch.cuda.empty_cache()
    finally:
        del model, tokenizer
        release_model()
    return [bool(p) for p in preds]
