"""Heads: raw features from one problem's signals, then the shared decision.

Each feature function is the one ``cvbench/cv.py`` / ``cvbench/whitebox.py`` cross-validated
(copied, so the shipped package needs nothing outside its own folder). ``decide`` z-scores
the raw features with the model's training statistics (``zscore`` heads, as
``cv.zscore_within_model``) and applies the shared logistic regression.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import torch

from uncertainty_profile.config import FEATURE_NAMES
from uncertainty_profile.extraction import format_problem

# ------------------------------------------------------------------ black-box text (cv.py cot_features)
HESITATION = re.compile(
    r"\b(wait|hmm+|alternatively|actually|let me (?:double[- ]check|re-?check|verify|reconsider)|"
    r"but (?:wait|hold on)|on second thought|i made (?:a|an) (?:mistake|error)|that'?s (?:not right|wrong))\b",
    re.IGNORECASE)
NUMBER_CLAIM = re.compile(r"(?:=|is)\s*\$?(-?\d{1,7})\$?[.\s]")
COT_FEATURES = ("log_tokens", "truncated", "finished_thinking", "hesitation_per_1k", "wait_per_1k",
                "n_boxed_distinct", "has_boxed", "claim_entropy", "verify_per_1k")


def cot_features(text: str, n_tokens: int, cap: int = 4096) -> dict:
    words = max(len(text.split()), 1)
    think, _, after = text.partition("</think>")
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    claims = NUMBER_CLAIM.findall(think)
    counts = Counter(claims)
    return {
        "log_tokens": math.log(n_tokens + 1),
        "truncated": float(n_tokens >= cap - 1),
        "finished_thinking": float(bool(after)),
        "hesitation_per_1k": 1000 * len(HESITATION.findall(think)) / words,
        "wait_per_1k": 1000 * len(re.findall(r"\bwait\b", think, re.IGNORECASE)) / words,
        "n_boxed_distinct": len(set(boxed)),
        "has_boxed": float(bool(boxed)),
        "claim_entropy": -sum(v / len(claims) * math.log(v / len(claims)) for v in counts.values()) if claims else 0.0,
        "verify_per_1k": 1000 * len(re.findall(r"\b(check|verify|confirm)\b", think, re.IGNORECASE)) / words,
    }


# ------------------------------------------------------------------ Chain-of-Embedding (whitebox.coe_features)
def coe_features(H: np.ndarray) -> dict:
    H = H.astype(np.float64)

    def angle(a, b):
        return np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))

    mag = np.linalg.norm(np.diff(H, axis=0), axis=1) / np.linalg.norm(H[-1] - H[0])
    ang = np.array([angle(H[l], H[l + 1]) for l in range(len(H) - 1)]) / angle(H[0], H[-1])
    return {"coe_r": mag.mean() - ang.mean(), "coe_c": np.hypot((mag * np.cos(ang)).mean(), (mag * np.sin(ang)).mean()),
            "coe_mag": mag.mean(), "coe_ang": ang.mean(), "coe_mag_var": mag.var(), "coe_ang_var": ang.var()}


# ------------------------------------------------------------------ CoT trajectory (cvbench/traj.py)
def traj_features(H: np.ndarray) -> dict:
    """Geometry of one layer's path over the CoT segments (H: segments x dim, wrap-up segment dropped)."""
    H = H.astype(np.float64)
    steps = np.linalg.norm(np.diff(H, axis=0), axis=1)
    norms = np.linalg.norm(H, axis=1)
    return {"step_mean": np.log(steps + 1e-12).mean(),
            "norm_slope": np.polyfit(np.arange(len(norms)), norms / norms.mean(), 1)[0]}


# ------------------------------------------------------------------ introspection (generate.py run_introspect)
INTROSPECT = {
    "robust": "Suppose you were asked to solve the following problem, and separately a slightly reworded version "
              "of it (renamed variables, a typo, an irrelevant extra sentence, or a paraphrase). Would your final "
              "answers to the two versions be the same?",
    "correct": "Suppose you were asked to solve the following problem. Would your final answer be correct?",
    "long": "Suppose you were asked to solve the following problem. Would your solution take more than 4,000 tokens?",
}


def no_think_prefix(prompt: str, tokenizer) -> str:
    """Close the thinking block so the next token is the answer itself (no chain of thought)."""
    if "<|channel|>" in tokenizer.get_vocab():  # gpt-oss (harmony): empty analysis, then the final channel
        return prompt + "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
    if prompt.rstrip().endswith("<think>"):
        return prompt.rstrip() + "\n\n</think>\n\n"
    return prompt + "<think>\n\n</think>\n\n" if "<think>" in tokenizer.get_vocab() else prompt


@torch.no_grad()
def introspect_features(model, tokenizer, problem: str) -> dict:
    device = model.get_input_embeddings().weight.device
    vocab = tokenizer.get_vocab()
    ids = {w: [vocab[t] for t in {w, "Ġ" + w, "▁" + w} if t in vocab] for w in ("Yes", "No")}
    row = {}
    for name, question in INTROSPECT.items():
        text = f"{question} Respond with only Yes or No.\n\nProblem:\n{problem}"
        prompt = no_think_prefix(format_problem(tokenizer, text), tokenizer)
        enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        logp = torch.log_softmax(model(**enc, logits_to_keep=1).logits[0, -1].float(), -1)
        yes, no = (torch.logsumexp(logp[ids[w]], 0).item() for w in ("Yes", "No"))
        row[name] = yes - no
    return {"introspect": row}


# ------------------------------------------------------------------ activation sites (whitebox.load_acts)
def site_vector(signals: dict, site: str) -> np.ndarray:
    """[L+1, d] at ``site``: last_prompt, resp_mean, or seg_<a>_<b> (mean of segments a..b-1)."""
    if site.startswith("seg_"):
        a, b = map(int, site.split("_")[1:])
        seg = signals.get("seg_mean")
        return signals["resp_mean"] if seg is None else seg[a:b].mean(0)  # short response: full mean
    return signals[site]


def mlp_logit(net: dict, x: np.ndarray) -> float:
    """sklearn StandardScaler + MLPClassifier (ReLU, logistic output), exported as arrays."""
    h = (x - np.asarray(net["scaler_mean"])) / np.asarray(net["scaler_scale"])
    layers = list(zip(net["coefs"], net["intercepts"]))
    for W, b in layers[:-1]:
        h = np.maximum(h @ np.asarray(W) + np.asarray(b), 0.0)
    W, b = layers[-1]
    z = float((h @ np.asarray(W) + np.asarray(b))[0])
    p = min(max(1.0 / (1.0 + math.exp(-z)), 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


# ------------------------------------------------------------------ heads: (art, entry, signals) -> raw features
def _length(art, entry, s):
    return [math.log1p(s["n_tokens"]), float(s["n_tokens"] >= 4095)]


def _cot_text(art, entry, s):
    feats = cot_features(s["text"], s["n_tokens"])
    return [feats[f] for f in COT_FEATURES]


def _uncertainty(art, entry, s):
    return [float(s["uncertainty"][f]) if s["uncertainty"].get(f) is not None else float("nan") for f in FEATURE_NAMES]


def _coe(art, entry, s):
    feats = coe_features(site_vector(s, art["site"]))
    return [feats[f] for f in art["features"]]


def _diffmean(art, entry, s):
    return [float((site_vector(s, w["site"])[entry["layer"]] - np.asarray(w["mean"])) @ np.asarray(w["direction"]))
            / w["scale"] for w in entry["windows"]]


def _saplma(art, entry, s):
    return [mlp_logit(entry["mlp"], site_vector(s, art["site"])[entry["layer"]].astype(np.float64))]


def _seg_traj(art, entry, s):
    """seg_probe window scores, then the trajectory features z-scored with the model's training statistics."""
    seg = s.get("seg_mean")
    t = entry["traj"]
    if seg is None:  # short response: no segments, the training median stands in
        raw = np.asarray(t["median"])
    else:
        feats = traj_features(seg[:t["end"], entry["layer"]])
        raw = np.asarray([feats[f] for f in art["features"]])
    return _diffmean(art, entry, s) + list((raw - np.asarray(t["mean"])) / np.asarray(t["sd"]))


def _introspect(art, entry, s):
    return [s["introspect"][f] for f in art["features"]]


HEADS = {"length": _length, "cot_text": _cot_text, "uncertainty": _uncertainty, "coe": _coe,
         "diffmean": _diffmean, "saplma": _saplma, "introspect": _introspect, "seg_traj": _seg_traj}


def decide(art: dict, entry: dict, signals: dict) -> bool:
    x = np.asarray(HEADS[art["head"]](art, entry, signals), dtype=np.float64)
    if art["zscore"]:
        x = np.where(np.isfinite(x), x, np.asarray(entry["median"]))
        x = (x - np.asarray(entry["mean"])) / np.asarray(entry["sd"])
    logit = float(np.asarray(art["logistic"]["coef"]) @ x + art["logistic"]["intercept"])
    return bool(logit >= 0.0) if math.isfinite(logit) else True
