"""Scalar uncertainty metrics computed over generated tokens."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np


MetricValue: TypeAlias = float | int
ConfidenceMetrics: TypeAlias = dict[str, MetricValue]


def _mean(values: np.ndarray) -> float:
    """Return the mean of an array or NaN when it is empty.

    Args:
        values: Values to average.

    Returns:
        Arithmetic mean, or NaN for an empty array.
    """

    return float(np.mean(values)) if values.size else float("nan")


def compute_generation_confidence_metrics(
    *,
    log_probs: np.ndarray,
    probs: np.ndarray,
    entropy: np.ndarray,
    top1_probs: np.ndarray,
    top2_margins: np.ndarray,
    selected_is_top1: np.ndarray,
    min_k_fraction: float,
    high_conf_threshold: float,
    low_conf_threshold: float,
) -> ConfidenceMetrics:
    """Summarize token-level confidence arrays into the feature set.

    Args:
        log_probs: Log-probabilities of the generated tokens.
        probs: Probabilities of the generated tokens.
        entropy: Predictive entropy at each generation step.
        top1_probs: Probability of the most likely token at each step.
        top2_margins: Probability margin between the two likeliest tokens.
        selected_is_top1: Whether each generated token was the top prediction.
        min_k_fraction: Fraction of least-likely tokens used by Min-K.
        high_conf_threshold: Inclusive high-confidence token threshold.
        low_conf_threshold: Inclusive low-confidence token threshold.

    Returns:
        Ordered-name-compatible scalar uncertainty features. Empty token arrays
        produce a zero token count and NaN-valued continuous features.

    Raises:
        ValueError: If ``min_k_fraction`` is outside ``(0, 1]``.
    """

    if not 0.0 < min_k_fraction <= 1.0:
        raise ValueError(f"min_k_fraction must be in (0, 1], got {min_k_fraction}")

    mean_logprob = _mean(log_probs)
    mean_nll = -mean_logprob if np.isfinite(mean_logprob) else float("nan")
    perplexity = float(np.exp(mean_nll)) if np.isfinite(mean_nll) else float("nan")

    if log_probs.size:
        k = max(1, int(np.ceil(log_probs.size * min_k_fraction)))
        min_k_logprob = float(np.mean(np.sort(log_probs)[:k]))
    else:
        min_k_logprob = float("nan")

    return {
        "generation_num_tokens": int(log_probs.size),
        "generation_mean_logprob": mean_logprob,
        "generation_mean_prob": _mean(probs),
        "generation_mean_nll": mean_nll,
        "generation_ppl": perplexity,
        "generation_min_logprob": (
            float(np.min(log_probs)) if log_probs.size else float("nan")
        ),
        "generation_max_logprob": (
            float(np.max(log_probs)) if log_probs.size else float("nan")
        ),
        "generation_std_logprob": (
            float(np.std(log_probs)) if log_probs.size else float("nan")
        ),
        "generation_min_k_logprob": min_k_logprob,
        "generation_mean_entropy": _mean(entropy),
        "generation_mean_top1_prob": _mean(top1_probs),
        "generation_mean_top2_margin": _mean(top2_margins),
        "generation_frac_selected_is_top1": _mean(
            selected_is_top1.astype(float)
        ),
        "generation_frac_high_conf_tokens": _mean(
            (probs >= high_conf_threshold).astype(float)
        ),
        "generation_frac_low_conf_tokens": _mean(
            (probs <= low_conf_threshold).astype(float)
        ),
    }
