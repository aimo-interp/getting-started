"""Codabench entry point for the CoT-window diff-in-means probe (cvbench seg_probe, windows 0-5 + 5-9)."""

from seg_probe.inference import predict_robustness


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    """Predict one native boolean robustness label per problem, in input order."""
    return predict_robustness(model_id, list(problems), reasoning_effort)
