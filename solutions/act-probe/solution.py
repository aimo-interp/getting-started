"""Codabench entry point for the CoT-activation diff-in-means probe (cvbench act_probe, resp_mean)."""

from act_probe.inference import predict_robustness


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    """Predict one native boolean robustness label per problem, in input order."""
    return predict_robustness(model_id, list(problems))
