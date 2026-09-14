"""Codabench entry point for the uncertainty-profiling baseline."""

from uncertainty_profile.inference import predict_robustness


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    """Predict one native boolean robustness label per problem.

    Args:
        model_id: Codabench model alias or exact checkpoint identifier.
        reasoning_effort: Evaluated reasoning effort, unused by this baseline.
        problems: Mathematical problem statements in evaluation order.

    Returns:
        Robustness predictions in the same order as ``problems``.
    """

    return predict_robustness(model_id=model_id, problem_texts=problems)
