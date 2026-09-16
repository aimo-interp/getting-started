"""Codabench entry point for the trained representation probe.

This is the *only* file the evaluator calls directly. It must expose:

* an ``are_robust(model_id, reasoning_effort, problems)`` function accepting problem strings and
  returning one bool per problem.

All heavy lifting (loading the probe artifact, running the model, scoring the
probes) lives in :mod:`probe_inference` so this file stays a thin adapter
between the Codabench contract and the implementation.
"""

from probe_inference import predict_robustness


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    """Predict robustness for a batch while preserving problem order.

    This baseline accepts reasoning_effort but does not use it in predictions.

    Returns a list of native Python ``bool`` values, one per input problem, in
    the same order as ``problems``. ``True`` means the model is predicted to be
    robust on that problem, ``False`` means not robust.
    """
    return predict_robustness(model_id, problems)
