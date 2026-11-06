"""Submission-side uncertainty feature extraction and regression inference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from uncertainty_profile.artifact import load_artifact
from uncertainty_profile.config import resolve_checkpoint_model_id
from uncertainty_profile.extraction import (
    extract_generation_features,
    load_model_and_tokenizer,
    release_model,
)


def predict_robustness(model_id: str, problem_texts: Sequence[str]) -> list[bool]:
    """Predict robustness without training or network access.

    Args:
        model_id: Codabench model alias or exact checkpoint identifier.
        problem_texts: Problem statements in submission order.

    Returns:
        One native Python boolean per input problem. If no matching artifact is
        bundled, every prediction is ``False``.

    Raises:
        RuntimeError: If model inference or regressor prediction is malformed.
    """

    texts = list(problem_texts)
    if not texts:
        return []
    checkpoint_model_id = resolve_checkpoint_model_id(model_id)
    artifact = load_artifact(checkpoint_model_id)
    if artifact is None:
        return [False] * len(texts)

    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizerBase | None = None
    try:
        model, tokenizer = load_model_and_tokenizer(
            artifact.model_id,
            local_files_only=True,
        )
        feature_rows = extract_generation_features(
            model=model,
            tokenizer=tokenizer,
            problem_texts=texts,
            config=artifact.generation_config,
            include_generated_text=False,
        )
        features = pd.DataFrame.from_records(
            feature_rows,
            columns=list(artifact.feature_names),
        )
        scores = np.asarray(artifact.estimator.predict(features), dtype=float)
        if scores.shape != (len(texts),):
            raise RuntimeError(
                "regressor returned scores with shape "
                f"{scores.shape}; expected ({len(texts)},)"
            )
        if not np.isfinite(scores).all():
            raise RuntimeError("regressor returned non-finite scores")
        return [bool(score < artifact.decision_threshold) for score in scores]
    finally:
        if model is not None or tokenizer is not None:
            model = None
            tokenizer = None
            release_model()
