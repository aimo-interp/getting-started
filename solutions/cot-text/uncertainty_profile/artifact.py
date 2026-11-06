"""Artifact schema, validation, lookup, and serialization."""

from __future__ import annotations

import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import joblib

from uncertainty_profile.config import FEATURE_NAMES, GenerationConfidenceConfig


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "uncertainty_regressor"
ARTIFACT_DIR_NAME = "uncertainty_artifacts"


class Predictor(Protocol):
    """Structural interface required from a bundled regression estimator."""

    def predict(self, features: object) -> object:
        """Return one continuous score per feature row."""


@dataclass(frozen=True)
class UncertaintyArtifact:
    """Validated runtime view of an uncertainty-regression artifact.

    Attributes:
        model_id: Exact Hugging Face model identifier supported by the artifact.
        feature_names: Ordered feature schema expected by the estimator.
        generation_config: Settings used when its training features were generated.
        estimator: Fitted regressor exposing a ``predict`` method.
        decision_threshold: Scores below this value are labeled robust.
        selected_params: Selected estimator family and hyperparameters.
        cv_results: Cross-validation selection metrics and candidate results.
        training_provenance: Dataset and training-process metadata.
        library_versions: Versions used to create the artifact.
        raw: Complete serialized artifact payload.
    """

    model_id: str
    feature_names: tuple[str, ...]
    generation_config: GenerationConfidenceConfig
    estimator: Predictor
    decision_threshold: float
    selected_params: dict[str, object]
    cv_results: dict[str, object]
    training_provenance: dict[str, object]
    library_versions: dict[str, str]
    raw: dict[str, object]


def safe_model_id(model_id: str) -> str:
    """Convert a model identifier into a portable artifact filename stem.

    Args:
        model_id: Model identifier to sanitize.

    Returns:
        Filename-safe representation of ``model_id``.
    """

    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def default_artifact_path(model_id: str) -> Path:
    """Return the bundled artifact path for a model identifier.

    Args:
        model_id: Exact model identifier.

    Returns:
        Expected artifact path inside the solution package.
    """

    solution_root = Path(__file__).resolve().parents[1]
    return solution_root / ARTIFACT_DIR_NAME / f"{safe_model_id(model_id)}.joblib"


def validate_artifact_payload(
    payload: object,
    *,
    path: Path | None = None,
) -> UncertaintyArtifact:
    """Validate an artifact payload and expose its typed runtime fields.

    Args:
        payload: Deserialized object to validate.
        path: Optional source path used in validation errors.

    Returns:
        Validated artifact view.

    Raises:
        RuntimeError: If the payload does not match the runtime schema.
    """

    label = path.name if path is not None else "artifact"
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a dictionary")

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"{label} has unsupported schema_version {payload.get('schema_version')!r}"
        )

    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise RuntimeError(
            f"{label} has unsupported artifact_type {payload.get('artifact_type')!r}"
        )

    model_id = payload.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"{label} has invalid model_id")

    feature_names = payload.get("feature_names")
    if not isinstance(feature_names, (list, tuple)) or not all(
        isinstance(name, str) for name in feature_names
    ):
        raise RuntimeError(f"{label} has invalid feature_names")

    feature_names_tuple = tuple(feature_names)
    if feature_names_tuple != FEATURE_NAMES:
        raise RuntimeError(f"{label} feature ordering does not match the runtime")

    raw_generation_config = payload.get("generation_config")
    if not isinstance(raw_generation_config, Mapping):
        raise RuntimeError(f"{label} has invalid generation_config")

    try:
        generation_config = GenerationConfidenceConfig.from_dict(
            raw_generation_config
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} has invalid generation_config: {exc}") from exc

    estimator = payload.get("estimator")
    if not callable(getattr(estimator, "predict", None)):
        raise RuntimeError(f"{label} estimator has no callable predict method")

    threshold = payload.get("decision_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
    ):
        raise RuntimeError(f"{label} has invalid decision_threshold")

    mapping_fields = (
        "selected_params",
        "cv_results",
        "training_provenance",
        "library_versions",
    )
    for field in mapping_fields:
        if not isinstance(payload.get(field), Mapping):
            raise RuntimeError(f"{label} has invalid {field}")

    return UncertaintyArtifact(
        model_id=model_id,
        feature_names=feature_names_tuple,
        generation_config=generation_config,
        estimator=cast(Predictor, estimator),
        decision_threshold=float(threshold),
        selected_params=dict(payload["selected_params"]),
        cv_results=dict(payload["cv_results"]),
        training_provenance=dict(payload["training_provenance"]),
        library_versions={
            str(key): str(value)
            for key, value in payload["library_versions"].items()
        },
        raw=payload,
    )


def load_artifact(model_id: str) -> UncertaintyArtifact | None:
    """Load the artifact matching an exact model identifier.

    Args:
        model_id: Exact model identifier requested by Codabench.

    Returns:
        Validated artifact, or ``None`` when no matching artifact exists.

    Raises:
        RuntimeError: If a present artifact fails schema validation.
    """

    path = default_artifact_path(model_id)
    if not path.is_file():
        return None
    artifact = validate_artifact_payload(joblib.load(path), path=path)
    if artifact.model_id != model_id:
        return None
    return artifact


def make_artifact_payload(
    *,
    model_id: str,
    generation_config: GenerationConfidenceConfig,
    estimator: Predictor,
    decision_threshold: float,
    selected_params: Mapping[str, object],
    cv_results: Mapping[str, object],
    training_provenance: Mapping[str, object],
    library_versions: Mapping[str, str],
) -> dict[str, object]:
    """Build and validate a serializable artifact payload.

    Args:
        model_id: Exact model identifier supported by the artifact.
        generation_config: Feature-generation settings used during training.
        estimator: Fitted regression estimator.
        decision_threshold: Score threshold below which labels are robust.
        selected_params: Selected model family and hyperparameters.
        cv_results: Cross-validation metrics and selection details.
        training_provenance: Dataset and training metadata.
        library_versions: Training environment versions.

    Returns:
        Validated artifact payload ready for serialization.

    Raises:
        RuntimeError: If any value violates the artifact schema.
    """

    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "model_id": model_id,
        "feature_names": list(FEATURE_NAMES),
        "generation_config": generation_config.to_dict(),
        "estimator": estimator,
        "decision_threshold": float(decision_threshold),
        "selected_params": dict(selected_params),
        "cv_results": dict(cv_results),
        "training_provenance": dict(training_provenance),
        "library_versions": dict(library_versions),
    }
    validate_artifact_payload(payload)
    return payload


def dump_artifact(payload: Mapping[str, object], path: Path) -> Path:
    """Atomically serialize a validated artifact.

    Args:
        payload: Artifact payload to serialize.
        path: Final joblib destination.

    Returns:
        ``path`` after the artifact has been committed.

    Raises:
        RuntimeError: If the payload fails validation.
        OSError: If the destination cannot be written atomically.
    """

    serialized = dict(payload)
    validate_artifact_payload(serialized, path=path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".joblib",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    try:
        joblib.dump(serialized, temporary_path, compress=3)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return path
