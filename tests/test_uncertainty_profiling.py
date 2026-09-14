"""Unit tests for the uncertainty-profiling runtime and training selection."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import get_type_hints
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
SOLUTION_ROOT = ROOT / "solutions" / "uncertainty-profiling"
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))

from uncertainty_profile.artifact import (  # noqa: E402
    make_artifact_payload,
    validate_artifact_payload,
)
from uncertainty_profile.config import (  # noqa: E402
    DEEPSEEK_MODEL_ID,
    FEATURE_NAMES,
    GenerationConfidenceConfig,
)
from uncertainty_profile.extraction import (  # noqa: E402
    GenerationConfidenceCollector,
    build_valid_token_mask,
    compute_batch_features_from_scores,
)
from uncertainty_profile.inference import predict_robustness  # noqa: E402
from uncertainty_profile.metrics import (  # noqa: E402
    compute_generation_confidence_metrics,
)


def load_training_module() -> ModuleType:
    """Load the training script as an importable test module."""

    path = SOLUTION_ROOT / "scripts" / "train_uncertainty_regressor.py"
    spec = importlib.util.spec_from_file_location("uncertainty_training_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load training script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_solution_module() -> ModuleType:
    """Load the uncertainty submission entry point for contract tests."""

    path = SOLUTION_ROOT / "solution.py"
    spec = importlib.util.spec_from_file_location("uncertainty_solution_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load uncertainty solution")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAINING = load_training_module()
SOLUTION = load_solution_module()


class FixedEstimator:
    """Test estimator returning a fixed prefix of predictions."""

    def __init__(
        self,
        predictions: Sequence[float] | Sequence[Sequence[float]],
    ) -> None:
        """Store deterministic predictions for later calls."""

        self.predictions = np.asarray(predictions, dtype=float)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return one fixed prediction entry per feature row."""

        return self.predictions[: len(frame)]


class ConfidenceMetricTests(unittest.TestCase):
    def test_metrics_match_expected_values(self) -> None:
        """Verify every confidence summary against hand-computed values."""

        metrics = compute_generation_confidence_metrics(
            log_probs=np.array([-1.0, -2.0, -3.0]),
            probs=np.exp(np.array([-1.0, -2.0, -3.0])),
            entropy=np.array([0.2, 0.4, 0.6]),
            top1_probs=np.array([0.7, 0.8, 0.9]),
            top2_margins=np.array([0.4, 0.5, 0.6]),
            selected_is_top1=np.array([1.0, 0.0, 1.0]),
            min_k_fraction=0.5,
            high_conf_threshold=0.3,
            low_conf_threshold=0.1,
        )
        self.assertEqual(metrics["generation_num_tokens"], 3)
        self.assertAlmostEqual(metrics["generation_mean_logprob"], -2.0)
        self.assertAlmostEqual(metrics["generation_mean_nll"], 2.0)
        self.assertAlmostEqual(metrics["generation_ppl"], np.exp(2.0))
        self.assertAlmostEqual(metrics["generation_min_k_logprob"], -2.5)
        self.assertAlmostEqual(metrics["generation_mean_entropy"], 0.4)
        self.assertAlmostEqual(
            metrics["generation_frac_selected_is_top1"], 2.0 / 3.0
        )

    def test_empty_response_produces_imputable_nans(self) -> None:
        """Verify empty responses retain a zero count and imputable NaNs."""

        metrics = compute_generation_confidence_metrics(
            log_probs=np.array([]),
            probs=np.array([]),
            entropy=np.array([]),
            top1_probs=np.array([]),
            top2_margins=np.array([]),
            selected_is_top1=np.array([]),
            min_k_fraction=0.2,
            high_conf_threshold=0.9,
            low_conf_threshold=0.1,
        )
        self.assertEqual(metrics["generation_num_tokens"], 0)
        for name in FEATURE_NAMES:
            self.assertTrue(np.isnan(metrics[name]), name)

    def test_padding_and_all_eos_ids_are_masked(self) -> None:
        """Verify padding and every configured EOS token are excluded."""

        token_ids = torch.tensor([[1, 4, 0], [2, 3, 5]])
        mask = build_valid_token_mask(
            token_ids,
            pad_token_id=0,
            eos_token_ids={4, 5},
        )
        self.assertEqual(
            mask.tolist(),
            [[True, False, False], [True, True, False]],
        )

    def test_batch_score_features_preserve_batch_order(self) -> None:
        """Verify post-hoc batch features remain in response order."""

        sequences = torch.tensor(
            [
                [7, 8, 1, 2, 4],
                [7, 8, 3, 4, 0],
            ]
        )
        scores = []
        for step_tokens in ([1, 3], [2, 4], [4, 0]):
            logits = torch.full((2, 5), -4.0)
            for row, token_id in enumerate(step_tokens):
                logits[row, token_id] = 4.0
            scores.append(logits)
        rows, generated = compute_batch_features_from_scores(
            sequences=sequences,
            scores=scores,
            prompt_length=2,
            pad_token_id=0,
            eos_token_ids={4},
            config=GenerationConfidenceConfig(max_new_tokens=3),
        )
        self.assertEqual(generated.tolist(), [[1, 2, 4], [3, 4, 0]])
        self.assertEqual(rows[0]["generation_num_tokens"], 2)
        self.assertEqual(rows[1]["generation_num_tokens"], 1)
        self.assertGreater(
            rows[1]["generation_mean_prob"],
            0.99,
        )

    def test_online_collector_matches_posthoc_greedy_features(self) -> None:
        """Verify online collection preserves the original feature values."""

        sequences = torch.tensor([[7, 8, 1, 2, 4], [7, 8, 3, 4, 0]])
        generated_ids = [[1, 2, 4], [3, 4, 0]]
        scores = []
        collector = GenerationConfidenceCollector()
        for step, step_tokens in enumerate(zip(*generated_ids)):
            logits = torch.full((2, 5), -4.0)
            for row, token_id in enumerate(step_tokens):
                logits[row, token_id] = 4.0
            scores.append(logits)
            collector(sequences[:, : 2 + step], logits)

        arguments = {
            "sequences": sequences,
            "prompt_length": 2,
            "pad_token_id": 0,
            "eos_token_ids": {4},
            "config": GenerationConfidenceConfig(max_new_tokens=3),
        }
        expected, _ = compute_batch_features_from_scores(scores=scores, **arguments)
        actual, _ = collector.compute_features(**arguments)
        for expected_row, actual_row in zip(expected, actual):
            for name in ("generation_num_tokens", *FEATURE_NAMES):
                self.assertAlmostEqual(expected_row[name], actual_row[name], places=6)


class ArtifactAndInferenceTests(unittest.TestCase):
    def test_entrypoint_accepts_and_forwards_problem_strings(self) -> None:
        """Verify the Codabench adapter accepts and forwards ``list[str]``."""

        problems = ["first problem", "second problem"]
        with mock.patch.object(
            SOLUTION,
            "predict_robustness",
            return_value=[True, False],
        ) as predictor:
            predictions = SOLUTION.are_robust("example/model", "default", problems)

        self.assertEqual(predictions, [True, False])
        predictor.assert_called_once_with(
            model_id="example/model",
            problem_texts=problems,
        )
        annotations = get_type_hints(SOLUTION.are_robust)
        self.assertEqual(annotations["problems"], list[str])
        self.assertEqual(annotations["return"], list[bool])
        self.assertFalse(hasattr(SOLUTION, "Problem"))

    def setUp(self) -> None:
        """Build one valid artifact payload for each test."""

        frame = pd.DataFrame([[0.0] * len(FEATURE_NAMES)], columns=FEATURE_NAMES)
        estimator = DummyRegressor(strategy="constant", constant=0.1).fit(
            frame, [0.0]
        )
        self.payload: dict[str, object] = make_artifact_payload(
            model_id="example/model",
            generation_config=GenerationConfidenceConfig(max_new_tokens=3),
            estimator=estimator,
            decision_threshold=0.2,
            selected_params={"family": "dummy"},
            cv_results={"best": {}},
            training_provenance={"source_dataset": "example/data"},
            library_versions={"scikit_learn": "test"},
        )

    def test_artifact_round_trip_and_schema_validation(self) -> None:
        """Verify artifact serialization and ordered feature validation."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.joblib"
            joblib.dump(self.payload, path)
            artifact = validate_artifact_payload(joblib.load(path), path=path)
            self.assertEqual(artifact.model_id, "example/model")
            self.assertEqual(artifact.feature_names, FEATURE_NAMES)
            self.assertEqual(artifact.decision_threshold, 0.2)
        invalid = dict(self.payload, feature_names=list(reversed(FEATURE_NAMES)))
        with self.assertRaisesRegex(RuntimeError, "feature ordering"):
            validate_artifact_payload(invalid)

    def test_unknown_model_falls_back_to_native_false(self) -> None:
        """Verify unsupported models return native false values without loading."""

        with (
            mock.patch(
                "uncertainty_profile.inference.load_artifact", return_value=None
            ) as artifact_loader,
            mock.patch(
                "uncertainty_profile.inference.load_model_and_tokenizer"
            ) as model_loader,
        ):
            predictions = predict_robustness("unknown/model", ["one", "two"])
        self.assertEqual(predictions, [False, False])
        self.assertTrue(all(type(value) is bool for value in predictions))
        artifact_loader.assert_called_once_with("unknown/model")
        model_loader.assert_not_called()

    def test_competition_alias_loads_deepseek_checkpoint(self) -> None:
        """Verify the Codabench alias resolves before artifact and model loading."""

        artifact = SimpleNamespace(
            model_id=DEEPSEEK_MODEL_ID,
            feature_names=FEATURE_NAMES,
            generation_config=GenerationConfidenceConfig(max_new_tokens=3),
            estimator=FixedEstimator([0.1]),
            decision_threshold=0.2,
        )
        rows = [{name: 0.0 for name in FEATURE_NAMES}]
        with (
            mock.patch(
                "uncertainty_profile.inference.load_artifact",
                return_value=artifact,
            ) as artifact_loader,
            mock.patch(
                "uncertainty_profile.inference.load_model_and_tokenizer",
                return_value=(object(), object()),
            ) as model_loader,
            mock.patch(
                "uncertainty_profile.inference.extract_generation_features",
                return_value=rows,
            ),
            mock.patch("uncertainty_profile.inference.release_model"),
        ):
            predictions = predict_robustness("qwen3-8b:low", ["one"])

        self.assertEqual(predictions, [True])
        artifact_loader.assert_called_once_with(DEEPSEEK_MODEL_ID)
        model_loader.assert_called_once_with(
            DEEPSEEK_MODEL_ID,
            local_files_only=True,
        )

    def test_known_model_uses_less_than_threshold_and_preserves_order(self) -> None:
        """Verify score direction, ordering, and native boolean conversion."""

        artifact = SimpleNamespace(
            model_id="example/model",
            feature_names=FEATURE_NAMES,
            generation_config=GenerationConfidenceConfig(max_new_tokens=3),
            estimator=FixedEstimator([0.1, 0.3]),
            decision_threshold=0.2,
        )
        rows = [
            {name: float(index) for name in FEATURE_NAMES}
            for index in (1, 2)
        ]
        with (
            mock.patch(
                "uncertainty_profile.inference.load_artifact",
                return_value=artifact,
            ),
            mock.patch(
                "uncertainty_profile.inference.load_model_and_tokenizer",
                return_value=(object(), object()),
            ),
            mock.patch(
                "uncertainty_profile.inference.extract_generation_features",
                return_value=rows,
            ),
            mock.patch("uncertainty_profile.inference.release_model") as release,
        ):
            predictions = predict_robustness("example/model", ["one", "two"])
        self.assertEqual(predictions, [True, False])
        self.assertTrue(all(type(value) is bool for value in predictions))
        release.assert_called_once_with()

    def test_malformed_regressor_scores_are_rejected(self) -> None:
        """Verify wrong-shaped and non-finite regressor outputs fail safely."""

        artifact = SimpleNamespace(
            model_id="example/model",
            feature_names=FEATURE_NAMES,
            generation_config=GenerationConfidenceConfig(max_new_tokens=3),
            estimator=FixedEstimator([[float("nan")]]),
            decision_threshold=0.2,
        )
        rows = [{name: 0.0 for name in FEATURE_NAMES}]
        with (
            mock.patch(
                "uncertainty_profile.inference.load_artifact",
                return_value=artifact,
            ),
            mock.patch(
                "uncertainty_profile.inference.load_model_and_tokenizer",
                return_value=(object(), object()),
            ),
            mock.patch(
                "uncertainty_profile.inference.extract_generation_features",
                return_value=rows,
            ),
            mock.patch("uncertainty_profile.inference.release_model") as release,
            self.assertRaisesRegex(RuntimeError, "scores with shape"),
        ):
            predict_robustness("example/model", ["one"])
        release.assert_called_once_with()

        artifact.estimator = FixedEstimator([float("nan")])
        with (
            mock.patch(
                "uncertainty_profile.inference.load_artifact",
                return_value=artifact,
            ),
            mock.patch(
                "uncertainty_profile.inference.load_model_and_tokenizer",
                return_value=(object(), object()),
            ),
            mock.patch(
                "uncertainty_profile.inference.extract_generation_features",
                return_value=rows,
            ),
            mock.patch("uncertainty_profile.inference.release_model"),
            self.assertRaisesRegex(RuntimeError, "non-finite"),
        ):
            predict_robustness("example/model", ["one"])

    def test_config_and_threshold_reject_boolean_numeric_values(self) -> None:
        """Verify booleans cannot masquerade as integer or float metadata."""

        values = GenerationConfidenceConfig().to_dict()
        values["max_new_tokens"] = True
        with self.assertRaisesRegex(ValueError, "positive integer"):
            GenerationConfidenceConfig.from_dict(values)

        invalid = dict(self.payload, decision_threshold=True)
        with self.assertRaisesRegex(RuntimeError, "decision_threshold"):
            validate_artifact_payload(invalid)


class TrainingSelectionTests(unittest.TestCase):
    @staticmethod
    def synthetic_frame() -> pd.DataFrame:
        """Build balanced grouped data with deterministic feature values."""

        records = []
        for index in range(20):
            robust = index % 2 == 0
            row = {
                name: float(robust) + feature_index * 0.001
                for feature_index, name in enumerate(FEATURE_NAMES)
            }
            row.update(
                {
                    "absolute_accuracy_decay": 0.0 if robust else 0.8,
                    "model_is_robust": robust,
                    "original_problem": f"problem-{index}",
                    "feature_model_id": "example/model",
                    "generation_config_json": json.dumps(
                        GenerationConfidenceConfig(max_new_tokens=3).to_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "source_dataset": "example/data",
                    "source_revision": "abc123",
                    "model_id": "source/model",
                }
            )
            records.append(row)
        return pd.DataFrame.from_records(records)

    def test_grouped_splits_have_no_leakage(self) -> None:
        """Verify no problem group crosses a training-validation boundary."""

        frame = self.synthetic_frame()
        TRAINING.validate_feature_data(frame, n_splits=5)
        splits = TRAINING.make_grouped_splits(frame, n_splits=5, seed=42)
        for train_indices, validation_indices in splits:
            train_groups = set(frame.iloc[train_indices]["original_problem"])
            validation_groups = set(frame.iloc[validation_indices]["original_problem"])
            self.assertFalse(train_groups & validation_groups)

    def test_threshold_direction_and_selection_are_deterministic(self) -> None:
        """Verify threshold direction and repeated model selection results."""

        threshold, balanced, accuracy = TRAINING.select_threshold(
            [True, True, False, False],
            [0.1, 0.2, 0.8, 0.9],
        )
        self.assertGreater(threshold, 0.2)
        self.assertLess(threshold, 0.8)
        self.assertEqual(balanced, 1.0)
        self.assertEqual(accuracy, 1.0)

        frame = self.synthetic_frame()
        specs = [
            TRAINING.CandidateSpec(
                family="extra_trees",
                params={
                    "n_estimators": 5,
                    "max_depth": None,
                    "min_samples_leaf": 1,
                    "max_features": 1.0,
                },
            )
        ]
        first = TRAINING.run_model_selection(
            frame, specs=specs, n_splits=5, seed=42
        )
        second = TRAINING.run_model_selection(
            frame, specs=specs, n_splits=5, seed=42
        )
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])

    def test_conflicting_group_labels_are_rejected(self) -> None:
        """Verify repeated groups cannot carry inconsistent binary labels."""

        frame = self.synthetic_frame()
        conflicting = frame.iloc[[0]].copy()
        conflicting[TRAINING.LABEL_COLUMN] = False
        frame = pd.concat([frame, conflicting], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "conflicting robustness labels"):
            TRAINING.validate_feature_data(frame, n_splits=5)

    def test_direct_partial_cache_is_rejected(self) -> None:
        """Verify training refuses a directly supplied partial cache file."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.partial.parquet"
            path.touch()
            with self.assertRaisesRegex(ValueError, "incomplete feature cache"):
                list(TRAINING.iter_parquet_files(path))

    def test_artifact_export_is_byte_deterministic(self) -> None:
        """Verify repeated seeded exports produce identical artifact bytes."""

        frame = self.synthetic_frame()
        spec = TRAINING.CandidateSpec(
            family="extra_trees",
            params={
                "n_estimators": 5,
                "max_depth": None,
                "min_samples_leaf": 1,
                "max_features": 1.0,
            },
        )
        checksums = []
        with tempfile.TemporaryDirectory() as directory:
            for index in range(2):
                pipeline = TRAINING.build_pipeline(spec, seed=42)
                pipeline.fit(
                    frame[list(FEATURE_NAMES)],
                    frame[TRAINING.TARGET_COLUMN],
                )
                payload = make_artifact_payload(
                    model_id="example/model",
                    generation_config=GenerationConfidenceConfig(max_new_tokens=3),
                    estimator=pipeline,
                    decision_threshold=0.2,
                    selected_params={"family": spec.family, **spec.params},
                    cv_results={"best": {"balanced_accuracy": 1.0}},
                    training_provenance={"source_dataset": "example/data"},
                    library_versions={"scikit_learn": "test"},
                )
                path = Path(directory) / f"artifact-{index}.joblib"
                TRAINING.dump_artifact(payload, path)
                checksums.append(TRAINING.sha256_file(path))
        self.assertEqual(checksums[0], checksums[1])


if __name__ == "__main__":
    unittest.main()
