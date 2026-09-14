"""Tests for the Codabench participant entry-point contract."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
INGESTION_PATH = ROOT / "components" / "ingestion_program" / "ingestion.py"


def load_ingestion_module() -> ModuleType:
    """Load the ingestion program as a test module."""

    spec = importlib.util.spec_from_file_location(
        "ingestion_contract_test",
        INGESTION_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load ingestion program")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


INGESTION = load_ingestion_module()


def write_evaluation_data(root: Path, model_id: str = "model-a:low"):
    input_dir = root / "dataset" / "input"
    reference_dir = root / "dataset" / "reference"
    input_dir.mkdir(parents=True)
    reference_dir.mkdir(parents=True)
    cases = [
        {
            "id": "case-001",
            "model_id": model_id,
            "problem": "What is 2 + 2?",
        },
        {
            "id": "case-002",
            "model_id": model_id,
            "problem": "What is 3 + 3?",
        },
    ]
    labels = [
        {"id": "case-001", "is_robust": True},
        {"id": "case-002", "is_robust": False},
    ]
    (input_dir / "cases.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in cases),
        encoding="utf-8",
    )
    (reference_dir / "labels.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in labels),
        encoding="utf-8",
    )
    return input_dir, reference_dir


class IngestionContractTests(unittest.TestCase):
    """Exercise entry-point validation and problem-string forwarding."""

    @staticmethod
    def write_solution(directory: Path, source: str) -> None:
        """Write a temporary participant entry point."""

        directory.mkdir(parents=True)
        (directory / "solution.py").write_text(source, encoding="utf-8")

    def test_optional_reasoning_effort_and_legacy_batching(self):
        signatures = [
            ("model_id, problems", False),
            ("model_id, reasoning_effort, problems", True),
            ("model_id, reasoning_effort='own-default', *, problems", True),
            ("model_id, problems, **kwargs", True),
            ("model_id, problems, /", False),
        ]
        for signature, accepts_effort in signatures:
            with self.subTest(signature=signature), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                input_dir, _ = write_evaluation_data(root)
                cases = [
                    {"id": "a", "model_id": "m", "problem": "first"},
                    {"id": "b", "model_id": "m", "problem": "second",
                     "reasoning_effort": "medium"},
                    {"id": "c", "model_id": "m", "problem": "third",
                     "reasoning_effort": "default"},
                ]
                (input_dir / "cases.jsonl").write_text(
                    "".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8"
                )
                submission = root / "submission"
                submission.mkdir()
                effort_expression = (
                    "kwargs['reasoning_effort']" if "**kwargs" in signature
                    else "reasoning_effort" if accepts_effort else "None"
                )
                (submission / "solution.py").write_text(
                    "calls = []\n"
                    f"def are_robust({signature}):\n"
                    f"    calls.append((model_id, problems, {effort_expression}))\n"
                    "    return [problem != 'second' for problem in problems]\n",
                    encoding="utf-8",
                )
                INGESTION.run(input_dir, root / "output", submission)
                calls = sys.modules["participant_solution"].calls
                expected = (
                    [("m", ["first", "third"], "default"), ("m", ["second"], "medium")]
                    if accepts_effort else [("m", ["first", "second", "third"], None)]
                )
                self.assertEqual(calls, expected)
                predictions = list(INGESTION.read_jsonl(root / "output/predictions.jsonl"))
                self.assertEqual([row["id"] for row in predictions], ["a", "b", "c"])
                self.assertEqual([row["is_robust"] for row in predictions], [True, False, True])
                self.assertTrue(all(row["valid"] for row in predictions))

    def test_reasoning_effort_internal_type_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir, _ = write_evaluation_data(root)
            submission = root / "submission"
            submission.mkdir()
            (submission / "solution.py").write_text(
                "calls = []\n"
                "def are_robust(model_id, reasoning_effort, problems):\n"
                "    calls.append(reasoning_effort)\n"
                "    raise TypeError('internal failure')\n", encoding="utf-8"
            )
            INGESTION.run(input_dir, root / "output", submission)
            self.assertEqual(sys.modules["participant_solution"].calls, ["default"])
            predictions = list(INGESTION.read_jsonl(root / "output/predictions.jsonl"))
            self.assertTrue(all(not row["valid"] for row in predictions))

    def test_invalid_reasoning_effort_is_rejected(self):
        for effort in [None, "", "  ", 1, False, []]:
            with self.subTest(effort=effort), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "cases.jsonl"
                path.write_text(json.dumps({
                    "id": "a", "model_id": "m", "problem": "p", "reasoning_effort": effort
                }), encoding="utf-8")
                with self.assertRaisesRegex(INGESTION.IngestionError, "reasoning_effort"):
                    INGESTION.load_cases(path)

    def test_accepts_documented_string_list_signature(self) -> None:
        """Accept the documented annotated entry-point signature."""

        with tempfile.TemporaryDirectory() as temporary:
            submission = Path(temporary) / "submission"
            self.write_solution(
                submission,
                "def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) "
                "-> list[bool]:\n"
                "    return [False for _ in problems]\n",
            )
            module = INGESTION.load_solution(submission)
        self.assertTrue(callable(module.are_robust))

    def test_annotations_are_not_required(self) -> None:
        """Require a callable without imposing runtime annotation checks."""

        with tempfile.TemporaryDirectory() as temporary:
            submission = Path(temporary) / "submission"
            self.write_solution(
                submission,
                "def are_robust(model_id, problems):\n"
                "    return [False for _ in problems]\n",
            )
            module = INGESTION.load_solution(submission)
        self.assertTrue(callable(module.are_robust))

    def test_run_passes_only_original_problem_strings(self) -> None:
        """Pass ordered original-problem strings to participant code."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            submission = root / "submission"
            input_dir.mkdir()
            self.write_solution(
                submission,
                "def are_robust(model_id: str, problems: list[str]) "
                "-> list[bool]:\n"
                "    assert model_id == 'example/model'\n"
                "    assert all(type(problem) is str for problem in problems)\n"
                "    return [problem.startswith('robust') for problem in problems]\n",
            )
            cases = [
                {
                    "id": "first",
                    "model_id": "example/model",
                    "problem": "robust example",
                },
                {
                    "id": "second",
                    "model_id": "example/model",
                    "problem": "non-robust example",
                },
            ]
            (input_dir / "cases.jsonl").write_text(
                "".join(json.dumps(case) + "\n" for case in cases),
                encoding="utf-8",
            )

            INGESTION.run(input_dir, output_dir, submission)
            predictions = [
                json.loads(line)
                for line in (output_dir / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(
            [prediction["is_robust"] for prediction in predictions],
            [True, False],
        )
        self.assertTrue(all(prediction["valid"] for prediction in predictions))

    def test_main_track_rejects_small_marker_case_insensitively(self) -> None:
        """Reject a small-track archive before importing participant code."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            submission = root / "submission"
            self.write_solution(
                submission,
                "raise RuntimeError('solution must not be imported')\n",
            )
            (submission / "SMALL.TXT").write_bytes(b"")

            with self.assertRaisesRegex(
                INGESTION.IngestionError,
                "not allowed for a main-track submission",
            ):
                INGESTION.run(root / "missing-input", root / "output", submission)

    def test_small_track_requires_marker_before_importing_solution(self) -> None:
        """Reject a missing small marker before importing participant code."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            submission = root / "submission"
            self.write_solution(
                submission,
                "raise RuntimeError('solution must not be imported')\n",
            )

            with self.assertRaisesRegex(
                INGESTION.IngestionError,
                "must contain small.txt",
            ):
                INGESTION.run(
                    root / "missing-input",
                    root / "output",
                    submission,
                    track="small",
                )


if __name__ == "__main__":
    unittest.main()
