"""Tests for participant-facing track import, scoring, and packaging tools."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrackToolingTests(unittest.TestCase):
    """Keep local participant tooling aligned with the live track contract."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.importer = load_module(
            "track_tooling_importer",
            ROOT / "scripts" / "import_hf_dataset.py",
        )
        cls.scoring = load_module(
            "track_tooling_scoring",
            ROOT / "components" / "scoring_program" / "scoring.py",
        )
        cls.builder = load_module(
            "track_tooling_builder",
            ROOT / "scripts" / "build.py",
        )

    def test_importer_writes_plain_problem_strings_and_deduplicates_variants(self) -> None:
        row = {
            "model_id": "example/model",
            "dataset_id": "dataset",
            "problem_id": "problem",
            "original_problem": "What is 2 + 2?",
            "permutation_type": ["rename"],
            "model_is_robust": True,
        }
        variant = {**row, "permutation_type": ["rephrase"]}

        cases, labels, summary = self.importer.convert_rows([row, variant])

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["problem"], "What is 2 + 2?")
        self.assertEqual(cases[0]["id"], labels[0]["id"])
        self.assertEqual(summary["duplicate_rows_removed"], 1)

    def test_small_scores_use_shared_leaderboard_keys(self) -> None:
        scores = self.scoring.compute_scores(
            {"case": True},
            {"case": {"id": "case", "is_robust": True, "valid": True}},
            set(),
            0,
            track="small",
        )

        self.assertEqual(
            {"accuracy", "coverage", "invalid_predictions"},
            set(scores),
        )

    def test_builder_creates_main_and_small_solution_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            original_dist = self.builder.DIST
            self.builder.DIST = Path(temporary)
            try:
                self.builder.build_solutions()
                self.builder.build_solutions(small=True)
            finally:
                self.builder.DIST = original_dist

            main_archive = Path(temporary) / "solution-always-true.zip"
            small_archive = Path(temporary) / "solution-always-true-small.zip"
            with ZipFile(main_archive) as bundle:
                self.assertNotIn("small.txt", bundle.namelist())
            with ZipFile(small_archive) as bundle:
                self.assertIn("small.txt", bundle.namelist())
                self.assertEqual(bundle.read("small.txt"), b"")


if __name__ == "__main__":
    unittest.main()
