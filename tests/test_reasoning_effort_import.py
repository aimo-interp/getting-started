import importlib.util
import unittest
from pathlib import Path


def source_row(**overrides):
    row = {
        "model_id": "model-a:low",
        "dataset_id": "dataset-a",
        "problem_id": "problem-a",
        "original_problem": "What is 2 + 2?",
        "permutation_type": "rename",
        "model_is_robust": True,
        "base_accuracy": 1.0,
    }
    row.update(overrides)
    return row



class ReasoningEffortImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts/import_hf_dataset.py"
        spec = importlib.util.spec_from_file_location("effort_importer", path)
        cls.importer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.importer)

    def test_reasoning_effort_is_preserved_and_default_ids_are_stable(self):
        original_cases, _, _ = self.importer.convert_rows([source_row()])
        rows = [source_row(), source_row(reasoning_effort="default"),
                source_row(reasoning_effort="medium", model_is_robust=False)]
        cases, labels, summary = self.importer.convert_rows(rows)
        self.assertEqual(cases[0], original_cases[0])
        self.assertEqual(cases[1]["reasoning_effort"], "medium")
        self.assertNotEqual(cases[0]["id"], cases[1]["id"])
        self.assertEqual([row["is_robust"] for row in labels], [True, False])
        self.assertEqual(summary["duplicate_rows_removed"], 1)
        self.assertEqual(self.importer.convert_rows(list(reversed(rows))), (cases, labels, summary))

    def test_invalid_reasoning_effort_is_rejected(self):
        for effort in [None, "", "  ", 1, False, []]:
            with self.subTest(effort=effort):
                with self.assertRaisesRegex(self.importer.DatasetImportError, "reasoning_effort"):
                    self.importer.convert_rows([source_row(reasoning_effort=effort)])

