#!/usr/bin/env python3
"""Run a participant's are_robust function over Codabench input data."""

import argparse
import importlib.util
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any


class IngestionError(RuntimeError):
    pass


def require_track(track: Any) -> str:
    if not isinstance(track, str) or track not in {"main", "small"}:
        raise IngestionError("invalid ingestion track configuration")
    return track


def load_track(path: Path | None = None) -> str:
    config_path = path or Path(__file__).with_name("track.json")
    if not config_path.is_file():
        return "main"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IngestionError("invalid ingestion track configuration") from exc
    track = config.get("track") if isinstance(config, dict) else None
    return require_track(track)


def validate_submission_track(submission_dir: Path, track: str) -> None:
    if not submission_dir.is_dir():
        raise IngestionError(f"submission directory does not exist: {submission_dir}")
    has_small_marker = any(
        path.is_file() and path.name.casefold() == "small.txt"
        for path in submission_dir.iterdir()
    )
    if track == "main" and has_small_marker:
        raise IngestionError("small.txt is not allowed for a main-track submission")
    if track == "small" and not has_small_marker:
        raise IngestionError("small-model-track submissions must contain small.txt at their root")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IngestionError(f"{path.name}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise IngestionError(f"{path.name}:{line_number}: expected a JSON object")
            yield value


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_ids = set()
    for line_number, case in enumerate(read_jsonl(path), start=1):
        case_id = case.get("id")
        model_id = case.get("model_id")
        problem = case.get("problem")
        if not isinstance(case_id, str) or not case_id:
            raise IngestionError(f"{path.name}:{line_number}: invalid id")
        if case_id in seen_ids:
            raise IngestionError(f"{path.name}:{line_number}: duplicate id {case_id!r}")
        if not isinstance(model_id, str) or not model_id:
            raise IngestionError(f"{path.name}:{line_number}: invalid model_id")
        if not isinstance(problem, str) or not problem:
            raise IngestionError(f"{path.name}:{line_number}: invalid problem")
        seen_ids.add(case_id)
        cases.append(case)
    if not cases:
        raise IngestionError("input dataset is empty")
    return cases


def load_solution(submission_dir: Path) -> ModuleType:
    solution_path = submission_dir / "solution.py"
    if not solution_path.is_file():
        raise IngestionError("submission must contain solution.py at its root")
    spec = importlib.util.spec_from_file_location("participant_solution", solution_path)
    if spec is None or spec.loader is None:
        raise IngestionError("could not load solution.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if sys.modules.get(spec.name) is module:
            del sys.modules[spec.name]
        raise
    if not callable(getattr(module, "are_robust", None)):
        raise IngestionError("solution.py must define callable are_robust(model_id, problems)")
    return module


def run(
    input_dir: Path,
    output_dir: Path,
    submission_dir: Path,
    track: str | None = None,
) -> None:
    submission_import_path = str(submission_dir.resolve())
    sys.path.insert(0, submission_import_path)
    try:
        _run(input_dir, output_dir, submission_dir, require_track(track) if track else load_track())
    finally:
        # Remove the exact entry inserted above, even if participant code changed
        # sys.path or added the same path independently.
        for index, entry in enumerate(sys.path):
            if entry is submission_import_path:
                del sys.path[index]
                break


def _run(input_dir: Path, output_dir: Path, submission_dir: Path, track: str) -> None:
    validate_submission_track(submission_dir, track)
    cases = load_cases(input_dir / "cases.jsonl")
    solution = load_solution(submission_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batches: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, case in enumerate(cases):
        batches.setdefault(case["model_id"], []).append((index, case))

    predictions = [
        {"id": case["id"], "is_robust": False, "valid": False} for case in cases
    ]
    failures = 0
    for model_id, batch in batches.items():
        batch_predictions = [False] * len(batch)
        batch_valid = [False] * len(batch)
        try:
            problems = [case["problem"] for _, case in batch]
            results = solution.are_robust(model_id, problems)
            if (
                type(results) is list
                and len(results) == len(problems)
                and all(type(result) is bool for result in results)
            ):
                batch_predictions = results
                batch_valid = [True] * len(batch)
        except Exception:
            # Participant exception details are intentionally not copied to results.
            pass

        failures += batch_valid.count(False)
        for (index, case), prediction, valid in zip(batch, batch_predictions, batch_valid):
            predictions[index] = {
                "id": case["id"],
                "is_robust": prediction,
                "valid": valid,
            }

    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, sort_keys=True, separators=(",", ":")))
            stream.write("\n")

    summary = {"cases": len(cases), "invalid_predictions": failures}
    (output_dir / "ingestion_summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--track", choices=["main", "small"])
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.input_dir,
        arguments.output_dir,
        arguments.submission_dir,
        track=arguments.track,
    )
