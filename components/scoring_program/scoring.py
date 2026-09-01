#!/usr/bin/env python3
"""Score predictions against Codabench reference data."""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class ScoringError(RuntimeError):
    pass


def require_track(track: Any) -> str:
    if not isinstance(track, str) or track not in {"main", "small"}:
        raise ScoringError("invalid scoring track configuration")
    return track


def load_track(path: Path | None = None) -> str:
    config_path = path or Path(__file__).with_name("track.json")
    if not config_path.is_file():
        return "main"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoringError("invalid scoring track configuration") from exc
    track = config.get("track") if isinstance(config, dict) else None
    return require_track(track)


def read_jsonl_strict(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScoringError(f"{path.name}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ScoringError(f"{path.name}:{line_number}: expected a JSON object")
            yield line_number, value


def load_labels(path: Path) -> dict[str, bool]:
    labels: dict[str, bool] = {}
    for line_number, row in read_jsonl_strict(path):
        case_id = row.get("id")
        label = row.get("is_robust")
        if not isinstance(case_id, str) or not case_id:
            raise ScoringError(f"{path.name}:{line_number}: invalid id")
        if case_id in labels:
            raise ScoringError(f"{path.name}:{line_number}: duplicate id {case_id!r}")
        if type(label) is not bool:
            raise ScoringError(f"{path.name}:{line_number}: label must be boolean")
        labels[case_id] = label
    if not labels:
        raise ScoringError("reference dataset is empty")
    return labels


def load_predictions(path: Path) -> tuple[dict[str, dict[str, Any]], set, int]:
    predictions: dict[str, dict[str, Any]] = {}
    duplicate_ids = set()
    malformed = 0
    if not path.is_file():
        return predictions, duplicate_ids, malformed

    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(row, dict):
                malformed += 1
                continue
            case_id = row.get("id")
            if not isinstance(case_id, str) or not case_id:
                malformed += 1
                continue
            if case_id in predictions:
                duplicate_ids.add(case_id)
            else:
                predictions[case_id] = row
    return predictions, duplicate_ids, malformed


def compute_scores(
    labels: dict[str, bool],
    predictions: dict[str, dict[str, Any]],
    duplicate_ids: set,
    malformed: int,
    track: str = "main",
) -> dict[str, Any]:
    require_track(track)
    correct = 0
    covered = 0

    for case_id, label in labels.items():
        row = predictions.get(case_id)
        if row is None or case_id in duplicate_ids:
            continue
        if type(row.get("valid")) is not bool or not row["valid"]:
            continue
        prediction = row.get("is_robust")
        if type(prediction) is not bool:
            continue
        covered += 1
        if prediction == label:
            correct += 1

    extra_ids = set(predictions) - set(labels)
    structural_errors = malformed + len(duplicate_ids) + len(extra_ids)
    total = len(labels)
    accuracy = correct / total
    coverage = covered / total
    invalid_predictions = (total - covered) + structural_errors

    suffix = "" if track == "main" else "_small"
    return {
        f"accuracy{suffix}": accuracy,
        f"coverage{suffix}": coverage,
        f"invalid_predictions{suffix}": invalid_predictions,
    }


def run(input_dir: Path, output_dir: Path, track: str | None = None) -> None:
    reference_dir = input_dir / "ref" if (input_dir / "ref").is_dir() else input_dir
    results_dir = input_dir / "res" if (input_dir / "res").is_dir() else input_dir
    labels = load_labels(reference_dir / "labels.jsonl")
    predictions, duplicate_ids, malformed = load_predictions(results_dir / "predictions.jsonl")
    scores = compute_scores(
        labels,
        predictions,
        duplicate_ids,
        malformed,
        track=require_track(track) if track else load_track(),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scores.json").write_text(
        json.dumps(scores, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--track", choices=["main", "small"])
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.input_dir, arguments.output_dir, track=arguments.track)
