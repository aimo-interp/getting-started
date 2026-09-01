#!/usr/bin/env python3
"""Run the ingestion and scoring pipeline locally."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument(
        "--small",
        action="store_true",
        help="run the submission against the small model track",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "data" / "val-sample" / "input",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=ROOT / "data" / "val-sample" / "reference",
    )
    return parser.parse_args()


def extract_submission(archive: Path, destination: Path) -> Path:
    try:
        with ZipFile(archive) as bundle:
            root = destination.resolve()
            for member in bundle.infolist():
                target = (destination / member.filename).resolve()
                if target != root and root not in target.parents:
                    raise SystemExit(f"submission ZIP contains an unsafe path: {member.filename}")
            bundle.extractall(destination)
    except BadZipFile as exc:
        raise SystemExit(f"invalid submission ZIP: {archive}") from exc
    return destination


def main() -> None:
    args = parse_args()
    submission_source = args.submission.resolve()
    track = "small" if args.small else "main"

    with tempfile.TemporaryDirectory(prefix="aimo-codabench-") as temporary:
        root = Path(temporary)
        if submission_source.is_dir():
            submission = submission_source
        elif submission_source.is_file() and submission_source.suffix.casefold() == ".zip":
            submission = extract_submission(submission_source, root / "submission")
        else:
            raise SystemExit("submission must be a directory or ZIP archive")
        if not (submission / "solution.py").is_file():
            raise SystemExit("submission must contain solution.py at its root")

        ingestion_output = root / "input" / "res"
        scoring_input = root / "input"
        scoring_output = root / "output"
        reference_link = scoring_input / "ref"
        reference_link.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(args.reference_dir.resolve(), reference_link)

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "components" / "ingestion_program" / "ingestion.py"),
                str(args.input_dir.resolve()),
                str(ingestion_output),
                str(submission),
                "--track",
                track,
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "components" / "scoring_program" / "scoring.py"),
                str(scoring_input),
                str(scoring_output),
                "--track",
                track,
            ],
            check=True,
        )
        scores = json.loads((scoring_output / "scores.json").read_text(encoding="utf-8"))
        print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
