#!/usr/bin/env python3
"""Build deterministic, independently uploadable Codabench artifacts."""

import argparse
import base64
import json
import re
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
FIXED_ZIP_TIME = (2020, 1, 1, 0, 0, 0)
SOLUTION_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def archive(output: Path, entries: dict[str, bytes]) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as bundle:
        for name in sorted(entries):
            info = ZipInfo(name, date_time=FIXED_ZIP_TIME)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, entries[name])
    return output


def directory_entries(directory: Path) -> dict[str, bytes]:
    if not directory.is_dir():
        raise SystemExit(f"directory does not exist: {directory}")
    entries = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if (
            path.is_file()
            and "__pycache__" not in relative.parts
            and not path.name.endswith((".pyc", ".pyo"))
            and not path.name.startswith(".")
        ):
            entries[relative.as_posix()] = path.read_bytes()
    if not entries:
        raise SystemExit(f"directory is empty: {directory}")
    return entries


def solutions() -> list[tuple[str, Path]]:
    root = ROOT / "solutions"
    if not root.is_dir():
        raise SystemExit(f"solutions directory does not exist: {root}")

    found = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        if not SOLUTION_NAME.fullmatch(directory.name):
            raise SystemExit(
                f"invalid solution directory name {directory.name!r}; "
                "use lowercase letters, digits, and hyphens"
            )
        if not (directory / "solution.py").is_file():
            raise SystemExit(f"solution is missing solution.py: {directory}")
        found.append((directory.name, directory))

    if not found:
        raise SystemExit(f"no solution directories found in {root}")
    return found


def solution_archive_name(name: str, small: bool = False) -> str:
    suffix = "-small" if small else ""
    return f"solution-{name}{suffix}.zip"


def build_solutions(small: bool = False) -> None:
    for name, directory in solutions():
        entries = directory_entries(directory)
        marker_names = [
            path for path in entries if "/" not in path and path.casefold() == "small.txt"
        ]
        if small:
            for marker_name in marker_names:
                del entries[marker_name]
            entries["small.txt"] = b""
        elif marker_names:
            raise SystemExit(
                f"main-track solution {name!r} contains small.txt; "
                "remove it or build with --small"
            )
        archive(DIST / solution_archive_name(name, small=small), entries)


def build_components(input_dir: Path, reference_dir: Path) -> None:
    archive(
        DIST / "ingestion-program.zip",
        directory_entries(ROOT / "components" / "ingestion_program"),
    )
    archive(
        DIST / "scoring-program.zip",
        directory_entries(ROOT / "components" / "scoring_program"),
    )
    build_solutions()
    archive(DIST / "input-data.zip", directory_entries(input_dir))
    archive(DIST / "reference-data.zip", directory_entries(reference_dir))


def read_resource_config(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise SystemExit(
            f"resource config not found: {path}; copy config/resources.example.json first"
        )
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise SystemExit("resource config must be a JSON object")
    return values


def require_keys(values: dict[str, str], keys: set) -> None:
    missing = sorted(key for key in keys if not isinstance(values.get(key), str) or not values[key])
    if missing:
        raise SystemExit(f"resource config is missing: {', '.join(missing)}")


def build_task(mode: str, resource_config: Path) -> None:
    if mode == "embedded":
        required_files = [
            "ingestion-program.zip",
            "scoring-program.zip",
            "input-data.zip",
            "reference-data.zip",
        ]
        entries = {
            "task.yaml": (ROOT / "task" / "task.embedded.yaml").read_bytes(),
        }
        for filename in required_files:
            path = DIST / filename
            if not path.is_file():
                raise SystemExit(f"missing {path}; build components first")
            entries[filename] = path.read_bytes()
        archive(DIST / "task-embedded.zip", entries)
        return

    values = read_resource_config(resource_config)
    keys = {"ingestion_program", "scoring_program", "input_data", "reference_data"}
    require_keys(values, keys)
    task_yaml = (ROOT / "task" / "task.linked.template.yaml").read_text(encoding="utf-8")
    replacements = {
        "{{INGESTION_PROGRAM}}": values["ingestion_program"],
        "{{SCORING_PROGRAM}}": values["scoring_program"],
        "{{INPUT_DATA}}": values["input_data"],
        "{{REFERENCE_DATA}}": values["reference_data"],
    }
    for placeholder, value in replacements.items():
        task_yaml = task_yaml.replace(placeholder, value)
    archive(DIST / "task-linked.zip", {"task.yaml": task_yaml.encode()})


def competition_task_definition(mode: str, resource_config: Path) -> str:
    if mode == "embedded":
        return "\n".join(
            [
                "  - index: 0",
                "    name: AIMO Robustness Validation v1",
                "    description: Public validation task; replace data before launch.",
                "    ingestion_program: ingestion-program.zip",
                "    scoring_program: scoring-program.zip",
                "    input_data: input-data.zip",
                "    reference_data: reference-data.zip",
            ]
        )
    values = read_resource_config(resource_config)
    require_keys(values, {"task"})
    return "\n".join(["  - index: 0", f"    key: {values['task']}"])


def competition_solution_definition() -> str:
    lines = []
    for index, (name, _) in enumerate(solutions()):
        lines.extend(
            [
                f"  - index: {index}",
                f"    path: {solution_archive_name(name)}",
                "    tasks:",
                "      - 0",
            ]
        )
    return "\n".join(lines)


def build_competition(mode: str, resource_config: Path) -> None:
    template = (ROOT / "competition" / "competition.template.yaml").read_text(encoding="utf-8")
    competition_yaml = template.replace(
        "{{TASK_DEFINITION}}", competition_task_definition(mode, resource_config)
    )
    competition_yaml = competition_yaml.replace(
        "{{SOLUTION_DEFINITION}}", competition_solution_definition()
    )
    entries = {
        "competition.yaml": competition_yaml.encode(),
        "pages/overview.md": (ROOT / "competition" / "pages" / "overview.md").read_bytes(),
        "pages/evaluation.md": (ROOT / "competition" / "pages" / "evaluation.md").read_bytes(),
        "pages/terms.md": (ROOT / "competition" / "pages" / "terms.md").read_bytes(),
        "assets/logo.png": base64.b64decode(
            (ROOT / "competition" / "assets" / "logo.png.base64")
            .read_text(encoding="ascii")
            .strip()
        ),
    }

    for name, _ in solutions():
        filename = solution_archive_name(name)
        solution = DIST / filename
        if not solution.is_file():
            raise SystemExit(f"missing {solution}; build components first")
        entries[filename] = solution.read_bytes()

    if mode == "embedded":
        for filename in [
            "ingestion-program.zip",
            "scoring-program.zip",
            "input-data.zip",
            "reference-data.zip",
        ]:
            path = DIST / filename
            if not path.is_file():
                raise SystemExit(f"missing {path}; build components first")
            entries[filename] = path.read_bytes()

    archive(DIST / f"competition-{mode}.zip", entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target",
        choices=["solutions", "components", "task", "competition", "all"],
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="with target=solutions, build small-track ZIPs containing a blank small.txt",
    )
    parser.add_argument("--mode", choices=["embedded", "linked"], default="embedded")
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
    parser.add_argument(
        "--resource-config",
        type=Path,
        default=ROOT / "config" / "resources.local.json",
    )
    parser.add_argument(
        "--include-linked",
        action="store_true",
        help="with target=all, also build UUID-linked task and competition",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.small and args.target != "solutions":
        raise SystemExit("--small is only supported with target=solutions")
    if args.target == "solutions":
        build_solutions(small=args.small)
    elif args.target == "components":
        build_components(args.input_dir, args.reference_dir)
    elif args.target == "task":
        build_task(args.mode, args.resource_config)
    elif args.target == "competition":
        build_competition(args.mode, args.resource_config)
    else:
        if DIST.exists():
            shutil.rmtree(DIST)
        build_components(args.input_dir, args.reference_dir)
        build_task("embedded", args.resource_config)
        build_competition("embedded", args.resource_config)
        if args.include_linked:
            build_task("linked", args.resource_config)
            build_competition("linked", args.resource_config)


if __name__ == "__main__":
    main()
