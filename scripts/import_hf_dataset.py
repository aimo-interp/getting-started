#!/usr/bin/env python3
"""Import a pinned Hugging Face dataset into Codabench input/reference data."""

import argparse
import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "aimo-interp/val-sample"
DEFAULT_REVISION = "1ae454ec1fad9727084eda8f9f3c9ae2239b21de"
DEFAULT_CONFIG = "default"
DEFAULT_SPLIT = "validation"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "val-sample"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100


class DatasetImportError(RuntimeError):
    pass


def hf_token() -> str | None:
    """Return a Hugging Face token from the environment or the local HF store.

    The dataset is private: requests must authenticate with a token that has
    access. Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN), or log in once with
    `uv run hf auth login`.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.is_file():
        return token_path.read_text(encoding="utf-8").strip()
    return None


def request_headers() -> dict[str, str]:
    headers = {"User-Agent": "aimo-codabench-dataset-importer/0.1"}
    token = hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_rows_from_files(dataset: str, revision: str, split: str) -> list[dict[str, Any]]:
    """Fetch rows by downloading the split's parquet files at the revision.

    The datasets-server rows API only serves public and gated datasets. For the
    private dataset this downloads the underlying data files directly, which
    works with any token that has read access.
    """
    try:
        import pandas
    except ImportError as exc:  # pragma: no cover - environment guard
        raise DatasetImportError(
            "reading a private dataset requires pandas and pyarrow: run `uv sync`"
        ) from exc

    info_url = f"https://huggingface.co/api/datasets/{dataset}/revision/{revision}"
    request = urllib.request.Request(info_url, headers=request_headers())
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            info = json.load(response)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise DatasetImportError(
            f"failed to list dataset files (check HF_TOKEN access): {exc}"
        ) from exc

    filenames = [
        sibling["rfilename"]
        for sibling in info.get("siblings", [])
        if isinstance(sibling, dict)
        and isinstance(sibling.get("rfilename"), str)
        and sibling["rfilename"].startswith(f"data/{split}-")
        and sibling["rfilename"].endswith(".parquet")
    ]
    if not filenames:
        raise DatasetImportError(f"no parquet files found for split {split!r} at {revision}")

    rows: list[dict[str, Any]] = []
    for filename in sorted(filenames):
        file_url = f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{filename}"
        request = urllib.request.Request(file_url, headers=request_headers())
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise DatasetImportError(f"failed to download {filename}: {exc}") from exc
        with tempfile.NamedTemporaryFile(suffix=".parquet") as handle:
            handle.write(payload)
            handle.flush()
            frame = pandas.read_parquet(handle.name)
        # JSON round-trip converts NumPy scalars and arrays to plain Python
        # types so the downstream validation sees the same shapes as the API.
        rows.extend(json.loads(frame.to_json(orient="records")))
    return rows


def fetch_page(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    offset: int,
    length: int,
) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "revision": revision,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    request = urllib.request.Request(f"{ROWS_ENDPOINT}?{query}", headers=request_headers())
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise DatasetImportError(
                "the dataset requires authentication (HTTP 401): set HF_TOKEN to a "
                "Hugging Face token that has access to it"
            ) from exc
        raise DatasetImportError(f"failed to fetch Hugging Face rows: {exc}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise DatasetImportError(f"failed to fetch Hugging Face rows: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetImportError("Hugging Face rows response must be an object")
    return payload


def extract_page_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries = payload.get("rows")
    if not isinstance(entries, list):
        raise DatasetImportError("Hugging Face rows response has no rows array")
    rows = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("row"), dict):
            raise DatasetImportError(f"Hugging Face row {index} is malformed")
        rows.append(entry["row"])
    return rows


def fetch_all_rows(
    dataset: str,
    revision: str,
    config: str,
    split: str,
) -> list[dict[str, Any]]:
    try:
        return fetch_rows_from_api(dataset, revision, config, split)
    except DatasetImportError as exc:
        print(f"rows API unavailable ({exc}); downloading dataset files instead")
        return fetch_rows_from_files(dataset, revision, split)


def fetch_rows_from_api(
    dataset: str,
    revision: str,
    config: str,
    split: str,
) -> list[dict[str, Any]]:
    rows = []
    total = None
    while total is None or len(rows) < total:
        payload = fetch_page(
            dataset=dataset,
            revision=revision,
            config=config,
            split=split,
            offset=len(rows),
            length=PAGE_SIZE,
        )
        if payload.get("partial") is True:
            raise DatasetImportError("Hugging Face returned a partial dataset response")
        page_rows = extract_page_rows(payload)
        reported_total = payload.get("num_rows_total")
        if not isinstance(reported_total, int) or reported_total < 0:
            raise DatasetImportError("Hugging Face response has invalid num_rows_total")
        if total is not None and reported_total != total:
            raise DatasetImportError("Hugging Face row count changed during import")
        total = reported_total
        if not page_rows and len(rows) < total:
            raise DatasetImportError("Hugging Face pagination ended before all rows arrived")
        rows.extend(page_rows)
    if len(rows) != total:
        raise DatasetImportError(f"Hugging Face returned {len(rows)} rows but reported {total}")
    return rows


def load_source_json(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetImportError(f"could not read source JSON: {exc}") from exc
    if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
        return payload
    if isinstance(payload, dict):
        return extract_page_rows(payload)
    raise DatasetImportError("source JSON must be a row list or Hugging Face rows response")


def require_string(row: dict[str, Any], field: str, row_number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise DatasetImportError(f"source row {row_number} has invalid {field}")
    return value


def require_string_list(row: dict[str, Any], field: str, row_number: int) -> list[str]:
    value = row.get(field)
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [value]
    else:
        raise DatasetImportError(f"source row {row_number} has invalid {field}")
    if (
        not isinstance(parsed, list)
        or not parsed
        or any(not isinstance(item, str) or not item for item in parsed)
    ):
        raise DatasetImportError(f"source row {row_number} has invalid {field}")
    return parsed


def make_case_id(
    model_id: str,
    dataset_id: str,
    problem_id: str,
    permutation_type: tuple[str, ...],
) -> str:
    identity = json.dumps(
        [model_id, dataset_id, problem_id, list(permutation_type)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return f"case-{hashlib.sha256(identity).hexdigest()[:24]}"


def convert_rows(
    source_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    unique_rows: dict[tuple[str, str, str, tuple[str, ...]], dict[str, Any]] = {}
    for row_number, row in enumerate(source_rows, start=1):
        model_id = require_string(row, "model_id", row_number)
        dataset_id = require_string(row, "dataset_id", row_number)
        problem_id = require_string(row, "problem_id", row_number)
        original_problem = require_string(row, "original_problem", row_number)
        permutation_type = tuple(require_string_list(row, "permutation_type", row_number))
        label = row.get("model_is_robust")
        if type(label) is not bool:
            raise DatasetImportError(f"source row {row_number} has invalid model_is_robust")

        key = (model_id, dataset_id, problem_id, permutation_type)
        previous = unique_rows.get(key)
        if previous is not None:
            if previous["original_problem"] != original_problem:
                raise DatasetImportError(
                    f"duplicate key {key!r} has conflicting problem statements"
                )
            if previous["model_is_robust"] != label:
                raise DatasetImportError(f"duplicate key {key!r} has conflicting robustness labels")
            continue
        unique_rows[key] = {
            "original_problem": original_problem,
            "model_is_robust": label,
        }

    if not unique_rows:
        raise DatasetImportError("source dataset is empty")

    cases = []
    labels = []
    seen_case_ids = set()
    for model_id, dataset_id, problem_id, permutation_type in sorted(unique_rows):
        row = unique_rows[(model_id, dataset_id, problem_id, permutation_type)]
        case_id = make_case_id(model_id, dataset_id, problem_id, permutation_type)
        if case_id in seen_case_ids:
            raise DatasetImportError(f"generated case ID collision: {case_id}")
        seen_case_ids.add(case_id)
        cases.append(
            {
                "id": case_id,
                "model_id": model_id,
                "problem": row["original_problem"],
                "permutation_type": list(permutation_type),
            }
        )
        labels.append(
            {
                "id": case_id,
                "is_robust": row["model_is_robust"],
            }
        )

    true_labels = sum(label["is_robust"] for label in labels)
    summary = {
        "source_rows": len(source_rows),
        "cases": len(cases),
        "duplicate_rows_removed": len(source_rows) - len(cases),
        "label_counts": {
            "false": len(labels) - true_labels,
            "true": true_labels,
        },
    }
    return cases, labels, summary


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    atomic_write(path, content)


def write_dataset(
    output_dir: Path,
    cases: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    write_jsonl(output_dir / "input" / "cases.jsonl", cases)
    write_jsonl(output_dir / "reference" / "labels.jsonl", labels)
    atomic_write(
        output_dir / "metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source-json",
        type=Path,
        help="read a saved rows response instead of accessing Hugging Face",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_json:
        source_rows = load_source_json(args.source_json)
        source = str(args.source_json)
    else:
        source_rows = fetch_all_rows(
            dataset=args.dataset,
            revision=args.revision,
            config=args.config,
            split=args.split,
        )
        source = f"https://huggingface.co/datasets/{args.dataset}/tree/{args.revision}"

    cases, labels, summary = convert_rows(source_rows)
    metadata = {
        "schema_version": 1,
        "source": source,
        "dataset": args.dataset,
        "revision": args.revision,
        "config": args.config,
        "split": args.split,
        **summary,
    }
    write_dataset(args.output_dir, cases, labels, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
