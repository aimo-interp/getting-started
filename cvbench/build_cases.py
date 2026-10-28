#!/usr/bin/env python3
"""Build the small-model cross-validation set at ``data/cvbench/cases.jsonl``.

Sources, all restricted to the four small competition models:

* ``train-main-v2``: organizer rows with a label (``is_robust`` not null).
* ``val-sample``: the ``qwen3-8b:low`` rows, which are DeepSeek-R1-0528-Qwen3-8B.
* ``labelgen``: rows produced by ``datagen/label.py`` from our own sampling.

Each row gets a ``group`` (hash of the problem text) so CV never splits one
problem across folds, even when several models share it.

    uv run cvbench/build_cases.py
"""

from __future__ import annotations

import glob
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HUB = Path("/data/gpfs/projects/punim0478/bansaab/hub")
OUT = ROOT / "data" / "cvbench" / "cases.jsonl"
SMALL_MODELS = {
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}
ALIASES = {"qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}


def group_of(problem: str) -> str:
    return hashlib.sha256(problem.strip().encode()).hexdigest()[:12]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    rows = []
    train = pd.read_parquet(glob.glob(str(HUB / "datasets--aimo-interp--train-main-v2/snapshots/*/data/*.parquet"))[0])
    for r in train.itertuples():
        if r.model_id in SMALL_MODELS and r.is_robust is not None:
            rows.append(dict(model_id=r.model_id, reasoning_effort=r.reasoning_effort, problem=r.problem,
                             is_robust=bool(r.is_robust), max_drop=float(r.max_drop), source="train-main-v2"))

    cases = {c["id"]: c for c in read_jsonl(ROOT / "data/val-sample/input/cases.jsonl")}
    for label in read_jsonl(ROOT / "data/val-sample/reference/labels.jsonl"):
        case = cases[label["id"]]
        if case["model_id"] in ALIASES:
            rows.append(dict(model_id=ALIASES[case["model_id"]], reasoning_effort="default", problem=case["problem"],
                             is_robust=bool(label["is_robust"]), max_drop=None, source="val-sample"))

    labelgen = ROOT / "data/labelgen/labels.jsonl"
    if labelgen.is_file():
        for r in read_jsonl(labelgen):
            if r["model_id"] in SMALL_MODELS and r["is_robust"] is not None:
                rows.append(dict(model_id=r["model_id"], reasoning_effort=r["reasoning_effort"], problem=r["problem"],
                                 is_robust=bool(r["is_robust"]), max_drop=r["max_drop"], source="labelgen"))

    # One row per (model, problem); the organizer label wins over our own.
    priority = {"train-main-v2": 0, "val-sample": 1, "labelgen": 2}
    rows.sort(key=lambda r: priority[r["source"]])
    seen, unique = set(), []
    for r in rows:
        key = (r["model_id"], group_of(r["problem"]))
        if key not in seen:
            seen.add(key)
            r["group"] = key[1]
            r["case_id"] = f"{r['model_id'].split('/')[-1]}:{key[1]}"
            unique.append(r)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in unique:
            f.write(json.dumps(r) + "\n")
    counts = Counter((r["model_id"], r["source"], r["is_robust"]) for r in unique)
    for key in sorted(counts, key=str):
        print(key, counts[key])
    print(f"{len(unique)} cases, {len({r['group'] for r in unique})} problems -> {OUT}")


if __name__ == "__main__":
    main()
