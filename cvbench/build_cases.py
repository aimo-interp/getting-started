#!/usr/bin/env python3
"""Build the small-model cross-validation set at ``data/cvbench/cases.jsonl``.

Sources, all restricted to the four small competition models:

* ``train-main-v2``: organizer rows with a label (``is_robust`` not null).
* ``labelgen``: rows produced by ``datagen/label.py`` from our own sampling.

The public sample datasets (``val-sample``, ``aimo-interp-challenge-sample-full``) name their
models with ``:low`` aliases such as ``qwen3-8b:low``, which do not identify a local checkpoint
(README: the aliases are never passed during the Competition phase), so they are not used.

Each row gets a ``group`` (hash of the problem text) so CV never splits one
problem across folds, even when several models share it.

``--oss`` builds ``data/cvbench/cases_oss.jsonl`` instead: the gpt-oss-120b rows of
``train-main-v2``. The reasoning effort is part of the model id (``openai/gpt-oss-120b:medium``), so each
effort is cached and probed as its own model.

    uv run cvbench/build_cases.py [--oss]
"""

from __future__ import annotations

import argparse
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
OSS = "openai/gpt-oss-120b"


def group_of(problem: str) -> str:
    return hashlib.sha256(problem.strip().encode()).hexdigest()[:12]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def oss_rows(train: pd.DataFrame) -> list[dict]:
    return [dict(model_id=f"{OSS}:{r.reasoning_effort}", reasoning_effort=r.reasoning_effort, problem=r.problem,
                 is_robust=bool(r.is_robust), max_drop=float(r.max_drop), source="train-main-v2")
            for r in train.itertuples() if r.model_id == OSS and r.is_robust is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--oss", action="store_true", help="Build the gpt-oss-120b set (cases_oss.jsonl).")
    args = parser.parse_args()
    out = OUT.with_name("cases_oss.jsonl") if args.oss else OUT
    rows = []
    train = pd.read_parquet(glob.glob(str(HUB / "datasets--aimo-interp--train-main-v2/snapshots/*/data/*.parquet"))[0])
    if args.oss:
        rows = oss_rows(train)
    for r in [] if args.oss else train.itertuples():
        if r.model_id in SMALL_MODELS and r.is_robust is not None:
            rows.append(dict(model_id=r.model_id, reasoning_effort=r.reasoning_effort, problem=r.problem,
                             is_robust=bool(r.is_robust), max_drop=float(r.max_drop), source="train-main-v2"))

    labelgen = ROOT / "data/labelgen/labels.jsonl"
    if labelgen.is_file() and not args.oss:
        for r in read_jsonl(labelgen):
            if r["model_id"] in SMALL_MODELS and r["is_robust"] is not None:
                rows.append(dict(model_id=r["model_id"], reasoning_effort=r["reasoning_effort"], problem=r["problem"],
                                 is_robust=bool(r["is_robust"]), max_drop=r["max_drop"], source="labelgen"))

    # One row per (model, problem); the organizer label wins over our own.
    priority = {"train-main-v2": 0, "labelgen": 1}
    rows.sort(key=lambda r: priority[r["source"]])
    seen, unique = set(), []
    for r in rows:
        key = (r["model_id"], group_of(r["problem"]))
        if key not in seen:
            seen.add(key)
            r["group"] = key[1]
            r["case_id"] = f"{r['model_id'].split('/')[-1]}:{key[1]}"
            unique.append(r)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in unique:
            f.write(json.dumps(r) + "\n")
    counts = Counter((r["model_id"], r["source"], r["is_robust"]) for r in unique)
    for key in sorted(counts, key=str):
        print(key, counts[key])
    print(f"{len(unique)} cases, {len({r['group'] for r in unique})} problems -> {out}")


if __name__ == "__main__":
    main()
