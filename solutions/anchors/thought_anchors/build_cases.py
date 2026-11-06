#!/usr/bin/env python3
"""Write cases.jsonl: every labelled gpt-oss-120b row of train-main-v2 (problem x effort).

train-main-v2 has no answers. The gold answers below were checked two ways:
* AIME 2025 problems: the MathArena/aime_2025 answer (texts identical, or differing
  only in the example grid shown).
* everything else (2026 AIME II problems, AIMO 2 reference problems and reworded
  variants of both, whose answers differ from the originals): solved by exact
  computation or brute force, and matching the majority of 8 high-effort gpt-oss-120b
  samples on Crusoe (7-8/8 agree on every problem).

    python solutions/anchors/thought_anchors/build_cases.py
"""

from __future__ import annotations

import glob
import hashlib
import json

import pandas as pd

from resample import CASES

HUB = "/data/gpfs/projects/punim0478/bansaab/hub/"
MODEL = "openai/gpt-oss-120b"
# group (sha256(problem)[:12]) -> (gold, how it was obtained)
GOLD = {
    "26a263ad8113": (70, "aime_2025 I"),
    "f862f0368b2a": (504, "aime_2025 I"),
    "0412b80ad1a4": (77, "aime_2025 I"),
    "9dd85c12874d": (81, "aime_2025 I (example grid differs)"),
    "7e76250549ad": (81, "aime_2025 I"),
    "f3d7b78d243a": (50, "aime_2026 II #6; computed"),
    "51697b0daa5c": (223, "aime_2026 II #12; computed"),
    "f7b4a98af9dd": (850, "computed (variant of aime_2026 II #10: sum of BC, not max)"),
    "757506c4e1d1": (16, "computed (variant of aime_2026 II #14: target -8, not 0)"),
    "a59a836b9807": (50, "computed (variant of AIMO2 ref: AB=20, R=26)"),
    "9dca5d882a4c": (902, "computed (AIMO2 ref)"),
    "d8be640af827": (143, "computed (AIMO2 ref)"),
    "2ca839cfbfc6": (50, "computed (AIMO2 ref)"),
    "d06c3e7f2d19": (201, "computed: N = 2^101 5^100 (AIMO2 ref)"),
    "a2e1c2f1ef6f": (32951, "computed (AIMO2 ref)"),
    "bde3afe054a4": (145040, "computed (variant of AIMO2 ref: j^445, M^143, mod 22^4)"),
    "57b6ba21f3f5": (2193, "computed: M = ceil(log2 10^100000) = 332193 (variant of AIMO2 ref: mod 10^4)"),
    "d9e9c4d93e19": (580, "computed (AIMO2 ref)"),
    "514ab5381d43": (42, "computed: triangle 6-7-8 (variant of AIMO2 ref: mod 7^2)"),
    "a09aac07cb64": (751, "computed: BE = 702/49"),
    "7443669f7e90": (669, "computed"),
}
# Cases already run under these names keep them, so their outputs are reused.
NAMED = {("7443669f7e90", "low"): "fragile_ninths", ("9dca5d882a4c", "low"): "robust_erased"}


def main() -> None:
    train = pd.read_parquet(glob.glob(HUB + "datasets--aimo-interp--train-main-v2/snapshots/*/data/*.parquet")[0])
    rows = train[(train.model_id == MODEL) & train.is_robust.notna()]
    cases = []
    for r in rows.itertuples():
        group = hashlib.sha256(r.problem.encode()).hexdigest()[:12]
        gold, how = GOLD[group]
        label = "robust" if r.is_robust else "fragile"
        cases.append({"case": NAMED.get((group, r.reasoning_effort), f"{label}_{group}_{r.reasoning_effort}"),
                      "model": MODEL, "effort": r.reasoning_effort, "problem": r.problem, "gold": gold,
                      "gold_source": how, "is_robust": bool(r.is_robust), "max_drop": float(r.max_drop),
                      "group": group, "source": "aimo-interp/train-main-v2@fa8af75"})
    cases.sort(key=lambda c: (c["is_robust"], c["group"], c["effort"]))
    with open(CASES, "w") as stream:
        stream.writelines(json.dumps(c) + "\n" for c in cases)
    print(f"wrote {len(cases)} cases to {CASES}")


if __name__ == "__main__":
    main()
