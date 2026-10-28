#!/usr/bin/env python3
"""Stage 0 (project env): build ``work/cases.json`` for the anchor analyses.

For every labelled case of ``--model`` in ``data/cvbench/cases.jsonl``:

* the gold answer, looked up by problem text in ``data/labelgen/problems.jsonl``
  and the AIME/AMC validation sets (None if absent; ``resample.py screen`` then
  falls back to the majority answer on X, which the label rule makes reliable);
* the perturbed prompts dX: the 4 code-made perturbations
  (``perturb_probe.perturbations``) at ``--seeds`` seeds, plus any LLM-written
  variants in ``datagen/written_variants.jsonl`` / ``data/labelgen/variants.jsonl``;
* the response-mean diff-in-means probe (``act_probe`` at layer frac 0.5,
  the 0.85 configuration) scored leave-one-problem-out, so each case gets an
  honest score and prediction; the full-data direction is saved for whitebox.py.

    uv run solutions/anchors/prepare.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "datagen"), str(ROOT / "solutions" / "perturb-probe")]

from common import extract_answer, normalize, parse_integer, read_jsonl  # noqa: E402
from perturb_probe.perturbations import make_perturbations  # noqa: E402

HUB = Path("/data/gpfs/projects/punim0478/bansaab/hub")
WORK = Path(os.environ.get("ANCHORS_WORK", ROOT / "data" / "anchors"))


def safe(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)


def gold_table() -> dict[str, int]:
    table = {}
    for row in read_jsonl(ROOT / "data/labelgen/problems.jsonl"):
        value = parse_integer(str(row.get("answer")))
        if value is not None:
            table[normalize(row["problem"])] = value
    for name in ("aimo-validation-aime", "aimo-validation-amc"):
        for path in glob.glob(str(HUB / f"datasets--AI-MO--{name}/snapshots/*/data/*.parquet")):
            df = pd.read_parquet(path)
            for problem, answer in zip(df["problem"], df["answer"]):
                value = parse_integer(str(answer))
                if value is None:
                    value = extract_answer(str(answer))
                if value is not None:
                    table.setdefault(normalize(problem), value)
    return table


def written_variants(problem: str) -> list[tuple[str, str]]:
    sys.path.insert(0, str(ROOT / "datagen"))
    from common import problem_id

    pid = problem_id(problem)
    out = []
    for path in (ROOT / "datagen/written_variants.jsonl", ROOT / "data/labelgen/variants.jsonl"):
        for row in read_jsonl(path):
            if row.get("problem_id") == pid:
                out.append((f"llm-{row['type']}-{row.get('variant', 0)}", row["text"]))
    return out


def probe_scores(cases: list[dict], model_id: str, layer_frac: float):
    """Leave-one-group-out diff-in-means projection of resp_mean; returns (scores, preds, full direction)."""
    acts = np.stack([np.load(ROOT / "data/cvbench/acts" / safe(model_id) / f"{c['group']}.npz")["resp_mean"]
                     .astype(np.float32) for c in cases])  # [n, L+1, d]
    layer = int(round(layer_frac * (acts.shape[1] - 1)))
    h = acts[:, layer]
    y = np.array([c["is_robust"] for c in cases])
    scores, preds = [], []
    for i in range(len(cases)):
        train = np.arange(len(cases)) != i
        mu_r, mu_n = h[train & y].mean(0), h[train & ~y].mean(0)
        direction = mu_r - mu_n
        direction /= np.linalg.norm(direction)
        mid = 0.5 * (mu_r + mu_n) @ direction  # threshold halfway between the class means
        s = float(h[i] @ direction - mid)
        scores.append(s)
        preds.append(s > 0)
    mu_r, mu_n = h[y].mean(0), h[~y].mean(0)
    direction = (mu_r - mu_n) / np.linalg.norm(mu_r - mu_n)
    proj = h @ direction
    return scores, preds, dict(layer=layer, direction=direction, mid=float(0.5 * (mu_r + mu_n) @ direction),
                               proj_std=float(proj.std()), mu_robust=mu_r, mu_spurious=mu_n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--layer-frac", type=float, default=0.5)
    args = parser.parse_args()

    cases = [c for c in read_jsonl(ROOT / "data/cvbench/cases.jsonl") if c["model_id"] == args.model]
    golds = gold_table()
    scores, preds, probe = probe_scores(cases, args.model, args.layer_frac)
    out = []
    for c, s, p in zip(cases, scores, preds):
        variants = {}
        for seed in args.seeds:
            for kind, text in make_perturbations(c["problem"], seed=seed):
                if text and text != c["problem"] and text not in variants.values():
                    variants[f"{kind}-s{seed}"] = text
        variants.update(dict(written_variants(c["problem"])))
        out.append(dict(group=c["group"], model_id=c["model_id"], problem=c["problem"], is_robust=c["is_robust"],
                        max_drop=c["max_drop"], gold=golds.get(normalize(c["problem"])),
                        probe_score=s, probe_pred=bool(p), probe_correct=bool(p) == c["is_robust"],
                        variants=variants))
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "cases.json").write_text(json.dumps(out, indent=1))
    np.savez(WORK / "probe.npz", **probe)
    print(f"{len(out)} cases for {args.model}; probe layer {probe['layer']}, "
          f"LOO accuracy {np.mean([o['probe_correct'] for o in out]):.3f}")
    for o in out:
        drop = "  n/a" if o["max_drop"] is None else f"{o['max_drop']:+.2f}"  # val-sample rows have no max_drop
        print(f"  {o['group']} robust={o['is_robust']!s:5} drop={drop} gold={o['gold']} "
              f"probe={o['probe_score']:+.3f} {'ok' if o['probe_correct'] else 'WRONG'} "
              f"variants={len(o['variants'])}")


if __name__ == "__main__":
    main()
