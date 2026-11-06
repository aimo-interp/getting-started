#!/usr/bin/env python3
"""CPU check that each package's shipped runtime reproduces its training: feed ``heads.decide`` the cached
cvbench signals of every training case and compare with the exporter's train accuracy.

    uv run solutions/trace-heads/research/check_runtime.py length-probe cot-text act-probe
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "cvbench"), str(HERE)]

import cv  # noqa: E402
from trace_engine import engine, heads  # noqa: E402


def signals_for(cases):
    greedy = cv.greedy_table(cases)
    intro = cv.by_group(cases, [r for m in cases.model_id.unique()
                                for r in cv.read_jsonl(cv.DATA / "introspect" / f"{cv.safe(m)}.jsonl")])
    for case_id, c in cases.iterrows():
        g = greedy.loc[case_id]
        s = {"text": g["generated_text"] if isinstance(g["generated_text"], str) else "",
             "n_tokens": int(g["generation_num_tokens"]) if g["generation_num_tokens"] == g["generation_num_tokens"] else 0,
             "uncertainty": g.to_dict(),
             "introspect": {k: intro.loc[case_id].get(k, np.nan) for k in ("robust", "correct", "long")}}
        z = np.load(cv.acts_file(c.model_id, c.group))
        s["last_prompt"], s["resp_mean"] = z["last_prompt"].astype(np.float32), z["resp_mean"].astype(np.float32)
        seg = cv.DATA / "segs" / cv.safe(c.model_id) / f"{c.group}.npz"
        if seg.is_file():
            s["seg_mean"] = np.load(seg)["seg_mean"].astype(np.float32)
        yield c, s


def main() -> None:
    cases = cv.load_cases()
    for name in sys.argv[1:]:
        art = json.loads((ROOT / "solutions" / name / "artifacts" / "head.json").read_text())
        ok, n = 0, 0
        for c, s in signals_for(cases):
            key, _ = engine.artifact_key(art["models"], *c.model_id.partition(":")[::2])
            entry = art["models"].get(key)
            pred = heads.decide(art, entry, s) if entry is not None else True
            ok += pred == bool(c.is_robust)
            n += 1
        print(f"{name:15s} runtime train acc {ok / n:.3f} vs exporter {art['training']['train_accuracy']:.3f}", flush=True)


if __name__ == "__main__":
    main()
