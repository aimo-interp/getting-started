#!/usr/bin/env python3
"""GPU check that the shipped inference path reproduces the cached training features.

Runs ``act_probe.inference`` end to end on cvbench cases of one model and compares,
per case, the response-mean hidden state with ``data/cvbench/acts`` and the
prediction with the one the artifact gives on the cached features.

    uv run solutions/act-probe/research/check_fidelity.py --model Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "cvbench"))

import cv  # noqa: E402
from act_probe import inference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()

    art = inference.load_artifact()
    entry = art["models"][args.model]
    gen = art["generation"]
    cases = [json.loads(line) for line in (cv.DATA / "cases.jsonl").read_text().splitlines() if line.strip()]
    cases = [c for c in cases if c["model_id"] == args.model][:args.limit]

    cached = [np.load(cv.acts_file(args.model, c["group"]))["resp_mean"][entry["layer"]].astype(np.float32)
              for c in cases]
    expected = [inference.decide(art, entry, h) for h in cached]

    t0 = time.time()
    got = inference.predict_robustness(args.model, [c["problem"] for c in cases])
    print(f"predict_robustness: {time.time() - t0:.0f}s for {len(cases)} problems")

    # Features again, to compare vectors (reuses the same functions; one more generation pass).
    model, tok = inference.load_model_and_tokenizer(args.model)
    responses = inference.greedy_responses(model, tok, [c["problem"] for c in cases], gen)
    greedy = {r["group"]: r["generated_text"] for r in cv.read_jsonl(cv.DATA / "greedy" / f"{cv.safe(args.model)}.jsonl")}
    for c, resp, h_old, e, g in zip(cases, responses, cached, expected, got):
        h = inference.response_mean(model, tok, c["problem"], resp, entry["layer"], entry["n_hidden_states"], gen)
        cos = float(h @ h_old / (np.linalg.norm(h) * np.linalg.norm(h_old) + 1e-8))
        z_old = float((h_old - entry["mean"]) @ entry["direction"]) / entry["scale"]
        z_new = float((h - entry["mean"]) @ entry["direction"]) / entry["scale"]
        same_text = resp == greedy.get(c["group"])
        print(f"{c['group']}: same_text={same_text} cos={cos:.4f} z cached {z_old:+.2f} new {z_new:+.2f} "
              f"pred cached {e} shipped {g} label {c['is_robust']}")
    agree = np.mean([e == g for e, g in zip(expected, got)])
    print(f"prediction agreement with cached features: {agree:.3f}")


if __name__ == "__main__":
    main()
