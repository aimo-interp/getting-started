#!/usr/bin/env python3
"""GPU check that the shipped inference path reproduces the cached training features.

Runs ``seg_probe.inference`` end to end on cvbench cases of one model and compares,
per case, the window means with ``data/cvbench/segs`` and the prediction with the
one the artifact gives on the cached features.

    uv run solutions/seg-probe/research/check_fidelity.py --model Qwen/Qwen3.5-4B
    uv run solutions/seg-probe/research/check_fidelity.py --model openai/gpt-oss-120b:medium   # case model id
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "cvbench"))

import cv  # noqa: E402
from seg_probe import inference  # noqa: E402


def cached_windows(art: dict, model_id: str, group: str, layer: int) -> list[np.ndarray]:
    seg = np.load(cv.DATA / "segs" / cv.safe(model_id) / f"{group}.npz")["seg_mean"].astype(np.float32)[:, layer]
    return [seg[a:b].mean(0) for a, b in art["windows"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()

    art = inference.load_artifact()
    base, _, effort_arg = args.model.partition(":")
    key, effort = inference.artifact_key(art, base, effort_arg or "default")
    entry = art["models"][key]
    gen = art["generation"]
    cases = cv.load_cases().reset_index().to_dict("records")
    cases = [c for c in cases if c["model_id"] == key][:args.limit]

    cached = [cached_windows(art, args.model, c["group"], entry["layer"]) for c in cases]
    expected = [inference.decide(art, entry, hs) for hs in cached]

    t0 = time.time()
    got = inference.predict_robustness(base, [c["problem"] for c in cases], effort_arg or "default")
    print(f"predict_robustness: {time.time() - t0:.0f}s for {len(cases)} problems")

    # Features again, to compare vectors (reuses the same functions; one more generation pass).
    model, tok = inference.load_model_and_tokenizer(base)
    responses = inference.greedy_responses(model, tok, [c["problem"] for c in cases], gen, effort)
    greedy = {r["group"]: r["generated_text"] for r in cv.read_jsonl(cv.DATA / "greedy" / f"{cv.safe(args.model)}.jsonl")}
    for c, resp, old, e, g in zip(cases, responses, cached, expected, got):
        new = inference.window_means(model, tok, c["problem"], resp, entry["layer"], entry["n_hidden_states"], gen,
                                     art["segments"], art["windows"], effort)
        parts = []
        for k, (w, h_old, h) in enumerate(zip(entry["windows"], old, new)):
            cos = float(h @ h_old / (np.linalg.norm(h) * np.linalg.norm(h_old) + 1e-8))
            z_old = float((h_old - w["mean"]) @ w["direction"]) / w["scale"]
            z_new = float((h - w["mean"]) @ w["direction"]) / w["scale"]
            parts.append(f"w{k} cos={cos:.4f} z {z_old:+.2f}->{z_new:+.2f}")
        same_text = resp == greedy.get(c["group"])
        print(f"{c['group']}: same_text={same_text} {' | '.join(parts)} "
              f"pred cached {e} shipped {g} label {c['is_robust']}")
    agree = np.mean([e == g for e, g in zip(expected, got)])
    print(f"prediction agreement with cached features: {agree:.3f}")


if __name__ == "__main__":
    main()
