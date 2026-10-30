#!/usr/bin/env python3
"""Reader-model probe (loop 2026-10-31): one fixed model reads every target model's greedy chain of thought.

Per-model probes are label-starved (7-48 cases each) because every model has its own hidden space. Here a
single reader (default Qwen3.5-4B) runs one forward pass over [problem prompt + target's greedy 4k response]
for all 118 cases, so all cases share one space and one diff-in-means direction can use all 35 fragile cases.
Saves the means over 10 equal response segments at every layer -> ``reader/<reader>/<target>/<group>.npz``.

    pyine/bin/python cvbench/reader.py                      # GPU
    pyine/bin/python cvbench/stack.py --blocks rdr traj tx  # CV (stack.py reader blocks)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "solutions" / "uncertainty-profiling"), str(ROOT / "cvbench")]
from uncertainty_profile.extraction import format_problem, load_model_and_tokenizer  # noqa: E402

DATA = ROOT / "data" / "cvbench"


def safe(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--segments", type=int, default=10)
    args = ap.parse_args()
    cases = [json.loads(l) for f in ("cases.jsonl", "cases_oss.jsonl") for l in open(DATA / f)]
    greedy = {}
    for f in (DATA / "greedy").glob("*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            greedy[(r["model_id"], r["group"])] = r.get("generated_text") or ""
    model, tok = load_model_and_tokenizer(args.reader, local_files_only=True)
    device = model.get_input_embeddings().weight.device
    out_root = DATA / "reader" / safe(args.reader)
    from tqdm import tqdm
    for c in tqdm(cases, desc="reader"):
        out = out_root / safe(c["model_id"]) / f"{c['group']}.npz"
        if out.is_file():
            continue
        text = greedy.get((c["model_id"], c["group"]), "")
        p_ids = tok(format_problem(tok, c["problem"]), add_special_tokens=False)["input_ids"]
        r_ids = tok(text, add_special_tokens=False)["input_ids"][:6144]
        hs = model(input_ids=torch.tensor([p_ids + r_ids], device=device), output_hidden_states=True,
                   logits_to_keep=1).hidden_states
        p, n = len(p_ids), len(r_ids)
        last = torch.stack([h[0, p - 1] for h in hs]).float().cpu().numpy()
        if n >= args.segments:
            b = np.linspace(0, n, args.segments + 1).round().astype(int)
            seg = torch.stack([torch.stack([h[0, p + x:p + z].float().mean(0) for x, z in zip(b[:-1], b[1:])])
                               for h in hs], 1).cpu().numpy()
        else:
            seg = np.repeat(last[None], args.segments, 0)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, seg_mean=seg.astype(np.float16), last_prompt=last.astype(np.float16), n_resp=n)
        del hs
    print("done", flush=True)


if __name__ == "__main__":
    main()
