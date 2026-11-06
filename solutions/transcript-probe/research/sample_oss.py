#!/usr/bin/env python3
"""Sample gpt-oss-120b training transcripts for the transcript probe through OpenRouter.

For each labelled gpt-oss case (``data/cvbench/cases_oss.jsonl``: problem x
reasoning effort), draw ``--n`` samples on the original problem with the
model's own chat template at that effort, temperature 1.0, ``--max-tokens``
(16384, the small models' budget), pinned to one provider (Crusoe). Rows are
written in the small models' ``screen.jsonl`` format (group, variant X, texts,
n_tokens, finish), one file per effort, so ``train.py --oss`` reads them the
same way. Reruns only top up missing samples.

    python solutions/transcript-probe/research/sample_oss.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "solutions" / "anchors" / "thought_anchors"))

import resample  # noqa: E402
from resample import complete, load_tokenizer, read_jsonl, reasoning_prompt  # noqa: E402

MODEL = "openai/gpt-oss-120b"
OUT = ROOT / "data" / "transcript_probe" / "oss"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--provider", nargs="*", default=["Crusoe"])
    args = parser.parse_args()
    resample.PROVIDER[:] = args.provider
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY")

    cases = list(read_jsonl(ROOT / "data" / "cvbench" / "cases_oss.jsonl"))
    tokenizer = load_tokenizer(MODEL)
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "samples.jsonl"
    have = defaultdict(int)
    for row in read_jsonl(raw) if raw.exists() else []:
        have[(row["group"], row["effort"])] += 1
    jobs = [c for c in cases for _ in range(max(0, args.n - have[(c["group"], c["reasoning_effort"])]))]
    print(f"{len(cases)} cases, {len(jobs)} samples to run", flush=True)
    lock = threading.Lock()

    def one(case: dict) -> None:
        prompt = reasoning_prompt(tokenizer, case["problem"], case["reasoning_effort"])
        text, finish, provider = complete(key, MODEL, prompt, args.max_tokens, [])
        row = {"group": case["group"], "effort": case["reasoning_effort"], "text": text, "finish": finish,
               "n_tokens": len(tokenizer(text, add_special_tokens=False)["input_ids"]), "provider": provider}
        with lock, raw.open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, jobs))

    # Group into the small models' screen format: one file per effort, one row per problem.
    rows = defaultdict(lambda: {"texts": [], "n_tokens": [], "finish": []})
    for r in read_jsonl(raw):
        row = rows[(r["effort"], r["group"])]
        row["texts"].append(r["text"])
        row["n_tokens"].append(r["n_tokens"])
        row["finish"].append("stop" if r["finish"] in ("stop", "eos", "end_turn", "") else "length")
    for effort in sorted({e for e, _ in rows}):
        with (OUT / f"screen_{effort}.jsonl").open("w") as stream:
            for (e, group), row in sorted(rows.items()):
                if e == effort:
                    stream.write(json.dumps({"group": group, "variant": "X", **row}) + "\n")
    short = [k for k, row in rows.items() if len(row["texts"]) < args.n]
    print(f"wrote {OUT}/screen_*.jsonl; {len(short)} case(s) short of {args.n} samples", flush=True)


if __name__ == "__main__":
    main()
