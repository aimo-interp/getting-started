"""Budget-forced answers for the sep2 samples that hit the token cap (s1-style budget forcing).

Most sep2 samples stop at the 8,192-token cap with no answer, so semantic entropy over them
mostly counts truncations. Per truncated sample, this appends a closing ``</think>`` and an
opening ``\\boxed{`` to the prompt + stored trace and decodes the answer greedily (a few tokens).
Answers that were already parsed are kept. Resumable: finished prompts are skipped.

Writes ``sep2/samples/<model>__default.forced.jsonl`` (one row per sample row, with ``answers``
filled and a ``forced`` mask) and ``sep2/<model>__forced.npz`` with the answers in the row order
of ``sep2/<model>.npz`` (``whitebox.py`` sep2_probe ``--sep-answers forced``).
Runs in the vLLM environment:

    python cvbench/force_answers.py --model Skywork/Skywork-OR1-Math-7B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "datagen"))

from common import append_jsonl, extract_answer, read_jsonl  # noqa: E402

DATA = ROOT / "data" / "cvbench"
FORCE = "\n</think>\n\nThe final answer is:\n\\[\n\\boxed{"


def safe(model_id: str) -> str:
    return model_id.replace("/", "_")


def prompt_texts() -> dict[tuple, str]:
    """(problem_id, type, variant) -> user text, as datagen/sample.py builds its jobs."""
    texts = {(r["problem_id"], "original", 0): r["problem"] for r in read_jsonl(ROOT / "data" / "labelgen" / "problems.jsonl")}
    for r in read_jsonl(ROOT / "data" / "labelgen" / "variants.jsonl"):
        texts[(r["problem_id"], r["type"], int(r["variant"]))] = r["text"]
    return texts


def force(args) -> Path:
    from vllm import LLM, SamplingParams

    base = DATA / "sep2" / "samples" / f"{safe(args.model)}__default"
    out = base.with_name(base.name + ".forced.jsonl")
    done = {(r["problem_id"], r["type"], int(r["variant"])) for r in read_jsonl(out)}
    texts = prompt_texts()
    rows = [(json.loads(s), json.loads(t)) for s, t in zip(open(f"{base}.jsonl"), open(f"{base}.traces.jsonl"))]
    rows = [(s, t) for s, t in rows if (s["problem_id"], s["type"], int(s["variant"])) not in done]
    print(f"[force] {args.model}: {len(rows)} prompts to do ({len(done)} done)", flush=True)
    if not rows:
        return out

    llm = LLM(args.model, seed=0, max_model_len=args.max_model_len, enable_prefix_caching=True,
              gpu_memory_utilization=args.gpu_memory_utilization)
    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=0.0, max_tokens=16, stop=["}"])
    for start in range(0, len(rows), args.chunk_size):
        chunk = rows[start:start + args.chunk_size]
        prompts, where = [], []
        for i, (s, t) in enumerate(chunk):
            key = (s["problem_id"], s["type"], int(s["variant"]))
            assert key == (t["problem_id"], t["type"], int(t["variant"])), key
            head = tok.apply_chat_template([{"role": "user", "content": texts[key]}], tokenize=False, add_generation_prompt=True)
            for j, (a, text) in enumerate(zip(s["answers"], t["texts"])):
                if a is None:
                    prompts.append(head + text + FORCE)
                    where.append((i, j))
        answers = [list(s["answers"]) for s, _ in chunk]
        forced = [[False] * len(s["answers"]) for s, _ in chunk]
        for (i, j), o in zip(where, llm.generate(prompts, params, use_tqdm=False)):
            answers[i][j] = extract_answer("\\boxed{" + o.outputs[0].text + "}")
            forced[i][j] = True
        append_jsonl(out, [{"problem_id": s["problem_id"], "type": s["type"], "variant": s["variant"],
                            "answers": a, "forced": f, "n_tokens": s["n_tokens"]}
                           for (s, _), a, f in zip(chunk, answers, forced)])
        n_forced = sum(map(sum, forced))
        n_parsed = sum(a is not None for (i, j) in where for a in [answers[i][j]])
        print(f"  {min(start + args.chunk_size, len(rows))}/{len(rows)} prompts; "
              f"forced {n_forced}, parsed {n_parsed}", flush=True)
    return out


def align(model_id: str, forced_path: Path) -> None:
    """The forced answers in the row order of sep2/<model>.npz (generate.py run_sep2's filter)."""
    texts = prompt_texts()
    greedy = {(r["problem_id"], r["type"], int(r["variant"]))
              for r in read_jsonl(DATA / "sep2" / "greedy" / f"{safe(model_id)}__default.traces.jsonl")}
    forced = {(r["problem_id"], r["type"], int(r["variant"])): r["answers"] for r in read_jsonl(forced_path)}
    rows = []
    for r in read_jsonl(DATA / "sep2" / "samples" / f"{safe(model_id)}__default.jsonl"):
        key = (r["problem_id"], r["type"], int(r["variant"]))
        if texts.get(key) and key in greedy:
            rows.append((r["answers"], forced.get(key)))
    z = np.load(DATA / "sep2" / f"{safe(model_id)}.npz")
    assert [json.loads(a) for a in z["answers"]] == [a for a, _ in rows], "row order differs from the sep2 npz"
    missing = sum(f is None for _, f in rows)
    if missing:
        print(f"[force] {missing} prompts not forced yet; not writing the aligned npz", flush=True)
        return
    path = DATA / "sep2" / f"{safe(model_id)}__forced.npz"
    np.savez(path, answers=np.array([json.dumps(f) for _, f in rows]))
    print(f"[force] wrote {path.name} ({len(rows)} prompts)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--chunk-size", type=int, default=64, help="Prompts per vLLM call (resume granularity).")
    parser.add_argument("--max-model-len", type=int, default=10240)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--align-only", action="store_true", help="Skip generation; only write the aligned npz.")
    args = parser.parse_args()
    out = DATA / "sep2" / "samples" / f"{safe(args.model)}__default.forced.jsonl"
    if not args.align_only:
        out = force(args)
    align(args.model, out)


if __name__ == "__main__":
    main()
