#!/usr/bin/env python3
"""Attention on an organiser distractor in right vs wrong gpt-oss-120b:low traces (cvbench, 2026-10-30).

Material is organiser-only: ``aimo-interp-challenge-sample-full`` lists, for gpt-oss-120b:low, the
perturbed problem texts that made it fail (``permutations_causing_decay``). For the ``distract`` type the
perturbation is an inserted clause (sometimes a few), located by a word-level diff against the original. The texts of the variants
it got right are not given, so the contrast is between traces on the same prompt: sample ``--n`` traces
per failing variant, grade each against the modal answer on the original problem, and compare how the
right and wrong traces attend to the inserted clause, token by token.

Commands (resumable, under ``data/cvbench/distract/``):

* ``build``: items.jsonl, one row per failing distract variant on a problem with base accuracy > 0.
* ``sample``: ``--n`` traces per variant and ``--n-orig`` per original problem through OpenRouter (low
  effort, T=1, the transcript probe's prompt and provider) -> samples.jsonl. Needs the key from .env.
* ``attn`` (GPU, 80 GB): one eager pass over prompt + reasoning per sampled variant trace, chunked over a KV
  cache; per response token and layer, the attention mass on the clause tokens (head mean and head max)
  and on the whole prompt (sink excluded), plus the per-(layer, head) mean over the trace ->
  attn/<item>__<k>.npz.
* ``analyze``: right vs wrong within each variant, per layer and pooled; prints the top-spiking tokens.

    uv run cvbench/distract_attn.py build
    set -a; source .env; set +a; uv run cvbench/distract_attn.py sample
    python cvbench/distract_attn.py attn          # cvbench/distract.slurm
    uv run cvbench/distract_attn.py analyze
"""

from __future__ import annotations

import argparse
import difflib
import glob
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "solutions" / "anchors" / "thought_anchors"),
                str(ROOT / "solutions" / "uncertainty-profiling")]

OUT = ROOT / "data" / "cvbench" / "distract"
SAMPLE_FULL = "/data/gpfs/projects/punim0478/bansaab/hub/datasets--aimo-interp--aimo-interp-challenge-sample-full"
ORG_MODEL, MODEL, EFFORT = "gpt-oss-120b:low", "openai/gpt-oss-120b", "low"
MAX_RESP = 8192  # reasoning tokens kept for the attention pass


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()] if path.is_file() else []


def inserted_spans(original: str, perturbed: str) -> list[tuple[int, int]]:
    """Character spans in ``perturbed`` inserted or replaced vs ``original`` (word-level diff, so a long
    insertion is one span rather than characters matched by chance inside it)."""
    import re
    a, b = re.findall(r"\S+|\s+", original), re.findall(r"\S+|\s+", perturbed)
    starts = np.cumsum([0] + [len(t) for t in b])
    spans = [(int(starts[j1]), int(starts[j2])) for tag, _, _, j1, j2 in
             difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if tag != "equal" and j2 > j1]
    return [(x, y) for x, y in spans if perturbed[x:y].strip()]


# --------------------------------------------------------------------------- build
def cmd_build(args) -> None:
    import pandas as pd
    s = pd.read_parquet(glob.glob(f"{SAMPLE_FULL}/snapshots/*/data/*.parquet")[0])
    s = s[(s.model_id == ORG_MODEL) & (s.permutation_type == "distract")]
    items = []
    for r in s.itertuples():
        if float(r.base_accuracy) <= 0:
            continue
        for k, q in enumerate(json.loads(r.permutations_causing_decay or "[]")):
            spans = inserted_spans(r.original_problem, q["question"])
            if not spans:
                continue
            items.append(dict(item=f"{r.problem_id}_{k}", problem_id=r.problem_id, original=r.original_problem,
                              perturbed=q["question"], spans=spans,
                              clause=" … ".join(q["question"][x:y] for x, y in spans),
                              base_accuracy=float(r.base_accuracy), variant_accuracy=float(q["variant_accuracy"])))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "items.jsonl").open("w") as f:
        f.writelines(json.dumps(i) + "\n" for i in items)
    print(f"{len(items)} failing distract variants on {len({i['problem_id'] for i in items})} problems "
          f"-> {OUT / 'items.jsonl'}")
    for i in items[:5]:
        print(f"  {i['item']}: {i['clause']!r}")


# --------------------------------------------------------------------------- sample
def cmd_sample(args) -> None:
    import resample
    from resample import complete, extract_answer, load_tokenizer, reasoning_of, reasoning_prompt
    resample.PROVIDER[:] = args.provider
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY (source .env)")
    items = read_jsonl(OUT / "items.jsonl")
    originals = {i["problem_id"]: i["original"] for i in items}
    path = OUT / "samples.jsonl"
    have = Counter(r["item"] for r in read_jsonl(path))
    jobs = [(i["item"], i["perturbed"]) for i in items for _ in range(max(0, args.n - have[i["item"]]))]
    jobs += [(f"{p}_orig", text) for p, text in originals.items()
             for _ in range(max(0, args.n_orig - have[f"{p}_orig"]))]
    print(f"{len(jobs)} samples to run", flush=True)
    tokenizer = load_tokenizer(MODEL)
    lock = threading.Lock()

    def one(job):
        item, text = job
        out, finish, provider = complete(key, MODEL, reasoning_prompt(tokenizer, text, EFFORT), args.max_tokens, [])
        reasoning = reasoning_of(out)
        row = dict(item=item, text=out, reasoning=reasoning, answer=extract_answer(out[len(reasoning):]),
                   finish=finish, provider=provider)
        with lock, path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    from tqdm import tqdm
    with ThreadPoolExecutor(args.workers) as pool:
        list(tqdm(pool.map(one, jobs), total=len(jobs), desc="sample"))
    rows = read_jsonl(path)
    for p in originals:
        answers = Counter(r["answer"] for r in rows if r["item"] == f"{p}_orig")
        print(f"  {p} original answers: {dict(answers.most_common(4))}")


# --------------------------------------------------------------------------- attn (GPU)
def cmd_attn(args) -> None:
    import torch
    from transformers import DynamicCache
    from resample import reasoning_prompt
    from uncertainty_profile.extraction import load_model_and_tokenizer

    items = {i["item"]: i for i in read_jsonl(OUT / "items.jsonl")}
    rows = [r for r in read_jsonl(OUT / "samples.jsonl") if r["item"] in items]
    todo, seen = [], Counter()
    for r in rows:
        k = seen[r["item"]]
        seen[r["item"]] += 1
        if not (OUT / "attn" / f"{r['item']}__{k}.npz").is_file():
            todo.append((r, k))
    print(f"[attn] {len(todo)} of {len(rows)} traces to do", flush=True)
    if not todo:
        return
    (OUT / "attn").mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(MODEL, local_files_only=True)
    model.set_attn_implementation("eager")
    device = model.get_input_embeddings().weight.device
    cfg = getattr(model.config, "text_config", model.config)
    heads = cfg.num_attention_heads
    torch.set_grad_enabled(False)
    from tqdm import tqdm
    for r, k in tqdm(todo, desc="attn"):
        start = time.time()
        item = items[r["item"]]
        prompt = reasoning_prompt(tokenizer, item["perturbed"], EFFORT)
        at = prompt.find(item["perturbed"])
        enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        prompt_ids = enc["input_ids"]
        clause = torch.tensor([any(a < at + y and b > at + x for x, y in item["spans"])
                               for a, b in enc["offset_mapping"]], device=device)
        response_ids = tokenizer(r["reasoning"], add_special_tokens=False)["input_ids"][:MAX_RESP]
        p, n = len(prompt_ids), len(response_ids)
        ids = prompt_ids + response_ids
        cache = DynamicCache()  # plain cache: sliding layers keep every key, the mask comes from the config
        span = ctx = smax = lh = None
        for a in range(0, len(ids), 128):
            piece = torch.tensor([ids[a:a + 128]], device=device)
            pos = torch.arange(a, a + piece.shape[1], device=device)
            out = model(input_ids=piece, past_key_values=cache, use_cache=True, cache_position=pos,
                        position_ids=pos[None], output_attentions=True, logits_to_keep=1)
            atts = [att for att in out.attentions if att is not None]
            if span is None:
                L = len(atts)
                span, smax, ctx = (torch.zeros(n, L, device=device) for _ in range(3))
                lh = torch.zeros(L, heads, dtype=torch.float64, device=device)
            q = pos >= p
            if q.any():
                t = pos[q] - p
                for j, att in enumerate(atts):
                    w = att[0][:, q, :p].float().to(device)  # [H, q, prompt keys]; layers may sit on other GPUs
                    m = (w * clause).sum(-1)  # [H, q]
                    span[t, j], smax[t, j] = m.mean(0), m.max(0).values
                    ctx[t, j] = w[..., 1:].sum(-1).mean(0)
                    lh[j] += m.sum(-1).double()
            del out, atts
        np.savez_compressed(OUT / "attn" / f"{r['item']}__{k}.npz",
                            span=span.cpu().numpy().astype(np.float16), span_max=smax.cpu().numpy().astype(np.float16),
                            ctx=ctx.cpu().numpy().astype(np.float16), span_lh=(lh / max(n, 1)).float().cpu().numpy(),
                            n_resp=n, n_prompt=p, n_clause=int(clause.sum()), response_ids=np.array(response_ids))
        del cache
        torch.cuda.empty_cache()
        tqdm.write(f"  {r['item']}#{k}: {n} resp tokens, clause {int(clause.sum())} tok, {time.time() - start:.0f}s")


# --------------------------------------------------------------------------- analyze
def cmd_analyze(args) -> None:
    from scipy.stats import wilcoxon
    from transformers import AutoTokenizer
    items = {i["item"]: i for i in read_jsonl(OUT / "items.jsonl")}
    rows = read_jsonl(OUT / "samples.jsonl")
    ref = {}
    for p in {i["problem_id"] for i in items.values()}:
        answers = Counter(r["answer"] for r in rows if r["item"] == f"{p}_orig" and r["answer"] is not None)
        ref[p] = answers.most_common(1)[0][0] if answers else None
    recs, seen = [], Counter()
    for r in rows:
        if r["item"] not in items:
            continue
        k = seen[r["item"]]
        seen[r["item"]] += 1
        f = OUT / "attn" / f"{r['item']}__{k}.npz"
        if not f.is_file():
            continue
        z = np.load(f)
        span, ctx = z["span"].astype(np.float32), z["ctx"].astype(np.float32)  # [n, L]
        n = len(span)
        share = span.mean(1) / np.maximum(ctx.mean(1), 1e-9)  # clause share of prompt attention, per token
        item = items[r["item"]]
        recs.append(dict(item=r["item"], k=k, problem=item["problem_id"], wrong=r["answer"] != ref[item["problem_id"]],
                         n=n, mass=span.mean(), share=share.mean(), peak=np.quantile(span.mean(1), 0.99),
                         early=span[: max(n // 4, 1)].mean(), late=span[3 * n // 4:].mean(),
                         layer=span.mean(0), lh=z["span_lh"], f=f))
    print(f"{len(recs)} traces with attention; reference answers {ref}")
    import pandas as pd
    d = pd.DataFrame(recs)
    print(f"wrong rate {d.wrong.mean():.2f}; per problem:\n{d.groupby('problem').wrong.agg(['mean', 'size']).round(2)}")
    mixed = d.groupby("item").wrong.transform(lambda w: 0 < w.mean() < 1).astype(bool)
    m = d[mixed]
    print(f"\n{m['item'].nunique()} variants have both right and wrong traces ({len(m)} traces)")
    for col in ("mass", "share", "peak", "early", "late", "n"):
        g = m.groupby(["item", "wrong"])[col].mean().unstack()
        diff = g[True] - g[False]
        p = wilcoxon(diff).pvalue if len(diff) >= 5 and diff.abs().sum() > 0 else float("nan")
        print(f"  {col:6s} wrong-right per variant: median {diff.median():+.4g}, wrong higher in "
              f"{(diff > 0).sum()}/{len(diff)}, Wilcoxon p={p:.3f}")
    if len(m):
        layer = np.stack(m.layer.to_numpy())
        w = m.wrong.to_numpy()
        gap = layer[w].mean(0) - layer[~w].mean(0)
        top = np.argsort(-np.abs(gap))[:5]
        print("  layers with the largest wrong-right gap in clause mass:",
              ", ".join(f"L{j} {gap[j]:+.4f}" for j in top))
    tok = AutoTokenizer.from_pretrained(MODEL)
    for wrong in (True, False):
        print(f"\n=== top clause-attention tokens, {'WRONG' if wrong else 'RIGHT'} traces (head mean over layers) ===")
        for rec in d[d.wrong == wrong].sort_values("peak", ascending=False).head(args.show).itertuples():
            z = np.load(rec.f)
            ids, s = z["response_ids"], z["span"].astype(np.float32).mean(1)
            for t in np.argsort(-s)[:3]:
                ctx_text = tok.decode(ids[max(t - 12, 0):t]) + " [[" + tok.decode(ids[t:t + 1]) + "]] " + \
                           tok.decode(ids[t + 1:t + 6])
                print(f"  {rec.item}#{rec.k} t={t}/{rec.n} mass {s[t]:.3f}: {ctx_text!r}")
        print(f"  clause of last shown: {items[rec.item]['clause']!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=8)
    s.add_argument("--n-orig", type=int, default=16)
    s.add_argument("--max-tokens", type=int, default=16384)
    s.add_argument("--workers", type=int, default=32)
    s.add_argument("--provider", nargs="*", default=["Crusoe"])
    sub.add_parser("attn")
    a = sub.add_parser("analyze")
    a.add_argument("--show", type=int, default=6)
    args = ap.parse_args()
    {"build": cmd_build, "sample": cmd_sample, "attn": cmd_attn, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
