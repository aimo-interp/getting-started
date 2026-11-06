#!/usr/bin/env python3
"""Does steering along the seg-probe direction make Skywork more robust? (2026-11-05)

Adds a constant vector to the residual stream after layer L/2 at every generated token. Directions are
out-of-fold: the diff-in-means gap (segments 0-4) is fitted without the test problem's fold. Unit: alpha=1 shifts
the state by one full robust-fragile class gap, ||mean(robust) - mean(fragile)||.

Conditions, on one problem's original + perturbed prompts, same seeds across conditions:
  base        no steering
  robust      +alpha * gap (towards robust)
  fragile     -alpha * gap (towards fragile), dose-response check, optional
  shuffled    gap from shuffled labels, rescaled to the real gap's norm (matched-norm control)

Targets: the 36 Skywork labelgen cases, the ones with gold answers and verified perturbations. Fragile cases only
by default. Fragile cases were selected for a high drop, so they regress to the mean on resampling: always compare
with `base`, never with the original label.

    python cvbench/steer.py --dry                               # CPU: check folds, directions, plan
    python cvbench/steer.py --limit 2 --alphas 0.25 0.5 1 2     # pilot: pick alpha on 2 problems, then exclude them
    python cvbench/steer.py --alphas 1 --shard 0/4              # one of 4 array tasks (cvbench/steer.slurm)
    python cvbench/steer.py --report                            # CPU: paired analysis of the rows on disk
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "cvbench"), str(ROOT / "solutions" / "transcript-probe")]
import cv  # noqa: E402

MODEL = "Skywork/Skywork-OR1-Math-7B"
LAYER = 14                     # hidden_states index = output of layers[LAYER - 1]
WINDOW = range(0, 5)           # segments 0-4, the seg probe's early window
TYPES = ("rephrase", "distract", "rename")  # fixed before seeing any steered output
SOLVE = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
OUT = cv.DATA / "steer" / cv.safe(MODEL)


def targets(roles):
    cases = cv.load_cases(["cases.jsonl"])
    cases = cases[cases.model_id == MODEL].reset_index()
    P = {r["problem"].strip(): r for r in cv.read_jsonl(ROOT / "data" / "labelgen" / "problems.jsonl")}
    V = {}
    for v in cv.read_jsonl(ROOT / "data" / "labelgen" / "variants.jsonl"):
        if v["type"] in TYPES and int(v["variant"]) == 0:
            V.setdefault(v["problem_id"], {})[v["type"]] = v["text"]
    y = cases.is_robust.astype(bool).to_numpy()
    H = np.stack([np.load(cv.DATA / "segs" / cv.safe(MODEL) / f"{g}.npz")["seg_mean"][list(WINDOW), LAYER]
                  .astype(np.float64).mean(0) for g in cases.group])
    fold = np.zeros(len(cases), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(H, y, cases.group)):
        fold[te] = k
    rng = np.random.default_rng(0)
    rows = []
    for i, c in enumerate(cases.itertuples()):
        p = P.get(c.problem.strip())
        if c.source != "labelgen" or p is None or (("robust" if c.is_robust else "fragile") not in roles):
            continue
        vs = V.get(p["problem_id"], {})
        if any(t not in vs for t in TYPES):
            print(f"  skip {c.group}: missing variant", flush=True)
            continue
        tr = fold != fold[i]
        gap = H[tr & y].mean(0) - H[tr & ~y].mean(0)
        ys = rng.permutation(y[tr])
        sh = H[tr][ys].mean(0) - H[tr][~ys].mean(0)
        sh *= np.linalg.norm(gap) / np.linalg.norm(sh)
        prompts = {"original": c.problem, **{t: vs[t] for t in TYPES}}
        rows.append(dict(group=c.group, gold=int(p["answer"]), robust=bool(c.is_robust), max_drop=c.max_drop,
                         prompts=prompts, gap=gap, shuffled=sh))
    return rows


def steer_hook(vec):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] != 1:          # prompt pass: leave the problem encoding untouched
            return output
        h = h + vec.to(h.dtype).to(h.device)
        return (h, *output[1:]) if isinstance(output, tuple) else h
    return hook


def run(args):
    import torch
    import generate as G
    from transcript_probe.features import extract_answer
    from uncertainty_profile.extraction import load_model_and_tokenizer
    rows = targets(args.roles)
    i, n = map(int, args.shard.split("/"))
    rows = rows[:args.limit] if args.limit else rows
    jobs = [(r, name, a) for r in rows for a in args.alphas for name in args.conditions]
    jobs = [j for k, j in enumerate(jobs) if k % n == i]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"shard{i}.jsonl"
    done = {(r["group"], r["prompt"], r["condition"], r["alpha"]) for r in cv.read_jsonl(path)}
    model, tok = load_model_and_tokenizer(MODEL, local_files_only=True)
    model.eval()
    device = model.get_input_embeddings().weight.device
    layer = G.decoder_layers(model)[LAYER - 1]
    for r, name, alpha in jobs:
        for ptype, text in r["prompts"].items():
            if (r["group"], ptype, name, alpha) in done:
                continue
            t0 = time.time()
            ids = tok(G.format_problem(tok, text + SOLVE), add_special_tokens=False, return_tensors="pt").to(device)
            u = {"base": 0 * r["gap"], "robust": alpha * r["gap"], "fragile": -alpha * r["gap"],
                 "shuffled": alpha * r["shuffled"]}[name]
            handle = layer.register_forward_hook(steer_hook(torch.tensor(u, dtype=torch.float32)))
            torch.manual_seed(int(r["group"][:8], 16) % 2**31 + list(r["prompts"]).index(ptype))
            try:
                with torch.no_grad():
                    out = model.generate(**ids, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                         num_return_sequences=args.n, max_new_tokens=args.budget,
                                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
            finally:
                handle.remove()
            gen = out[:, ids["input_ids"].shape[1]:]
            finished = [(g == (tok.pad_token_id or tok.eos_token_id)).any().item() or len(g) < args.budget for g in gen]
            answers = [extract_answer(tok.decode(g, skip_special_tokens=True)) if f else None
                       for g, f in zip(gen, finished)]
            row = dict(group=r["group"], prompt=ptype, condition=name, alpha=alpha, gold=r["gold"], answers=answers,
                       finished=float(np.mean(finished)), acc=float(np.mean([a == r["gold"] for a in answers])),
                       seconds=round(time.time() - t0))
            with path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(row, flush=True)
            del out, gen
            torch.cuda.empty_cache()


def report():
    import pandas as pd
    R = pd.DataFrame([r for p in sorted(OUT.glob("shard*.jsonl")) for r in cv.read_jsonl(p)])
    rng = np.random.default_rng(0)
    for alpha, D in R.groupby("alpha"):
        A = D.pivot_table(index=["group", "condition"], columns="prompt", values="acc")
        pert = A.drop(columns="original")
        S = pd.DataFrame({"original": A["original"], "perturbed": pert.mean(1),
                          "max_drop": (-pert.sub(A["original"], axis=0)).max(1),
                          "finished": D.groupby(["group", "condition"]).finished.mean()})
        print(f"\nalpha {alpha}  n problems {D.group.nunique()}\n", S.groupby("condition").mean().round(3))
        for c in ("robust", "fragile", "shuffled"):
            if c not in S.index.get_level_values(1) or "base" not in S.index.get_level_values(1):
                continue
            for m in ("perturbed", "max_drop", "original"):
                d = (S.xs(c, level=1)[m] - S.xs("base", level=1)[m]).dropna().to_numpy()
                boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]
                print(f"  {c} - base {m}: {d.mean():+.3f} [{np.quantile(boot, .025):+.3f},{np.quantile(boot, .975):+.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    ap.add_argument("--conditions", nargs="+", default=["base", "robust", "shuffled"])
    ap.add_argument("--roles", nargs="+", default=["fragile"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()
    if args.dry:
        rows = targets(args.roles)
        g = np.array([np.linalg.norm(r["gap"]) for r in rows])
        c = np.mean([r["gap"] @ r["shuffled"] / (np.linalg.norm(r["gap"]) * np.linalg.norm(r["shuffled"])) for r in rows])
        batches = len(rows) * len(args.alphas) * len(args.conditions) * (1 + len(TYPES))
        print(f"{len(rows)} problems; {batches} batches of {args.n}; gap norm {g.mean():.2f}; gap/shuffled cosine {c:.3f}")
        print("max_drop", np.round([r["max_drop"] for r in rows], 2))
        return None
    return run(args)


if __name__ == "__main__":
    main()
