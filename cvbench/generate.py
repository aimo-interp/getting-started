#!/usr/bin/env python3
"""GPU stage: cache every model-side signal the CV methods need.

Stages, each resumable and written under ``data/cvbench/<stage>/<model>.*``:

* ``greedy``: the uncertainty-profiling baseline's own generation (greedy,
  2,048 prompt / 4,096 new tokens, user-only chat template) with its 14
  features, plus the generated text.
* ``acts``: one forward pass per problem over prompt + greedy response,
  recording hidden states at every layer for the last prompt token, the mean
  over the response, and the last prompt token of each code-made perturbation
  (``perturb_probe.perturbations``); plus the per-token entropy and selected
  log-prob trace of the greedy response.
* ``pool``: prompt-only forward passes for X and its 4 perturbations, saving at
  every layer the mean over the problem-text tokens and over the last 4 chat-template
  tokens (``prob_mean``/``tmpl_last`` and their ``pert_*`` counterparts).
* ``segs``: the ``acts`` forward pass again, saving at every layer the mean over
  each of ``--segments`` equal token segments of the greedy response
  (``seg_mean`` [S, L+1, d]), so any contiguous pooling window over the CoT can be
  scored offline (``whitebox.py seg_probe``).
* ``sep``: training data for the semantic-entropy probe. For every labelgen prompt
  of the model (originals and perturbed variants, ``data/labelgen/samples``), the
  last-prompt-token hidden state at every layer (same template as ``acts``) and the
  answer entropy over its saved samples (math: semantic entropy = entropy of the
  final answer; no answer counts as one class) -> ``sep/<model>.npz``.
* ``sep2``: the semantic-entropy probe data in the paper's setup (Kossen et al., 2024).
  Needs ``sep2/samples`` (10 samples per prompt at T=1, top-p 0.9, top-k 50) and
  ``sep2/greedy`` (one greedy response per prompt), both from ``datagen/sample.py
  --no-suffix`` (``cvbench/sep2.slurm``). Per prompt, one pass over prompt + greedy
  response saves the hidden state at the token before generation (TBG) and the last
  response token (SLT) at every layer, plus the sampled answers -> ``sep2/<model>.npz``;
  the cvbench cases' SLT from their cached greedy responses -> ``sep2/<model>__cases.npz``.
* ``lens``: the length probe's training data. For every original problem the model was sampled on
  (``len/samples``, from ``cvbench/lens.slurm``; else ``sep2/samples``), the unpadded last-prompt hidden
  state at every layer and the mean log length of its sampled responses -> ``len/<model>.npz``.
* ``introspect``: zero-shot self-prediction (Binder et al., 2024, "Looking Inward"): hypothetical questions
  about the model's own behaviour on the problem, answered with no chain of thought (the thinking block is
  closed), read as the Yes-vs-No log-odds of the next token -> ``introspect/<model>.jsonl``.
* ``sample``: the forced-answer perturbation probe (X + 4 perturbations,
  ``--n`` traces each, ``--budget`` thinking tokens, forced answers and
  answer log-probs).

    uv run cvbench/generate.py --stages greedy acts
    uv run cvbench/generate.py --cases cases_oss.jsonl --stages greedy acts segs pool   # gpt-oss-120b by effort

A model id may carry a reasoning effort (``openai/gpt-oss-120b:medium``): the base model is
loaded once and every stage formats prompts with that effort.
    uv run cvbench/generate.py --stages sample --n 4 --budget 2048
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "solutions" / "uncertainty-profiling"), str(ROOT / "solutions" / "perturb-probe")]

from perturb_probe.perturbations import make_perturbations  # noqa: E402
from perturb_probe.probe import probe_problem  # noqa: E402
from uncertainty_profile.config import GenerationConfidenceConfig  # noqa: E402
from uncertainty_profile.extraction import (  # noqa: E402
    format_problem,
    iter_generation_feature_batches,
    load_model_and_tokenizer,
    release_model,
)

DATA = ROOT / "data" / "cvbench"
MODELS = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
          "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]


def safe(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def run_greedy(model, tokenizer, model_id: str, todo: dict[str, str], batch_size: int) -> None:
    out = DATA / "greedy" / f"{safe(model_id)}.jsonl"
    done = {r["group"] for r in read_jsonl(out)}
    groups = [g for g in todo if g not in done]
    print(f"[greedy] {model_id}: {len(groups)} to do ({len(done)} done)", flush=True)
    config = GenerationConfidenceConfig(batch_size=batch_size)
    for start, rows in iter_generation_feature_batches(model=model, tokenizer=tokenizer,
                                                       problem_texts=[todo[g] for g in groups],
                                                       config=config, include_generated_text=True):
        append_jsonl(out, [{"group": g, "model_id": model_id, **row}
                           for g, row in zip(groups[start:start + len(rows)], rows)])


@torch.no_grad()
def hidden_last(model, tokenizer, texts: list[str]) -> np.ndarray:
    """Last-token hidden states at every layer, [len(texts), L+1, d]."""
    device = model.get_input_embeddings().weight.device
    enc = tokenizer([format_problem(tokenizer, t) for t in texts], return_tensors="pt", padding=True,
                    truncation=True, max_length=2048).to(device)
    hs = model(**enc, output_hidden_states=True).hidden_states
    return torch.stack([h[:, -1] for h in hs], dim=1).float().cpu().numpy()


@torch.no_grad()
def run_acts(model, tokenizer, model_id: str, todo: dict[str, str]) -> None:
    out_dir = DATA / "acts" / safe(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    greedy = {r["group"]: r for r in read_jsonl(DATA / "greedy" / f"{safe(model_id)}.jsonl")}
    groups = [g for g in todo if not (out_dir / f"{g}.npz").is_file()]
    print(f"[acts] {model_id}: {len(groups)} to do", flush=True)
    device = model.get_input_embeddings().weight.device
    for k, g in enumerate(groups, 1):
        start = time.time()
        prompt = format_problem(tokenizer, todo[g])
        prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=2048)["input_ids"]
        response = greedy.get(g, {}).get("generated_text", "")
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"][:4096]
        ids = torch.tensor([prompt_ids + response_ids], device=device)
        outputs = model(input_ids=ids, output_hidden_states=True)
        p = len(prompt_ids)
        hs = outputs.hidden_states
        last_prompt = torch.stack([h[0, p - 1] for h in hs]).float().cpu().numpy()
        if response_ids:
            resp_mean = torch.stack([h[0, p:].float().mean(0) for h in hs]).cpu().numpy()
            # In chunks of positions: a float log-softmax over gpt-oss's 201k vocabulary is ~5 GB per 6k tokens.
            ent, sel = [], []
            for a in range(p - 1, ids.shape[1] - 1, 1024):
                logp = torch.log_softmax(outputs.logits[0, a:min(a + 1024, ids.shape[1] - 1)].float(), dim=-1)
                ent.append(-(logp.exp() * logp).nan_to_num().sum(-1))
                sel.append(logp.gather(-1, ids[0, a + 1:a + 1 + len(logp)].unsqueeze(-1)).squeeze(-1))
                del logp
            entropy, selected = torch.cat(ent).cpu().numpy(), torch.cat(sel).cpu().numpy()
        else:
            resp_mean = np.zeros_like(last_prompt)
            entropy = selected = np.zeros(0, dtype=np.float32)
        del outputs, hs
        variants = make_perturbations(todo[g], seed=0)
        pert = hidden_last(model, tokenizer, [t for _, t in variants])
        np.savez_compressed(out_dir / f"{g}.npz", last_prompt=last_prompt.astype(np.float16),
                            resp_mean=resp_mean.astype(np.float16), pert_last=pert.astype(np.float16),
                            pert_types=np.array([kind for kind, _ in variants]),
                            entropy=entropy.astype(np.float16), selected_logp=selected.astype(np.float16))
        torch.cuda.empty_cache()
        print(f"  {k}/{len(groups)} {g}: {len(response_ids)} resp tokens, {time.time() - start:.1f}s", flush=True)


@torch.no_grad()
def run_segs(model, tokenizer, model_id: str, todo: dict[str, str], segments: int) -> None:
    out_dir = DATA / "segs" / safe(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    greedy = {r["group"]: r for r in read_jsonl(DATA / "greedy" / f"{safe(model_id)}.jsonl")}
    groups = [g for g in todo if not (out_dir / f"{g}.npz").is_file()]
    print(f"[segs] {model_id}: {len(groups)} to do", flush=True)
    device = model.get_input_embeddings().weight.device
    for k, g in enumerate(groups, 1):
        prompt = format_problem(tokenizer, todo[g])
        prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=2048)["input_ids"]
        response = greedy.get(g, {}).get("generated_text", "")
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"][:4096]
        p, n = len(prompt_ids), len(response_ids)
        if n < segments:
            print(f"  {k}/{len(groups)} {g}: only {n} resp tokens, skipped", flush=True)
            continue
        hs = model(input_ids=torch.tensor([prompt_ids + response_ids], device=device), output_hidden_states=True,
                   logits_to_keep=1).hidden_states
        bounds = np.linspace(0, n, segments + 1).round().astype(int)
        seg = torch.stack([torch.stack([h[0, p + a:p + b].float().mean(0) for a, b in zip(bounds[:-1], bounds[1:])])
                           for h in hs], 1)  # [S, L+1, d]
        np.savez_compressed(out_dir / f"{g}.npz", seg_mean=seg.cpu().numpy().astype(np.float16), n_resp=n)
        del hs
        torch.cuda.empty_cache()
        print(f"  {k}/{len(groups)} {g}: {n} resp tokens", flush=True)


def run_lens(model, tokenizer, model_id: str) -> None:
    out = DATA / "len" / f"{safe(model_id)}.npz"
    paths = [DATA / d / "samples" / f"{safe(model_id)}__default.jsonl" for d in ("len", "sep2")]
    path = next((p for p in paths if p.is_file()), None)
    if out.is_file() or path is None:
        print(f"[lens] {model_id}: {'done' if out.is_file() else 'no samples'}", flush=True)
        return
    problems = {r["problem_id"]: r["problem"] for r in read_jsonl(ROOT / "data" / "labelgen" / "problems.jsonl")}
    rows = [r for r in read_jsonl(path) if r["type"] == "original" and r["problem_id"] in problems]
    print(f"[lens] {model_id}: {len(rows)} originals from {path.parent.parent.name}", flush=True)
    # one prompt at a time: unpadded, like the cases' acts last_prompt the probe is applied to
    h = np.stack([hidden_last(model, tokenizer, [problems[r["problem_id"]]])[0].astype(np.float16) for r in rows])
    np.savez(out, h=h, problem=np.array([problems[r["problem_id"]] for r in rows]),
             log_len=np.array([np.log1p(np.array(r["n_tokens"], float)).mean() for r in rows]))
    print(f"[lens] {model_id}: wrote {out.name}", flush=True)


INTROSPECT = {
    "robust": "Suppose you were asked to solve the following problem, and separately a slightly reworded version "
              "of it (renamed variables, a typo, an irrelevant extra sentence, or a paraphrase). Would your final "
              "answers to the two versions be the same?",
    "correct": "Suppose you were asked to solve the following problem. Would your final answer be correct?",
    "long": "Suppose you were asked to solve the following problem. Would your solution take more than 4,000 tokens?",
}


def no_think_prefix(prompt: str, tokenizer) -> str:
    """Close the thinking block so the next token is the answer itself (no chain of thought)."""
    if prompt.rstrip().endswith("<think>"):
        return prompt.rstrip() + "\n\n</think>\n\n"
    return prompt + "<think>\n\n</think>\n\n" if "<think>" in tokenizer.get_vocab() else prompt


@torch.no_grad()
def run_introspect(model, tokenizer, model_id: str, todo: dict[str, str]) -> None:
    out = DATA / "introspect" / f"{safe(model_id)}.jsonl"
    done = {r["group"] for r in read_jsonl(out)}
    groups = [g for g in todo if g not in done]
    print(f"[introspect] {model_id}: {len(groups)} to do", flush=True)
    device = model.get_input_embeddings().weight.device
    vocab = tokenizer.get_vocab()
    ids = {w: [vocab[t] for t in {w, "Ġ" + w, "▁" + w} if t in vocab] for w in ("Yes", "No")}
    for g in groups:
        row = {"group": g, "model_id": model_id}
        for name, question in INTROSPECT.items():
            text = f"{question} Respond with only Yes or No.\n\nProblem:\n{todo[g]}"
            prompt = no_think_prefix(format_problem(tokenizer, text), tokenizer)
            enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
            logp = torch.log_softmax(model(**enc, logits_to_keep=1).logits[0, -1].float(), -1)
            yes, no = (torch.logsumexp(logp[ids[w]], 0).item() for w in ("Yes", "No"))
            row[name] = yes - no
            row[name + "_mass"] = float(np.exp(yes) + np.exp(no))  # how much of the next token is Yes/No at all
        append_jsonl(out, [row])
    print(f"[introspect] {model_id}: done", flush=True)


def answer_entropy(answers: list) -> float:
    counts = np.unique(np.array(["NA" if a is None else str(a) for a in answers]), return_counts=True)[1]
    q = counts / counts.sum()
    return float(-(q * np.log(q)).sum())


def run_sep(model, tokenizer, model_id: str, batch: int = 1) -> None:
    # batch 1: unpadded, like the cvbench cases' acts last_prompt the probe is applied to
    path = ROOT / "data" / "labelgen" / "samples" / f"{safe(model_id)}__default.jsonl"
    out = DATA / "sep" / f"{safe(model_id)}.npz"
    if not path.is_file() or out.is_file():
        print(f"[sep] {model_id}: {'done' if out.is_file() else 'no labelgen samples'}", flush=True)
        return
    problems = {r["problem_id"]: r["problem"] for r in read_jsonl(ROOT / "data" / "labelgen" / "problems.jsonl")}
    variants = {(r["problem_id"], r["type"], int(r["variant"])): r["text"]
                for r in read_jsonl(ROOT / "data" / "labelgen" / "variants.jsonl")}
    rows = []
    for r in read_jsonl(path):
        text = problems.get(r["problem_id"]) if r["type"] == "original" else variants.get(
            (r["problem_id"], r["type"], int(r["variant"])))
        if text and r.get("answers"):
            rows.append((problems[r["problem_id"]], r["type"], text, answer_entropy(r["answers"])))
    print(f"[sep] {model_id}: {len(rows)} prompts", flush=True)
    tokenizer.padding_side = "left"
    hs = [hidden_last(model, tokenizer, [t for _, _, t, _ in rows[i:i + batch]]) for i in range(0, len(rows), batch)]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, h=np.concatenate(hs).astype(np.float16), entropy=np.array([e for *_, e in rows]),
                        problem=np.array([p for p, *_ in rows]), type=np.array([k for _, k, _, _ in rows]))


@torch.no_grad()
def tbg_slt(model, tokenizer, text: str, response: str) -> tuple[np.ndarray, np.ndarray, int]:
    """One pass over prompt + response: hidden states at the last prompt token (TBG) and the last
    response token (SLT) at every layer, [L+1, d] each, and the response length."""
    device = model.get_input_embeddings().weight.device
    prompt_ids = tokenizer(format_problem(tokenizer, text), add_special_tokens=False, truncation=True,
                           max_length=2048)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"][:4096]
    hs = model(input_ids=torch.tensor([prompt_ids + response_ids], device=device), output_hidden_states=True,
               logits_to_keep=1).hidden_states
    p = len(prompt_ids)
    tbg = torch.stack([h[0, p - 1] for h in hs]).float().cpu().numpy()
    slt = torch.stack([h[0, -1] for h in hs]).float().cpu().numpy()
    del hs
    torch.cuda.empty_cache()
    return tbg, slt, len(response_ids)


def run_sep2(model, tokenizer, model_id: str, todo: dict[str, str]) -> None:
    out = DATA / "sep2" / f"{safe(model_id)}.npz"
    samples = DATA / "sep2" / "samples" / f"{safe(model_id)}__default.jsonl"
    greedy_path = DATA / "sep2" / "greedy" / f"{safe(model_id)}__default.traces.jsonl"
    if out.is_file() or not samples.is_file() or not greedy_path.is_file():
        print(f"[sep2] {model_id}: {'done' if out.is_file() else 'no sep2 samples or greedy'}", flush=True)
        return
    problems = {r["problem_id"]: r["problem"] for r in read_jsonl(ROOT / "data" / "labelgen" / "problems.jsonl")}
    variants = {(r["problem_id"], r["type"], int(r["variant"])): r["text"]
                for r in read_jsonl(ROOT / "data" / "labelgen" / "variants.jsonl")}
    greedy = {(r["problem_id"], r["type"], int(r["variant"])): r["texts"][0] for r in read_jsonl(greedy_path)}
    rows = []
    for r in read_jsonl(samples):
        key = (r["problem_id"], r["type"], int(r["variant"]))
        text = problems.get(r["problem_id"]) if r["type"] == "original" else variants.get(key)
        if text and key in greedy:
            rows.append((problems[r["problem_id"]], r["type"], text, greedy[key], r["answers"]))
    print(f"[sep2] {model_id}: {len(rows)} prompts", flush=True)
    tbg, slt, n_resp = [], [], []
    for k, (_, _, text, response, _) in enumerate(rows, 1):
        t, s, n = tbg_slt(model, tokenizer, text, response)
        tbg.append(t.astype(np.float16)), slt.append(s.astype(np.float16)), n_resp.append(n)
        if k % 100 == 0:
            print(f"  {k}/{len(rows)}", flush=True)
    # SLT of the cvbench cases (their TBG is acts' last_prompt), from the cached greedy responses.
    cached = {r["group"]: r.get("generated_text", "") for r in read_jsonl(DATA / "greedy" / f"{safe(model_id)}.jsonl")}
    groups = list(todo)
    case_slt = np.stack([tbg_slt(model, tokenizer, todo[g], cached.get(g, ""))[1].astype(np.float16) for g in groups])
    np.savez(DATA / "sep2" / f"{safe(model_id)}__cases.npz", group=np.array(groups), h_slt=case_slt)
    np.savez(out, h_tbg=np.stack(tbg), h_slt=np.stack(slt), n_resp=np.array(n_resp),
             answers=np.array([json.dumps(a) for *_, a in rows]), problem=np.array([p for p, *_ in rows]),
             type=np.array([k for _, k, *_ in rows]))
    print(f"[sep2] {model_id}: wrote {out.name} and {len(groups)} case SLTs", flush=True)


@torch.no_grad()
def pooled(model, tokenizer, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per text: mean over the problem-text tokens and mean over the last 4 template tokens, [n, L+1, d] each."""
    device = model.get_input_embeddings().weight.device
    prob, tmpl = [], []
    for text in texts:
        prompt = format_problem(tokenizer, text)
        start = prompt.find(text)
        enc = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=2048,
                        return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0]
        span = ((offsets[:, 0] >= start) & (offsets[:, 1] <= start + len(text)) & (offsets[:, 1] > offsets[:, 0]))
        if start < 0 or not span.any():
            span = torch.ones(len(offsets), dtype=torch.bool)
        hs = model(**enc.to(device), output_hidden_states=True).hidden_states
        span = span.to(device)
        prob.append(torch.stack([h[0, span].float().mean(0) for h in hs]).cpu().numpy())
        tmpl.append(torch.stack([h[0, -4:].float().mean(0) for h in hs]).cpu().numpy())
    return np.stack(prob), np.stack(tmpl)


def run_pool(model, tokenizer, model_id: str, todo: dict[str, str]) -> None:
    out_dir = DATA / "pool" / safe(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = [g for g in todo if not (out_dir / f"{g}.npz").is_file()]
    print(f"[pool] {model_id}: {len(groups)} to do", flush=True)
    for k, g in enumerate(groups, 1):
        variants = make_perturbations(todo[g], seed=0)
        prob, tmpl = pooled(model, tokenizer, [todo[g]] + [t for _, t in variants])
        np.savez_compressed(out_dir / f"{g}.npz", prob_mean=prob[0].astype(np.float16), tmpl_last=tmpl[0].astype(np.float16),
                            pert_prob_mean=prob[1:].astype(np.float16), pert_tmpl_last=tmpl[1:].astype(np.float16),
                            pert_types=np.array([kind for kind, _ in variants]))
        print(f"  {k}/{len(groups)} {g}", flush=True)


def run_sample(model, tokenizer, model_id: str, todo: dict[str, str], n: int, budget: int) -> None:
    out = DATA / "sample" / f"{safe(model_id)}__n{n}_b{budget}.jsonl"
    done = {r["group"] for r in read_jsonl(out)}
    groups = [g for g in todo if g not in done]
    print(f"[sample] {model_id}: {len(groups)} to do ({len(done)} done)", flush=True)
    for k, g in enumerate(groups, 1):
        start = time.time()
        features = probe_problem(model, tokenizer, model_id, "default", todo[g], n=n, budget=budget, seed=0)
        features.pop("x_raw", None)
        append_jsonl(out, [{"group": g, "model_id": model_id, "n": n, "budget": budget,
                            "seconds": round(time.time() - start, 1), **features}])
        print(f"  {k}/{len(groups)} {g}: ref {features['reference']} agree {features.get('agree_mean')} "
              f"{time.time() - start:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", default=["greedy", "acts"], choices=["greedy", "acts", "sample", "pool", "segs", "sep", "sep2", "lens", "introspect"])
    parser.add_argument("--models", nargs="+", default=None, help="Default: every model in the case file.")
    parser.add_argument("--cases", default="cases.jsonl", help="Case file under data/cvbench.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--segments", type=int, default=10)
    args = parser.parse_args()

    cases = read_jsonl(DATA / args.cases)
    models = args.models or list(dict.fromkeys(c["model_id"] for c in cases))
    for base in dict.fromkeys(m.partition(":")[0] for m in models):
        todos = {m: {c["group"]: c["problem"] for c in cases if c["model_id"] == m}
                 for m in models if m.partition(":")[0] == base}
        if not any(todos.values()):
            continue
        model, tokenizer = load_model_and_tokenizer(base, local_files_only=True)
        print(f"[load] {base}: devices {sorted(set(map(str, getattr(model, 'hf_device_map', {}).values())))}", flush=True)
        template = tokenizer.apply_chat_template
        try:
            for model_id, todo in todos.items():
                effort = model_id.partition(":")[2]
                # gpt-oss reads the effort from its chat template, which every stage formats through.
                tokenizer.apply_chat_template = functools.partial(template, reasoning_effort=effort) if effort else template
                for stage in args.stages:
                    if stage == "greedy":
                        run_greedy(model, tokenizer, model_id, todo, args.batch_size)
                    elif stage == "acts":
                        run_acts(model, tokenizer, model_id, todo)
                    elif stage == "pool":
                        run_pool(model, tokenizer, model_id, todo)
                    elif stage == "sep":
                        run_sep(model, tokenizer, model_id)
                    elif stage == "sep2":
                        run_sep2(model, tokenizer, model_id, todo)
                    elif stage == "introspect":
                        run_introspect(model, tokenizer, model_id, todo)
                    elif stage == "lens":
                        run_lens(model, tokenizer, model_id)
                    elif stage == "segs":
                        run_segs(model, tokenizer, model_id, todo, args.segments)
                    else:
                        run_sample(model, tokenizer, model_id, todo, args.n, args.budget)
        finally:
            del model, tokenizer
            release_model()


if __name__ == "__main__":
    main()
