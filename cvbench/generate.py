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
* ``tsegs``: ``segs`` over the transcript probe's 8 sampled traces per case (T=1, up to 16k tokens:
  ``data/anchors/**/screen.jsonl``, ``data/transcript_probe/oss/screen_<effort>.jsonl``) instead of the
  4k greedy one, cut at each of ``--tsegs-budgets`` -> ``tsegs/<model>/<group>.npz``
  (``seg_mean_<B>`` [samples, S, L+1, d]), so seg_probe and the transcript features can be scored on
  the same trace.
* ``lookback``: Lookback Lens attention features over the greedy response (prompt vs response attention
  per layer and head, averaged within ``--segments`` segments) -> ``lookback/<model>/<group>.npz``.
* ``noise``: latent-noise sensitivity: log p of the model's own budget-forced answer after its greedy trace,
  clean and with relative Gaussian noise on the mid-layer residual stream over the CoT -> ``noise/<model>/<group>.npz``.
* ``hedge``: one pass per case over the greedy trace for two methods. (A) hedge-localised states: at every
  layer, the mean hidden state at hedge tokens ("wait", "hmm", "actually", "let me double-check", ...), over
  the ``HEDGE_PRE`` tokens before each hedge and the ``HEDGE_POST`` tokens from it on. (B) early-answer
  commitment: the model's own budget-forced answer A after the full trace, then log p(A) when the trace is
  cut at each of ``--segments`` boundaries and closed with the same forcing suffix -> ``hedge/<model>/<group>.npz``.
* ``effort``: training data for the effort probe (``cvbench/effort.py``). For every sep2 prompt of the model
  (originals and variants) that has sampled lengths (``sep2/samples`` or ``len/samples``) and a greedy
  response (``sep2/greedy``, 4k), one pass over prompt + response saves the mean hidden state over the
  first 1/2/3/5/9 of 10 response segments and the last prompt token, at layer fracs ``EFFORT_FRACS``,
  with the mean log sampled length -> ``effort/<model>/<pid>_<type>_<variant>.npz``.
* ``difficulty``: prompt-only passes over ``difficulty/prompts.jsonl`` (E2H-AMC problems with human
  difficulty ratings, ``cvbench/difficulty.py --build``), same sites as ``liref``, saved in chunks of 250
  prompts -> ``difficulty/<model>/chunk_NNN.npz`` (resumable). The difficulty direction is fitted offline.
* ``liref``: Linear Reasoning Feature contrast set (Hong et al., 2025): prompt-only passes over
  ``liref/contrast.jsonl`` (reasoning questions, GSM8K, vs memory questions, PopQA), saving the last prompt
  token and the mean over the question tokens at every layer -> ``liref/<model>.npz``. The direction
  mean(reasoning) - mean(memory) per layer is formed offline (``cvbench/stack.py`` liref blocks).
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
* ``nothink``: classical semantic entropy with no chain of thought: ``--n`` samples (T=1) of the answer to
  the original, thinking block closed; ``--nt-mode direct`` prefills ``\\boxed{`` (answer tokens only),
  ``free`` lets the model answer in ``--nt-max-new`` tokens -> ``nothink/<model>__<mode>.jsonl``.
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
    if "<|channel|>" in tokenizer.get_vocab():  # gpt-oss (harmony): empty analysis, then the final channel
        return prompt + "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
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


NT_PREFILL = "The final answer is $\\boxed{"


def nothink_prompt(tokenizer, problem: str, mode: str) -> str:
    """No chain of thought: think block closed; ``direct`` also prefills the boxed answer."""
    prompt = no_think_prefix(format_problem(tokenizer, problem + SOLVE_SUFFIX), tokenizer)
    return prompt + NT_PREFILL if mode == "direct" else prompt


@torch.no_grad()
def run_nothink(model, tokenizer, model_id: str, todo: dict[str, str], n: int, mode: str, max_new: int,
                chunk: int = 50) -> None:
    """Classical semantic entropy without reasoning: ``n`` samples at T=1 of the answer to the original
    problem with the thinking block closed. ``direct``: the answer is prefilled up to ``\\boxed{`` and
    only the number is decoded (~10 tokens); ``free``: the model writes a short answer (``max_new``
    tokens) and the last boxed / final integer is parsed. -> ``nothink/<model>__<mode>.jsonl``."""
    sys.path.insert(0, str(ROOT / "datagen"))
    from common import extract_answer, parse_boxed  # noqa: E402
    out = DATA / "nothink" / f"{safe(model_id)}__{mode}.jsonl"
    done = {r["group"] for r in read_jsonl(out)}
    groups = [g for g in todo if g not in done]
    print(f"[nothink:{mode}] {model_id}: {len(groups)} to do ({len(done)} done)", flush=True)
    device = model.get_input_embeddings().weight.device
    torch.manual_seed(0)
    for k, g in enumerate(groups, 1):
        start = time.time()
        prompt = nothink_prompt(tokenizer, todo[g], mode)
        enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        p = enc.input_ids.shape[1]
        # first-token distribution of the answer (direct: the leading digit)
        logits = model(**enc, logits_to_keep=1).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        tok_ent = float(-(lp.exp() * lp).sum())
        texts, n_tok = [], []
        for s in range(0, n, chunk):
            m = min(chunk, n - s)
            gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                 max_new_tokens=max_new, num_return_sequences=m,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)[:, p:]
            for row in gen.tolist():
                ids = [t for t in row if t not in (tokenizer.pad_token_id, tokenizer.eos_token_id)]
                n_tok.append(len(ids))
                texts.append(tokenizer.decode(ids, skip_special_tokens=True))
        if mode == "direct":
            answers = [parse_boxed(t.split("}", 1)[0]) if "}" in t else parse_boxed(t) for t in texts]
        else:
            answers = [extract_answer(t) for t in texts]
        given = [a for a in answers if a is not None]
        top = np.unique(np.array([str(a) for a in given]), return_counts=True)[1].max() / n if given else 0.0
        row = {"group": g, "model_id": model_id, "mode": mode, "n": n, "max_new": max_new,
               "answers": answers, "n_tokens": n_tok, "first_tok_entropy": tok_ent,
               "se": answer_entropy(answers), "top_share": float(top), "na_rate": 1 - len(given) / n,
               "n_distinct": len(set(given)), "seconds": round(time.time() - start, 2),
               "example": texts[0][:300]}
        append_jsonl(out, [row])
        print(f"  {k}/{len(groups)} {g}: se {row['se']:.2f} top {top:.2f} na {row['na_rate']:.2f} "
              f"distinct {row['n_distinct']} ex {texts[0][:40]!r} {row['seconds']}s", flush=True)
        torch.cuda.empty_cache()


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


SCREENS = {"Qwen/Qwen3.5-4B": ROOT / "data" / "anchors",
           **{m: ROOT / "data" / "anchors" / safe(m) for m in MODELS[1:]},
           **{f"openai/gpt-oss-120b:{e}": ROOT / "data" / "transcript_probe" / "oss" / f"screen_{e}.jsonl"
              for e in ("low", "medium", "high")}}
SOLVE_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."  # datagen/common.py
HARMONY_FINAL = "<|end|><|start|>assistant<|channel|>final<|message|>"


def screen_texts(model_id: str) -> dict[str, dict]:
    """The transcript probe's training samples (8 per original problem, T=1, 16k): group -> screen row."""
    path = SCREENS[model_id]
    path = path / "screen.jsonl" if path.is_dir() else path
    return {r["group"]: r for r in read_jsonl(path) if r["variant"] == "X"}


def screen_continuation(tokenizer, problem: str, text: str) -> tuple[str, str]:
    """Rebuild (prompt, continuation) as the sample was generated: the model's chat template on problem +
    SOLVE_SUFFIX (vLLM, ``solutions/anchors/resample.py``), or for gpt-oss the prompt opened into the analysis
    channel (OpenRouter, ``thought_anchors/resample.py``), whose saved text lost its harmony markers."""
    text = text.replace("Ġ", " ").replace("Ċ", "\n") if "Ġ" in text else text  # DeepSeek's byte-level pieces
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": problem + SOLVE_SUFFIX}],
                                           tokenize=False, add_generation_prompt=True)
    if "<|channel|>" in tokenizer.get_vocab():
        prompt += "<|channel|>analysis<|message|>"
        text = text.replace("assistantfinal", HARMONY_FINAL, 1)
    return prompt, text


@torch.no_grad()
def chunked_hidden_states(model, ids: list[int], device, chunk: int = 1024) -> tuple:
    """``model(..., output_hidden_states=True).hidden_states`` computed ``chunk`` tokens at a time over a KV
    cache: eager attention (gpt-oss) otherwise builds one [heads, seq, seq] score tensor, which does not fit
    next to the 120B weights at 16k tokens."""
    from transformers import DynamicCache
    cache = DynamicCache(config=model.config)
    parts = []
    for start in range(0, len(ids), chunk):
        piece = torch.tensor([ids[start:start + chunk]], device=device)
        pos = torch.arange(start, start + piece.shape[1], device=device)
        out = model(input_ids=piece, past_key_values=cache, use_cache=True, cache_position=pos,
                    position_ids=pos[None], output_hidden_states=True, logits_to_keep=1)
        parts.append(out.hidden_states)
        del out
    return tuple(torch.cat([p[layer] for p in parts], 1) for layer in range(len(parts[0])))


@torch.no_grad()
def run_tsegs(model, tokenizer, model_id: str, todo: dict[str, str], segments: int, budgets: list[int]) -> None:
    """``segs`` over the transcript probe's sampled 16k traces instead of the 4k greedy one: per sample and
    budget B, the per-layer mean over each of ``segments`` equal token segments of the first B response tokens
    (one forward pass at the largest budget; causal, so shorter cuts reuse its prefix)."""
    out_dir = DATA / "tsegs" / safe(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    screens = screen_texts(model_id)
    groups = [g for g in todo if g in screens and not (out_dir / f"{g}.npz").is_file()]
    print(f"[tsegs] {model_id}: {len(groups)} to do ({sum(g not in screens for g in todo)} without samples)",
          flush=True)
    device = model.get_input_embeddings().weight.device
    for k, g in enumerate(groups, 1):
        start = time.time()
        segs = {b: [] for b in budgets}
        n_resp = []
        for text in screens[g]["texts"]:
            prompt, text = screen_continuation(tokenizer, todo[g], text)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            response_ids = tokenizer(text, add_special_tokens=False)["input_ids"][:max(budgets)]
            p, n = len(prompt_ids), len(response_ids)
            n_resp.append(n)
            if n < segments:  # too short to segment: NaN, the loader falls back
                for b in budgets:
                    segs[b].append(None)
                continue
            hs = chunked_hidden_states(model, prompt_ids + response_ids, device)
            for b in budgets:
                bounds = np.linspace(0, min(n, b), segments + 1).round().astype(int)
                segs[b].append(torch.stack([torch.stack([h[0, p + a:p + c].float().mean(0)
                                                         for a, c in zip(bounds[:-1], bounds[1:])])
                                            for h in hs], 1).cpu().numpy().astype(np.float16))  # [S, L+1, d]
            del hs
            torch.cuda.empty_cache()
        shape = next((s.shape for s in segs[budgets[0]] if s is not None), None)
        if shape is None:
            print(f"  {k}/{len(groups)} {g}: no sample long enough, skipped", flush=True)
            continue
        fill = lambda rows: np.stack([r if r is not None else np.full(shape, np.nan, np.float16) for r in rows])
        np.savez_compressed(out_dir / f"{g}.npz", n_resp=np.array(n_resp),
                            finish=np.array(screens[g]["finish"]), budgets=np.array(budgets),
                            **{f"seg_mean_{b}": fill(segs[b]) for b in budgets})  # [samples, S, L+1, d]
        print(f"  {k}/{len(groups)} {g}: {len(n_resp)} samples, max {max(n_resp)} tokens, "
              f"{time.time() - start:.0f}s", flush=True)


@torch.no_grad()
def run_lookback(model, tokenizer, model_id: str, todo: dict[str, str], segments: int) -> None:
    """Lookback Lens features (Chuang et al., 2024) over the greedy response, per attention layer and head,
    averaged within each of ``segments`` equal token segments. Per response token: mean attention on the
    prompt (position 0, the attention sink, excluded) over mean attention on the response so far
    (``lb_ratio``); total attention on the prompt (``ctx_mass``), on the sink (``sink``) and on prompt tokens
    that contain a digit (``num_mass``). Eager attention, ``output_attentions`` in chunks over a KV cache.
    Layers without attention weights (Qwen3.5 linear attention) are dropped; ``layers`` keeps the indices."""
    out_dir = DATA / "lookback" / safe(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    greedy = {r["group"]: r for r in read_jsonl(DATA / "greedy" / f"{safe(model_id)}.jsonl")}
    groups = [g for g in todo if not (out_dir / f"{g}.npz").is_file()]
    print(f"[lookback] {model_id}: {len(groups)} to do", flush=True)
    from transformers import DynamicCache
    device = model.get_input_embeddings().weight.device
    previous = model.config._attn_implementation
    model.set_attn_implementation("eager")
    text_config = getattr(model.config, "text_config", model.config)
    heads, types = text_config.num_attention_heads, list(getattr(text_config, "layer_types", None) or [])
    attn_layers = [i for i, t in enumerate(types) if "linear" not in t] if types else None
    # A config-built cache keeps only the last `window` keys on sliding layers, so key index != position;
    # a plain cache keeps every key (the sliding mask still comes from the config). Linear-attention models
    # (Qwen3.5) need the config-built cache for their recurrent state and have no sliding layers.
    sliding = any("sliding" in t for t in types)
    chunk = 128 if heads >= 64 else 256
    try:
        for k, g in enumerate(groups, 1):
            start = time.time()
            prompt = format_problem(tokenizer, todo[g])
            prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=2048)["input_ids"]
            response = greedy.get(g, {}).get("generated_text", "")
            response_ids = tokenizer(response, add_special_tokens=False)["input_ids"][:4096]
            p, n = len(prompt_ids), len(response_ids)
            if n < segments:
                print(f"  {k}/{len(groups)} {g}: only {n} resp tokens, skipped", flush=True)
                continue
            ids = prompt_ids + response_ids
            digit = torch.tensor([any(ch.isdigit() for ch in tokenizer.decode([t])) for t in prompt_ids], device=device)
            digit[0] = False
            seg_of = torch.as_tensor(np.minimum(np.arange(n) * segments // n, segments - 1), device=device)
            cache = DynamicCache() if sliding else DynamicCache(config=model.config)
            sums, layers = None, None
            for a in range(0, len(ids), chunk):
                piece = torch.tensor([ids[a:a + chunk]], device=device)
                pos = torch.arange(a, a + piece.shape[1], device=device)
                out = model(input_ids=piece, past_key_values=cache, use_cache=True, cache_position=pos,
                            position_ids=pos[None], output_attentions=True, logits_to_keep=1)
                keep = [(i, att) for i, att in enumerate(out.attentions) if att is not None]
                if layers is None:  # attentions either span every layer (None where absent) or attention layers only
                    by_index = len(out.attentions) == text_config.num_hidden_layers or attn_layers is None
                    layers = [i if by_index else attn_layers[i] for i, _ in keep]
                    sums = torch.zeros(4, segments, len(layers), heads, device=device, dtype=torch.float64)
                q = pos >= p  # response queries in this chunk
                if q.any():
                    qpos = pos[q]
                    n_new = (qpos - p + 1).double()  # response tokens visible to each query, itself included
                    seg = seg_of[qpos - p]
                    for j, (_, att) in enumerate(keep):
                        w = att[0][:, q].double()  # [H, q, keys]
                        ctx = w[..., 1:p].sum(-1)
                        new = w[..., p:].sum(-1)
                        ctx_mean, new_mean = ctx / max(p - 1, 1), new / n_new
                        feats = torch.stack([ctx_mean / (ctx_mean + new_mean + 1e-12), ctx, w[..., 0],
                                             (w[..., :p] * digit).sum(-1)])  # [4, H, q]
                        sums[:, :, j].index_add_(1, seg, feats.permute(0, 2, 1))
                del out, keep
            counts = torch.bincount(seg_of, minlength=segments).double()[None, :, None, None]
            f = (sums / counts).float().cpu().numpy().astype(np.float16)
            np.savez_compressed(out_dir / f"{g}.npz", lb_ratio=f[0], ctx_mass=f[1], sink=f[2], num_mass=f[3],
                                layers=np.array(layers), n_resp=n, n_prompt=p)  # each [S, L_attn, H]
            del cache
            torch.cuda.empty_cache()
            print(f"  {k}/{len(groups)} {g}: {n} resp tokens, {len(layers)} attn layers, {time.time() - start:.0f}s",
                  flush=True)
    finally:
        model.set_attn_implementation(previous)


NOISE_SIGMAS = (0.03, 0.1, 0.3, 1.0)  # noise norm / the token's own residual norm
NOISE_REGIONS = ("half", "all")       # CoT tokens [0, n/2) or [0, n)
NOISE_DRAWS = 2
NOISE_FRAC = 0.5                      # layer depth, as the probes
FORCE_THINK = "\n</think>\n\nThe final answer is:\n\\[\n\\boxed{"  # cvbench/force_answers.py
FORCE_HARMONY = "<|end|><|start|>assistant<|channel|>final<|message|>The final answer is \\(\\boxed{"


def decoder_layers(model):
    n = getattr(model.config, "text_config", model.config).num_hidden_layers
    return next(m for name, m in model.named_modules()
                if isinstance(m, torch.nn.ModuleList) and len(m) == n and name.endswith("layers"))


def forced_answer_text(model, response: str) -> tuple[str, str, bool]:
    """(reasoning cut at its end-of-thinking marker, force suffix, whether a marker was found)."""
    harmony = model.config.model_type == "gpt_oss"
    marker = "assistantfinal" if harmony else "</think>"  # gpt-oss: harmony headers decoded without specials
    cut = response.find(marker)
    return (response if cut < 0 else response[:cut]), (FORCE_HARMONY if harmony else FORCE_THINK), cut >= 0


@torch.no_grad()
def answer_logprobs(model, ids: list[int], n_answer: int, device, hook_state: dict | None = None,
                    chunk: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    """log p of the last ``n_answer`` tokens of ``ids`` and the argmax at those positions, chunked over a KV
    cache; the last chunk holds at least the suffix + answer so their logits come from one call."""
    from transformers import DynamicCache
    cache = DynamicCache(config=model.config)
    tail = max(n_answer + 64, 1)
    starts = list(range(0, len(ids) - tail, chunk)) + [max(len(ids) - tail, 0)]
    for i, a in enumerate(starts):
        b = starts[i + 1] if i + 1 < len(starts) else len(ids)
        if a >= b:
            continue
        pos = torch.arange(a, b, device=device)
        if hook_state is not None:
            hook_state["offset"] = a
        out = model(input_ids=torch.tensor([ids[a:b]], device=device), past_key_values=cache, use_cache=True,
                    cache_position=pos, position_ids=pos[None], logits_to_keep=n_answer + 1 if b == len(ids) else 1)
    logits = out.logits[0, :-1].float()  # positions predicting the answer tokens
    logp = torch.log_softmax(logits, -1).gather(-1, torch.tensor(ids[-n_answer:], device=device)[:, None])[:, 0]
    return logp.cpu(), logits.argmax(-1).cpu()


@torch.no_grad()
def run_noise(model, tokenizer, model_id: str, todo: dict[str, str]) -> None:
    """Latent-noise sensitivity of the model's own answer. The greedy trace (cut at its end-of-thinking marker)
    is closed with a budget-forcing suffix and the answer A decoded greedily up to ``}``; then prompt + trace
    + suffix + A is teacher-forced once clean and once per (region, sigma, draw) with noise added to the
    residual stream after layer ``round(NOISE_FRAC * L)`` on the CoT tokens of the region: a random unit
    direction per token (same draws for every sigma) times sigma times that token's residual norm.
    Saves log p(A) per token clean, mean log p(A) noisy, and whether the argmax over A changed."""
    out_dir = DATA / "noise" / safe(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    greedy = {r["group"]: r for r in read_jsonl(DATA / "greedy" / f"{safe(model_id)}.jsonl")}
    groups = [g for g in todo if not (out_dir / f"{g}.npz").is_file()]
    print(f"[noise] {model_id}: {len(groups)} to do", flush=True)
    device = model.get_input_embeddings().weight.device
    layers = decoder_layers(model)
    layer = int(round(NOISE_FRAC * len(layers)))
    state = {"offset": 0, "noise": None, "lo": 0, "hi": 0, "sigma": 0.0}

    def hook(module, args, output):
        if state["noise"] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output  # [1, T, d]
        pos = state["offset"] + torch.arange(h.shape[1], device=h.device)
        m = (pos >= state["lo"]) & (pos < state["hi"])
        if m.any():
            hm = h[0, m]
            h = h.clone()
            h[0, m] = hm + state["sigma"] * hm.float().norm(dim=-1, keepdim=True).to(hm.dtype) * state["noise"][pos[m]]
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return output

    handle = layers[layer - 1].register_forward_hook(hook)  # output of block `layer` = hidden_states[layer]
    try:
        for k, g in enumerate(groups, 1):
            start = time.time()
            prompt = format_problem(tokenizer, todo[g])
            prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=2048)["input_ids"]
            reasoning, suffix, had_marker = forced_answer_text(model, greedy.get(g, {}).get("generated_text", ""))
            cot_ids = tokenizer(reasoning, add_special_tokens=False)["input_ids"][:4096]
            p, n = len(prompt_ids), len(cot_ids)
            if n < 10:
                print(f"  {k}/{len(groups)} {g}: only {n} CoT tokens, skipped", flush=True)
                continue
            head = prompt_ids + cot_ids + tokenizer(suffix, add_special_tokens=False)["input_ids"]
            enc = torch.tensor([head], device=device)
            gen = model.generate(input_ids=enc, attention_mask=torch.ones_like(enc), max_new_tokens=16,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)[0, len(head):].tolist()
            answer = []
            for t in gen:
                answer.append(t)
                if "}" in tokenizer.decode([t]):
                    break
            ids = head + answer
            lp_clean, am_clean = answer_logprobs(model, ids, len(answer), device)
            d = getattr(model.config, "text_config", model.config).hidden_size
            noisy = np.zeros((len(NOISE_REGIONS), len(NOISE_SIGMAS), NOISE_DRAWS), np.float32)
            flip = np.zeros_like(noisy, dtype=bool)
            for di in range(NOISE_DRAWS):
                gen_ = torch.Generator(device=device).manual_seed(1000 + di)
                u = torch.randn(len(ids), d, device=device, generator=gen_)
                state["noise"] = (u / u.norm(dim=-1, keepdim=True)).to(model.dtype)
                for ri, region in enumerate(NOISE_REGIONS):
                    state["lo"], state["hi"] = p, p + (n // 2 if region == "half" else n)
                    for si, sigma in enumerate(NOISE_SIGMAS):
                        state["sigma"] = sigma
                        lp, am = answer_logprobs(model, ids, len(answer), device, state)
                        noisy[ri, si, di] = lp.mean().item()
                        flip[ri, si, di] = bool((am != am_clean).any())
                state["noise"] = None
            np.savez_compressed(out_dir / f"{g}.npz", logp_clean=lp_clean.numpy(), logp_noisy=noisy, flip=flip,
                                sigmas=np.array(NOISE_SIGMAS), regions=np.array(NOISE_REGIONS), layer=layer,
                                answer=tokenizer.decode(answer), had_marker=had_marker, n_cot=n, n_prompt=p)
            torch.cuda.empty_cache()
            print(f"  {k}/{len(groups)} {g}: {n} CoT tokens, answer {tokenizer.decode(answer)!r}, "
                  f"logp {lp_clean.mean():.2f} -> {np.round(noisy[0].mean(-1), 2).tolist()} (half), "
                  f"{time.time() - start:.0f}s", flush=True)
    finally:
        handle.remove()
        state["noise"] = None


@torch.no_grad()
def run_liref(model, tokenizer, model_id: str) -> None:
    out = DATA / "liref" / f"{safe(model_id)}.npz"
    rows = read_jsonl(DATA / "liref" / "contrast.jsonl")
    if out.is_file() or not rows:
        print(f"[liref] {model_id}: {'done' if out.is_file() else 'no contrast.jsonl'}", flush=True)
        return
    print(f"[liref] {model_id}: {len(rows)} contrast prompts", flush=True)
    prob, tmpl = pooled(model, tokenizer, [r["text"] for r in rows])  # [n, L+1, d]: question mean, last 4 template
    device = model.get_input_embeddings().weight.device
    last = []
    for r in rows:
        ids = tokenizer(format_problem(tokenizer, r["text"]), add_special_tokens=False, truncation=True,
                        max_length=2048, return_tensors="pt")["input_ids"].to(device)
        hs = model(input_ids=ids, output_hidden_states=True, logits_to_keep=1).hidden_states
        last.append(torch.stack([h[0, -1] for h in hs]).float().cpu().numpy().astype(np.float16))
        del hs
    torch.cuda.empty_cache()
    np.savez_compressed(out, h_last=np.stack(last), h_prob=prob.astype(np.float16), h_tmpl=tmpl.astype(np.float16),
                        kind=np.array([r["kind"] for r in rows]), source=np.array([r["source"] for r in rows]))
    print(f"[liref] {model_id}: wrote {out.name}", flush=True)


@torch.no_grad()
def run_difficulty(model, tokenizer, model_id: str, chunk: int = 250) -> None:
    """Prompt-site states (last prompt token, question mean, template mean; every layer) over
    difficulty/prompts.jsonl, in chunks so a 20-min slot can stop and resume."""
    rows = read_jsonl(DATA / "difficulty" / "prompts.jsonl")
    out_dir = DATA / "difficulty" / safe(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = model.get_input_embeddings().weight.device
    n_chunks = (len(rows) + chunk - 1) // chunk
    for ci in range(n_chunks):
        out = out_dir / f"chunk_{ci:03d}.npz"
        if out.is_file():
            continue
        part = rows[ci * chunk:(ci + 1) * chunk]
        prob, tmpl = pooled(model, tokenizer, [r["text"] for r in part])
        last = []
        for r in part:
            ids = tokenizer(format_problem(tokenizer, r["text"]), add_special_tokens=False, truncation=True,
                            max_length=2048, return_tensors="pt")["input_ids"].to(device)
            hs = model(input_ids=ids, output_hidden_states=True, logits_to_keep=1).hidden_states
            last.append(torch.stack([h[0, -1] for h in hs]).float().cpu().numpy().astype(np.float16))
            del hs
        torch.cuda.empty_cache()
        np.savez_compressed(out, h_last=np.stack(last), h_prob=prob.astype(np.float16), h_tmpl=tmpl.astype(np.float16),
                            rating=np.array([float(r["rating"]) for r in part]),
                            pid=np.array([r["id"] for r in part]))
        print(f"[difficulty] {model_id}: chunk {ci + 1}/{n_chunks} ({len(part)} prompts)", flush=True)
    print(f"[difficulty] {model_id}: done ({len(rows)} prompts)", flush=True)


EFFORT_FRACS = (0.3, 0.5, 0.7)
EFFORT_WINDOWS = ((0, 1), (0, 2), (0, 3), (0, 5), (0, 9))


@torch.no_grad()
def run_effort(model, tokenizer, model_id: str, segments: int = 10) -> None:
    out_dir = DATA / "effort" / safe(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    greedy_path = DATA / "sep2" / "greedy" / f"{safe(model_id)}__default.traces.jsonl"
    if not greedy_path.is_file():
        print(f"[effort] {model_id}: no sep2 greedy traces", flush=True)
        return
    problems = {r["problem_id"]: r["problem"] for r in read_jsonl(ROOT / "data" / "labelgen" / "problems.jsonl")}
    variants = {(r["problem_id"], r["type"], int(r["variant"])): r["text"]
                for r in read_jsonl(ROOT / "data" / "labelgen" / "variants.jsonl")}
    greedy = {(r["problem_id"], r["type"], int(r["variant"])): r["texts"][0] for r in read_jsonl(greedy_path)}
    target = {}
    for d in ("sep2", "len"):  # sep2: 10 samples at 8k; len: 4 samples; either gives mean log tokens
        path = DATA / d / "samples" / f"{safe(model_id)}__default.jsonl"
        for r in read_jsonl(path) if path.is_file() else []:
            key = (r["problem_id"], r["type"], int(r["variant"]))
            n = np.array(r["n_tokens"], float)
            target.setdefault(key, (float(np.log1p(n).mean()), float((n >= 8192).mean()), len(n)))
    keys = [k for k in greedy if k in target and (k[1] == "original" or k in variants)]
    todo = [k for k in keys if not (out_dir / f"{k[0]}_{k[1]}_{k[2]}.npz").is_file()]
    print(f"[effort] {model_id}: {len(keys)} prompts with greedy + sampled lengths, {len(todo)} to do", flush=True)
    device = model.get_input_embeddings().weight.device
    for k, key in enumerate(todo, 1):
        pid, kind, var = key
        text = problems[pid] if kind == "original" else variants[key]
        prompt_ids = tokenizer(format_problem(tokenizer, text), add_special_tokens=False, truncation=True,
                               max_length=2048)["input_ids"]
        response_ids = tokenizer(greedy[key], add_special_tokens=False)["input_ids"][:4096]
        p, n = len(prompt_ids), len(response_ids)
        if n < segments:
            continue
        hs = model(input_ids=torch.tensor([prompt_ids + response_ids], device=device), output_hidden_states=True,
                   logits_to_keep=1).hidden_states
        layers = [int(round(f * (len(hs) - 1))) for f in EFFORT_FRACS]
        bounds = np.linspace(0, n, segments + 1).round().astype(int)
        arrays = {"last_prompt": torch.stack([hs[l][0, p - 1] for l in layers]).float().cpu().numpy().astype(np.float16)}
        for a, b in EFFORT_WINDOWS:
            arrays[f"seg_{a}_{b}"] = torch.stack([hs[l][0, p + bounds[a]:p + bounds[b]].float().mean(0)
                                                  for l in layers]).cpu().numpy().astype(np.float16)  # [fracs, d]
        del hs
        log_len, cap_rate, n_samples = target[key]
        np.savez(out_dir / f"{pid}_{kind}_{var}.npz", **arrays, fracs=np.array(EFFORT_FRACS), log_len=log_len,
                 cap_rate=cap_rate, n_samples=n_samples, n_resp=n, problem_id=pid, type=kind, variant=var,
                 problem=problems[pid])
        if k % 100 == 0:
            torch.cuda.empty_cache()
            print(f"  {k}/{len(todo)}", flush=True)
    print(f"[effort] {model_id}: done", flush=True)


@torch.no_grad()
def run_pcprompt(model, tokenizer, model_id: str) -> None:
    """Competence probe (pcorrect.py): prompt-only forward on every labelgen original; last-prompt state at
    EFFORT_FRACS (+ every layer) -> effort/<model>/<pid>_original_0.npz, the file pcorrect.table() reads."""
    out_dir = DATA / "effort" / safe(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = {r["problem_id"]: r["problem"] for r in read_jsonl(ROOT / "data" / "labelgen" / "problems.jsonl")}
    todo = [pid for pid in problems if not (out_dir / f"{pid}_original_0.npz").is_file()]
    print(f"[pcprompt] {model_id}: {len(todo)} of {len(problems)} to do", flush=True)
    device = model.get_input_embeddings().weight.device
    for k, pid in enumerate(todo, 1):
        ids = tokenizer(format_problem(tokenizer, problems[pid]), add_special_tokens=False, truncation=True,
                        max_length=2048)["input_ids"]
        hs = model(input_ids=torch.tensor([ids], device=device), output_hidden_states=True,
                   logits_to_keep=1).hidden_states
        allh = torch.stack([h[0, -1] for h in hs]).float().cpu().numpy().astype(np.float16)
        layers = [int(round(f * (len(hs) - 1))) for f in EFFORT_FRACS]
        np.savez(out_dir / f"{pid}_original_0.npz", last_prompt=allh[layers], last_prompt_all=allh,
                 fracs=np.array(EFFORT_FRACS), problem_id=pid, type="original", variant=0, problem=problems[pid])
        del hs
        if k % 50 == 0:
            print(f"  {k}/{len(todo)}", flush=True)
    print(f"[pcprompt] {model_id}: done", flush=True)


HEDGE = re.compile(r"\b(wait|hmm+|alternatively|actually|let me (?:double[- ]check|re-?check|verify|reconsider)|"
                   r"but (?:wait|hold on)|on second thought|i made (?:a|an) (?:mistake|error)|that'?s (?:not right|wrong))\b",
                   re.IGNORECASE)  # cvbench/cv.py HESITATION
HEDGE_PRE, HEDGE_POST = 16, 32  # tokens before a hedge (what triggered it) / from it on (the reconsideration)


@torch.no_grad()
def run_hedge(model, tokenizer, model_id: str, todo: dict[str, str], segments: int) -> None:
    out_dir = DATA / "hedge" / safe(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    greedy = {r["group"]: r for r in read_jsonl(DATA / "greedy" / f"{safe(model_id)}.jsonl")}
    groups = [g for g in todo if not (out_dir / f"{g}.npz").is_file()]
    print(f"[hedge] {model_id}: {len(groups)} to do", flush=True)
    device = model.get_input_embeddings().weight.device
    for k, g in enumerate(groups, 1):
        start = time.time()
        prompt = format_problem(tokenizer, todo[g])
        prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=2048)["input_ids"]
        response = greedy.get(g, {}).get("generated_text", "")
        enc = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
        response_ids, offsets = enc["input_ids"][:4096], enc["offset_mapping"][:4096]
        p, n = len(prompt_ids), len(response_ids)
        if n < segments:
            print(f"  {k}/{len(groups)} {g}: only {n} resp tokens, skipped", flush=True)
            continue
        # (A) hedge-localised states
        ends = np.array([e for _, e in offsets])
        hedge_pos = sorted({int(np.searchsorted(ends, m.start(), side="right")) for m in HEDGE.finditer(response)})
        hedge_pos = [t for t in hedge_pos if t < n]
        hs = chunked_hidden_states(model, prompt_ids + response_ids, device)
        feats = {}
        if hedge_pos:
            idx = {"at": hedge_pos,
                   "pre": sorted({t for h in hedge_pos for t in range(max(h - HEDGE_PRE, 0), h)} - set(hedge_pos)),
                   "post": sorted({t for h in hedge_pos for t in range(h, min(h + HEDGE_POST, n))})}
            for name, pos in idx.items():
                if not pos:
                    pos = hedge_pos
                at = torch.tensor([p + t for t in pos], device=hs[0].device)
                feats[f"hedge_{name}"] = torch.stack([h[0, at].float().mean(0) for h in hs]).cpu().numpy().astype(np.float16)
        del hs
        torch.cuda.empty_cache()
        # (B) early-answer commitment curve
        reasoning, suffix, had_marker = forced_answer_text(model, response)
        cot_ids = tokenizer(reasoning, add_special_tokens=False)["input_ids"][:4096]
        suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
        head = prompt_ids + cot_ids + suffix_ids
        x = torch.tensor([head], device=device)
        gen = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=16, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)[0, len(head):].tolist()
        answer = []
        for t in gen:
            answer.append(t)
            if "}" in tokenizer.decode([t]):
                break
        bounds = np.linspace(0, len(cot_ids), segments + 1).round().astype(int)[1:]
        logp, match = [], []
        full_am = None
        for b in bounds[::-1]:  # full trace first: its argmax is the reference
            lp, am = answer_logprobs(model, prompt_ids + cot_ids[:b] + suffix_ids + answer, len(answer), device)
            full_am = am if full_am is None else full_am
            logp.append(lp.mean().item())
            match.append(bool((am == full_am).all()))
        np.savez_compressed(out_dir / f"{g}.npz", **feats, hedge_pos=np.array(hedge_pos, int), n_resp=n, n_prompt=p,
                            commit_logp=np.array(logp[::-1], np.float32), commit_match=np.array(match[::-1]),
                            commit_bounds=bounds, answer=tokenizer.decode(answer), had_marker=had_marker)
        torch.cuda.empty_cache()
        print(f"  {k}/{len(groups)} {g}: {n} resp tokens, {len(hedge_pos)} hedges, answer {tokenizer.decode(answer)!r}, "
              f"logp curve {np.round(logp[::-1], 1).tolist()}, {time.time() - start:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", default=["greedy", "acts"], choices=["greedy", "acts", "sample", "pool", "segs", "sep", "sep2", "lens", "introspect", "tsegs", "lookback", "noise", "hedge", "effort", "liref", "difficulty", "pcprompt", "nothink"])
    parser.add_argument("--nt-mode", default="direct", choices=["direct", "free"])
    parser.add_argument("--nt-max-new", type=int, default=16)
    parser.add_argument("--models", nargs="+", default=None, help="Default: every model in the case file.")
    parser.add_argument("--cases", default="cases.jsonl", help="Case file under data/cvbench.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--segments", type=int, default=10)
    parser.add_argument("--tsegs-budgets", type=int, nargs="+", default=[8192, 16384], help="tsegs: response cuts.")
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
                    elif stage == "noise":
                        run_noise(model, tokenizer, model_id, todo)
                    elif stage == "liref":
                        run_liref(model, tokenizer, model_id)
                    elif stage == "pcprompt":
                        run_pcprompt(model, tokenizer, model_id)
                    elif stage == "difficulty":
                        run_difficulty(model, tokenizer, model_id)
                    elif stage == "effort":
                        run_effort(model, tokenizer, model_id, args.segments)
                    elif stage == "hedge":
                        run_hedge(model, tokenizer, model_id, todo, args.segments)
                    elif stage == "lookback":
                        run_lookback(model, tokenizer, model_id, todo, args.segments)
                    elif stage == "nothink":
                        run_nothink(model, tokenizer, model_id, todo, args.n, args.nt_mode, args.nt_max_new)
                    elif stage == "tsegs":
                        run_tsegs(model, tokenizer, model_id, todo, args.segments, sorted(args.tsegs_budgets))
                    else:
                        run_sample(model, tokenizer, model_id, todo, args.n, args.budget)
        finally:
            del model, tokenizer
            release_model()


if __name__ == "__main__":
    main()
