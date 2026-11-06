#!/usr/bin/env python3
"""Attention look-back regions for any cvbench model (generalises circuit_attr.py attn, 2026-11-02).
Per case: for layers lo..hi (Skywork's 6..13 of 28 as fractions of depth: lo = round(6/28 L), hi = L/2 - 1), per
head and response decile, attention mass on [BOS sink, template, problem, CoT > 64 back, last 64]; layers without
attention weights (Qwen3.5 linear attention) are NaN -> data/cvbench/circuit_attn/<model>/<group>.npz

    python cvbench/attn_collect.py --models deepseek-ai/DeepSeek-R1-0528-Qwen3-8B allenai/Olmo-3-7B-Think Qwen/Qwen3.5-4B
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "cvbench"), str(ROOT / "solutions" / "uncertainty-profiling")]
import cv  # noqa: E402

SEGMENTS, RECENT = 10, 64


class Stop(Exception):
    pass


def decoder_layers(model):
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers", "model.model.layers"):
        obj = model
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            return obj
        except AttributeError:
            continue
    raise AttributeError("decoder layers not found")


def layer_range(L):
    return int(round(6 / 28 * L)), L // 2 - 1


def run(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from uncertainty_profile.extraction import format_problem
    cases = cv.load_cases(list(cv.CASE_FILES))
    cases = cases[cases.model_id == model_id]
    greedy = {r["group"]: r for r in cv.read_jsonl(cv.DATA / "greedy" / f"{cv.safe(model_id)}.jsonl")}
    out = cv.DATA / "circuit_attn" / cv.safe(model_id)
    out.mkdir(parents=True, exist_ok=True)
    todo = [g for g in cases.group if not (out / f"{g}.npz").is_file()]
    print(f"[attn] {model_id}: {len(todo)} to do", flush=True)
    if not todo:
        return
    # same tokenizer as the runtime (uncertainty_profile.extraction.load_model_and_tokenizer): the fast tokenizer
    # first, since AutoTokenizer picks a legacy conversion for some checkpoints (DeepSeek-R1-0528-Qwen3-8B)
    from transformers import PreTrainedTokenizerFast
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(model_id, local_files_only=True)
    except (OSError, TypeError, ValueError):
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True, dtype=torch.bfloat16,
                                                 attn_implementation="eager", device_map="cuda:0").eval()
    layers = decoder_layers(model)
    lo, hi = layer_range(len(layers))
    print(f"  {len(layers)} layers, using {lo}..{hi}", flush=True)
    st = {}

    def hook(l):
        def f(mod, inp, o):
            A = o[1] if isinstance(o, tuple) and len(o) > 1 else None
            if A is None:
                return
            p, n, ps, pe, b = st["p"], st["n"], st["ps"], st["pe"], st["bounds"]
            Q = A[0, :, p:p + n].float()
            prob = Q[..., ps:pe].sum(-1)
            sink = Q[..., 0]
            templ = Q[..., 1:p].sum(-1) - prob
            C = Q[..., p:].cumsum(-1)
            i = torch.arange(n, device=Q.device)
            far = torch.where(i - RECENT - 1 >= 0, C[:, i, (i - RECENT - 1).clamp(min=0)], torch.zeros_like(sink))
            recent = C[:, i, i] - far
            R = torch.stack([sink, templ, prob, far, recent], -1)
            st["reg"][l] = torch.stack([R[:, a:bb].mean(1) for a, bb in zip(b[:-1], b[1:])], 1).cpu().numpy()
            del Q, C, R
        return f

    def stop(m, i, o):
        raise Stop

    hooks = [layers[l].self_attn.register_forward_hook(hook(l)) for l in range(lo, hi + 1) if hasattr(layers[l], "self_attn")]
    hooks.append(layers[hi].register_forward_hook(stop))
    nH = None
    for k, g in enumerate(todo, 1):
        problem = cases.loc[cases.group == g, "problem"].iloc[0]
        prompt = format_problem(tok, problem)
        enc = tok(prompt, add_special_tokens=False, truncation=True, max_length=2048, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        c0 = prompt.find(problem); c1 = c0 + len(problem)
        inside = [j for j, (a, b) in enumerate(offs) if a >= c0 and b <= c1 and b > a]
        resp = tok(greedy.get(g, {}).get("generated_text", ""), add_special_tokens=False)["input_ids"][:4096]
        p, n = len(ids), len(resp)
        if n < SEGMENTS or not inside:
            print(f"  {g}: skipped (n={n})", flush=True); continue
        st.update(p=p, n=n, ps=inside[0], pe=inside[-1] + 1, reg={},
                  bounds=np.linspace(0, n, SEGMENTS + 1).round().astype(int))
        with torch.no_grad():
            try:
                model(input_ids=torch.tensor([ids + resp], device="cuda:0"))
            except Stop:
                pass
        nH = nH or next(iter(st["reg"].values())).shape[0]
        regions = np.stack([st["reg"].get(l, np.full((nH, SEGMENTS, 5), np.nan, np.float32)) for l in range(lo, hi + 1)])
        np.savez_compressed(out / f"{g}.npz", regions=regions.astype(np.float32), layers=np.arange(lo, hi + 1))
        torch.cuda.empty_cache()
        if k in (1, len(todo)) or k % 10 == 0:
            print(f"  {k}/{len(todo)} {g}: layers with weights {sorted(st['reg'])}", flush=True)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    for m in ap.parse_args().models:
        run(m)
