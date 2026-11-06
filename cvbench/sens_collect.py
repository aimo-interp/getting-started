#!/usr/bin/env python3
"""Logit-lens commitment depth + problem-perturbation sensitivity of the forced answer (2026-11-03).

Setup as generate.py --stages noise: greedy trace (cut at end-of-thinking, <= 4096 tokens) closed with the
forcing suffix, answer A decoded greedily up to '}'. Then
  clean pass (prompt + trace + suffix + A), all hidden states:
    lens_logp [L+1, |A|]  logit-lens log p of each answer token from every layer (final norm + lm_head)
  per perturbation of the problem (perturb_probe.make_perturbations: typos, rename, distract, rephrase), the
  SAME trace + suffix + A teacher-forced after the perturbed prompt:
    pert_logp [4]   mean log p(A);  pert_flip [4] argmax over A changed
    pert_cos_cot [4] cosine distance of the mean layer-L/2 CoT residual to clean; pert_cos_last [4] same, last pos
-> data/cvbench/sens/<model>/<group>.npz

    python cvbench/sens_collect.py --models Skywork/Skywork-OR1-Math-7B ...
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "cvbench")]
import generate as G  # noqa: E402
from perturb_probe.perturbations import make_perturbations  # noqa: E402


def final_norm(model):
    for path in ("model.norm", "model.language_model.norm", "model.model.norm", "language_model.model.norm"):
        obj = model
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            return obj
        except AttributeError:
            continue
    raise AttributeError("final norm not found")


@torch.no_grad()
def forward(model, ids, n_ans, device, all_hidden=False, mid=None):
    """log p of the last n_ans tokens, argmax there, and hidden states (all layers or the `mid` one)."""
    out = model(input_ids=torch.tensor([ids], device=device), output_hidden_states=True, use_cache=False)
    logits = out.logits[0, -n_ans - 1:-1].float()
    tgt = torch.tensor(ids[-n_ans:], device=device)
    lp = torch.log_softmax(logits, -1).gather(-1, tgt[:, None])[:, 0]
    hs = out.hidden_states if all_hidden else out.hidden_states[mid]
    return lp, logits.argmax(-1), hs


@torch.no_grad()
def run(model_id):
    from uncertainty_profile.extraction import load_model_and_tokenizer
    import cv
    cases = cv.load_cases(list(cv.CASE_FILES))
    cases = cases[cases.model_id == model_id]
    greedy = {r["group"]: r for r in cv.read_jsonl(cv.DATA / "greedy" / f"{cv.safe(model_id)}.jsonl")}
    out_dir = cv.DATA / "sens" / cv.safe(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(g, p) for g, p in zip(cases.group, cases.problem) if not (out_dir / f"{g}.npz").is_file()]
    print(f"[sens] {model_id}: {len(todo)} to do", flush=True)
    if not todo:
        return
    model, tok = load_model_and_tokenizer(model_id, local_files_only=True)
    model.eval()
    device = model.get_input_embeddings().weight.device
    norm, head = final_norm(model), model.get_output_embeddings()
    L = len(G.decoder_layers(model))
    mid = int(round(0.5 * L))
    for k, (g, problem) in enumerate(todo, 1):
        t0 = time.time()
        reasoning, suffix, _ = G.forced_answer_text(model, greedy.get(g, {}).get("generated_text", ""))
        cot = tok(reasoning, add_special_tokens=False)["input_ids"][:4096]
        if len(cot) < 10:
            print(f"  {g}: skipped", flush=True); continue
        suf = tok(suffix, add_special_tokens=False)["input_ids"]
        prompt = tok(G.format_problem(tok, problem), add_special_tokens=False)["input_ids"]
        head_ids = prompt + cot + suf
        enc = torch.tensor([head_ids], device=device)
        gen = model.generate(input_ids=enc, attention_mask=torch.ones_like(enc), max_new_tokens=16, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)[0, len(head_ids):].tolist()
        ans = []
        for t in gen:
            ans.append(t)
            if "}" in tok.decode([t]):
                break
        n = len(ans)
        lp0, am0, hs = forward(model, head_ids + ans, n, device, all_hidden=True)
        tgt = torch.tensor(ans, device=device)
        lens = []
        for h in hs:  # logit lens at the positions predicting the answer tokens
            x = h[0, -n - 1:-1]
            x = norm(x) if h is not hs[-1] else x      # last hidden state is already normed in HF
            lens.append(torch.log_softmax(head(x).float(), -1).gather(-1, tgt[:, None])[:, 0])
        lens = torch.stack(lens).cpu().numpy()
        p = len(prompt)
        h0 = hs[mid][0].float()
        cot0, last0 = h0[p:p + len(cot)].mean(0), h0[-1]
        del hs
        variants = make_perturbations(problem)
        plp, pflip, pcc, pcl = [], [], [], []
        for _, vtext in variants:
            vp = tok(G.format_problem(tok, vtext), add_special_tokens=False)["input_ids"]
            lp, am, h = forward(model, vp + cot + suf + ans, n, device, mid=mid)
            h = h[0].float()
            plp.append(lp.mean().item()); pflip.append(bool((am != am0).any()))
            pcc.append(1 - torch.cosine_similarity(h[len(vp):len(vp) + len(cot)].mean(0), cot0, 0).item())
            pcl.append(1 - torch.cosine_similarity(h[-1], last0, 0).item())
        np.savez_compressed(out_dir / f"{g}.npz", lens_logp=lens.astype(np.float32), logp_clean=lp0.cpu().numpy(),
                            pert_logp=np.array(plp), pert_flip=np.array(pflip), pert_cos_cot=np.array(pcc),
                            pert_cos_last=np.array(pcl), pert_types=np.array([t for t, _ in variants]),
                            answer=tok.decode(ans), mid=mid, n_cot=len(cot))
        torch.cuda.empty_cache()
        print(f"  {k}/{len(todo)} {g}: ans {tok.decode(ans)!r} clean {lp0.mean():.2f} pert {np.round(plp, 2).tolist()} "
              f"{time.time() - t0:.0f}s", flush=True)
    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    for m in ap.parse_args().models:
        run(m)
