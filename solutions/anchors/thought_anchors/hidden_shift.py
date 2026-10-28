#!/usr/bin/env python3
"""Hidden shift under a fixed reasoning trace (white-box, HF transformers, one GPU).

For each case screened by ``resample.py case``, take every original-prompt
trace from ``screen.jsonl`` and teacher-force it, unchanged, after the original
prompt X and after each altered prompt. One forward pass per (trace, prompt),
no generation. At the trace positions, compare each prompt against X:

* ``kl``      KL(p(.|X, trace<t) || p(.|prompt, trace<t)) of the next token
* ``dlogp``   log p(trace token | prompt) - log p(trace token | X), summed = the
              log-likelihood ratio of the whole trace
* ``cos``     1 - cosine of the residual stream, per layer
* ``router``  1 - Jaccard overlap of the top-4 expert sets, per MoE layer
* ``gold``    change in log p of the gold answer token at the trace's last \\boxed{
* ``prompt_cos``  1 - cosine at the last prompt token (prompt-only baseline, no trace)

Prompts: the 4 code-made perturbations (seed 0, as in ``resample.py``), and
three controls that should do nothing: ``ctrl_ws`` (extra blank lines),
``ctrl_neutral`` (a number-free irrelevant sentence where the distractor goes,
mean of 3 sentences), ``distract_nonum`` (the case's distractor with its
numbers replaced by words). The trace's first ``--skip`` tokens are left out
of the means (attention-sink / position effects).

Per-sentence KL is lined up with the ``transplant`` P(correct) curve of the
trace the curves were run on (sentences split as in ``resample.py``).

    uv run solutions/anchors/thought_anchors/hidden_shift.py
    uv run solutions/anchors/thought_anchors/hidden_shift.py --cases robust_erased --traces 2

Outputs per case dir: ``hidden_shift.json`` (summary) and ``hidden_shift.npz``
(per-token kl / dlogp / mid-layer cos / router shift, per trace and prompt).
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/gpfs/projects/punim0478/bansaab")  # online: the MXFP4 kernel resolves its version tag

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "datagen"), str(ROOT / "solutions" / "perturb-probe")]

from common import read_jsonl  # noqa: E402
from perturb_probe.perturbations import make_perturbations  # noqa: E402
from resample import reasoning_of, reasoning_prompt, split_sentences  # noqa: E402

FINAL = "<|end|><|start|>assistant<|channel|>final<|message|>"
NEUTRAL = ("The problem was written neatly on a clean sheet of paper.",
           "It was a quiet afternoon and the room was mostly empty.",
           "A student read the following problem slowly before starting.")


# ---------------------------------------------------------------------------
# Prompts and traces
# ---------------------------------------------------------------------------

def inserted_span(original: str, altered: str) -> tuple[int, int]:
    """Character span of the single sentence ``distract`` inserted into ``original``."""
    ops = [op for op in difflib.SequenceMatcher(None, original, altered, autojunk=False).get_opcodes()
           if op[0] != "equal"]
    return min(op[3] for op in ops), max(op[4] for op in ops)


def nonum(sentence: str) -> str:
    words = iter(("many", "some"))
    return re.sub(r"\d+", lambda m: next(words, "several"), sentence)


def prompts_for(problem: str) -> dict[str, list[str]]:
    """name -> prompt texts (controls with several texts are averaged)."""
    perturbed = dict(make_perturbations(problem, 0))
    out = {"X": [problem], **{kind: [text] for kind, text in perturbed.items()}}
    start, end = inserted_span(problem, perturbed["distract"])
    dx = perturbed["distract"]
    sentence = dx[start:end].strip()
    out["distract_nonum"] = [dx[:start] + dx[start:end].replace(sentence, nonum(sentence)) + dx[end:]]
    out["ctrl_neutral"] = [dx[:start] + dx[start:end].replace(sentence, s) + dx[end:] for s in NEUTRAL]
    out["ctrl_ws"] = [problem.replace("\n", "\n\n") + "\n"]
    return out


def trace_parts(text: str) -> tuple[str, str]:
    """(reasoning, final answer) of a screen completion; providers print the harmony boundary as text."""
    reasoning = reasoning_of(text)
    rest = text[len(reasoning):]
    final = re.sub(r"^\s*assistant\s*final", "", rest) if not rest.startswith("<|end|>") else rest.split("<|message|>", 1)[-1]
    return reasoning, final


def sentence_starts(reasoning: str, sentences: list[str]) -> list[int]:
    """Character offset of each ``split_sentences`` chunk in the raw reasoning text."""
    starts, cursor = [], 0
    for sentence in sentences:
        head = r"\s+".join(map(re.escape, sentence.split()[:6]))
        match = re.compile(head).search(reasoning, cursor)
        starts.append(match.start() if match else cursor)
        cursor = match.end() if match else cursor
    return starts


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Model:
    def __init__(self, model_id: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id)
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="cuda").eval()
        print(f"loaded {model_id} in {time.time() - t0:.0f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
        self.routes: dict[int, torch.Tensor] = {}
        layers = self.model.model.layers
        for i, layer in enumerate(layers):
            layer.mlp.register_forward_hook(self._hook(i, layer.mlp.router.top_k))
        self.n_layers = len(layers)

    def _hook(self, i: int, top_k: int):
        def hook(module, inputs, output):
            # The MXFP4 forward skips the router module and returns (out, router_logits);
            # the dense one returns (out, router_scores). Top-k of either is the expert set.
            self.routes[i] = output[1].topk(top_k, dim=-1).indices
        return hook

    @torch.no_grad()
    def run(self, prompt_ids: list[int], resp_ids: list[int], chunk: int = 1024):
        """Logits predicting each trace token, residual stream at trace tokens, expert sets, last-prompt state.

        The sequence goes through in chunks over a KV cache: eager attention with sinks
        materialises a full score matrix, which runs out of memory on 3k+ token traces.
        """
        n = len(resp_ids)
        ids = torch.tensor([prompt_ids + resp_ids], device="cuda")
        logits, hidden, routes, cache = [], [], [], None
        for s in range(0, ids.shape[1], chunk):
            self.routes.clear()
            out = self.model(input_ids=ids[:, s:s + chunk], past_key_values=cache, use_cache=True,
                             output_hidden_states=True)
            cache = out.past_key_values
            logits.append(out.logits[0].to(torch.bfloat16))
            hidden.append(torch.stack([h[0] for h in out.hidden_states]))  # [L+1, c, d]
            routes.append(torch.stack([self.routes[i].reshape(out.logits.shape[1], -1) for i in range(self.n_layers)]))
            del out
        logits, hidden, routes = torch.cat(logits)[-n - 1:-1], torch.cat(hidden, 1), torch.cat(routes, 1)
        prompt_last = hidden[:, -n - 1]  # [L+1, d]
        return logits, hidden[:, -n:], routes[:, -n:], prompt_last  # logits[p] predicts trace token p


def compare(ref, alt, resp_ids: torch.Tensor, chunk: int = 512) -> dict[str, torch.Tensor]:
    (lx, hx, rx, px), (lv, hv, rv, pv) = ref, alt
    kl, dlogp = [], []
    for s in range(0, lx.shape[0], chunk):
        a, b = torch.log_softmax(lx[s:s + chunk].float(), -1), torch.log_softmax(lv[s:s + chunk].float(), -1)
        kl.append((a.exp() * (a - b)).sum(-1))
        tok = resp_ids[s:s + chunk, None]
        dlogp.append((b.gather(-1, tok) - a.gather(-1, tok))[:, 0])
    cos = 1 - torch.nn.functional.cosine_similarity(hx.float(), hv.float(), dim=-1)  # [L+1, n]
    inter = (rx[..., :, None] == rv[..., None, :]).any(-1).sum(-1).float()  # [L, n]
    k = rx.shape[-1]
    router = 1 - inter / (2 * k - inter)
    prompt_cos = 1 - torch.nn.functional.cosine_similarity(px.float(), pv.float(), dim=-1)  # [L+1]
    return dict(kl=torch.cat(kl), dlogp=torch.cat(dlogp), cos=cos, router=router, prompt_cos=prompt_cos)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_case(m: Model, case_dir: Path, args) -> dict:
    case = json.loads((case_dir / "case.json").read_text())
    gold_tok = m.tok(str(case["gold"]), add_special_tokens=False)["input_ids"]
    prompts = prompts_for(case["problem"])
    assert prompts["distract"][0] == case["dx_text"] or case["perturbation"] != "distract", "perturbations drifted"
    prompt_ids = {name: [m.tok(reasoning_prompt(m.tok, t, case["effort"]), add_special_tokens=False)["input_ids"]
                         for t in texts] for name, texts in prompts.items()}
    screen = [r for r in read_jsonl(case_dir / "screen.jsonl") if r["curve"] == "original"][: args.traces]
    curve_trace = next(i for i, r in enumerate(screen) if r["correct"])  # resample.py's x_traces[0]
    mid = m.n_layers // 2

    per_trace, arrays = [], {}
    for ti, row in enumerate(screen):
        reasoning, final = trace_parts(row["text"])
        resp = reasoning + FINAL + final
        enc = m.tok(resp, add_special_tokens=False, return_offsets_mapping=True)
        resp_ids, offsets = enc["input_ids"], enc["offset_mapping"]
        n_reason = sum(1 for s, _ in offsets if s < len(reasoning))
        box = resp.rfind("\\boxed{")
        box_pos = next((i for i, (s, _) in enumerate(offsets) if s >= box + len("\\boxed{")), None) if box >= 0 else None
        ids_t = torch.tensor(resp_ids, device="cuda")
        ref = m.run(prompt_ids["X"][0], resp_ids)
        ref_gold = (torch.log_softmax(ref[0][box_pos].float(), -1)[gold_tok[0]].item() if box_pos is not None else None)
        result = dict(trace=ti, correct=row["correct"], answer=row["answer"], n_tokens=len(resp_ids),
                      n_reasoning=n_reason, logp_gold_X=ref_gold, prompts={})
        keep = slice(args.skip, n_reason)
        # noise_X: X itself with other chunk boundaries -- the kernels' numerical floor.
        conditions = [(name, ids, 1024) for name, ids in prompt_ids.items() if name != "X"]
        for name, ids_list, chunk in conditions + [("noise_X", prompt_ids["X"], 1000)]:
            stats, golds = [], []
            for ids in ids_list:
                alt = m.run(ids, resp_ids, chunk=chunk)
                stats.append(compare(ref, alt, ids_t))
                if box_pos is not None:
                    golds.append(torch.log_softmax(alt[0][box_pos].float(), -1)[gold_tok[0]].item() - ref_gold)
                del alt
            avg = {k: torch.stack([s[k] for s in stats]).mean(0) for k in stats[0]}
            gold = float(np.mean(golds)) if golds else None
            result["prompts"][name] = dict(
                kl=avg["kl"][keep].mean().item(),
                kl_p99=avg["kl"][keep].quantile(0.99).item(),
                kl_final=avg["kl"][n_reason:].mean().item(),
                dlogp_sum=avg["dlogp"][keep].sum().item(),
                cos_mid=avg["cos"][mid, keep].mean().item(),
                cos_layers=avg["cos"][:, keep].mean(1).tolist(),
                router=avg["router"][:, keep].mean().item(),
                router_layers=avg["router"][:, keep].mean(1).tolist(),
                prompt_cos_mid=avg["prompt_cos"][mid].item(),
                dlogp_gold=gold)
            for key, value in (("kl", avg["kl"]), ("dlogp", avg["dlogp"]), ("cos_mid", avg["cos"][mid]),
                               ("router", avg["router"].mean(0))):
                arrays[f"t{ti}/{name}/{key}"] = value.cpu().numpy().astype(np.float32)
        if ti == curve_trace:
            sentences = split_sentences(reasoning)
            starts = sentence_starts(reasoning, sentences)
            sent_of = np.searchsorted(starts, [s for s, _ in offsets[:n_reason]], side="right") - 1
            result["sentence_kl"] = {
                name: [float(arrays[f"t{ti}/{name}/kl"][:n_reason][sent_of == j].mean()) if (sent_of == j).any() else None
                       for j in range(len(sentences))] for name in result["prompts"]}
            result["sentences"] = sentences
        per_trace.append(result)
        print(f"  {case_dir.name} trace {ti} ({'ok' if row['correct'] else 'wrong'}, {len(resp_ids)} tok): "
              + " ".join(f"{k}={v['kl']:.4f}" for k, v in result["prompts"].items()), flush=True)
        del ref
        torch.cuda.empty_cache()

    np.savez_compressed(case_dir / "hidden_shift.npz", **arrays)
    return dict(case=case["case"], is_robust=case["is_robust"], screen_accuracy=case["screen_accuracy"],
                curve_trace=curve_trace, traces=per_trace)


def report(summary: dict) -> None:
    metrics = ("kl", "kl_p99", "dlogp_sum", "cos_mid", "router", "prompt_cos_mid", "dlogp_gold")
    for c in summary.values():
        print(f"\n== {c['case']} (robust={c['is_robust']}) | screen accuracy {c['screen_accuracy']}")
        names = list(c["traces"][0]["prompts"])
        print(f"{'prompt':15s}" + "".join(f"{m:>15s}" for m in metrics) + f"{'kl/neutral':>12s}{'kl/ws':>9s}"
              f"{'excess/neutral':>16s}")
        excess = lambda t, name: t["prompts"][name]["kl"] - t["prompts"]["noise_X"]["kl"]  # noqa: E731
        for name in names:
            vals = {m: [t["prompts"][name][m] for t in c["traces"] if t["prompts"][name][m] is not None] for m in metrics}
            line = "".join(f"{np.mean(v):9.4f}±{np.std(v):<5.3f}" if v else f"{'-':>15s}" for v in vals.values())
            ratio_n = np.mean([t["prompts"][name]["kl"] / t["prompts"]["ctrl_neutral"]["kl"] for t in c["traces"]])
            ratio_w = np.mean([t["prompts"][name]["kl"] / t["prompts"]["ctrl_ws"]["kl"] for t in c["traces"]])
            ratio_e = np.mean([excess(t, name) for t in c["traces"]]) / np.mean([excess(t, "ctrl_neutral") for t in c["traces"]])
            print(f"{name:15s}{line}{ratio_n:12.2f}{ratio_w:9.2f}{ratio_e:16.2f}")
        t = c["traces"][c["curve_trace"]]
        sk = t["sentence_kl"]["distract"]
        top = sorted((v, j) for j, v in enumerate(sk) if v is not None)[-5:][::-1]
        print(f"top distract-KL sentences on the curve trace (#{c['curve_trace']}):")
        for v, j in top:
            print(f"  k={j:3d} kl={v:.4f}  {t['sentences'][j][:110]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "thought_anchors_crusoe")
    parser.add_argument("--cases", nargs="*", default=["fragile_ninths", "robust_erased"])
    parser.add_argument("--effort", default="low")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--traces", type=int, default=8, help="Original-prompt screen traces per case.")
    parser.add_argument("--skip", type=int, default=8, help="Leading trace tokens left out of the means.")
    args = parser.parse_args()

    m = Model(args.model)
    summary = {}
    for name in args.cases:
        case_dir = args.data_dir / f"{name}__{args.effort}"
        summary[name] = run_case(m, case_dir, args)
        (case_dir / "hidden_shift.json").write_text(json.dumps(summary[name], indent=1))
    report(summary)


if __name__ == "__main__":
    main()
