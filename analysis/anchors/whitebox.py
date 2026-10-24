#!/usr/bin/env python3
"""White-box analyses (project env, HF transformers) on the cases chosen by ``resample.py``.

Reads ``work/selection.json`` and, where available, ``work/anchors.jsonl``.

* ``probe`` (analysis 3): the response-mean diff-in-means direction (layer
  frac 0.5, fitted leave-this-problem-out on ``data/cvbench/acts``) applied
  token by token to the X and dX traces. Saves the per-token projection, its
  running mean (= the probe's score if the response stopped there) and the
  per-sentence mean, and lines them up with the P(correct) curves.
* ``lens`` (analysis 4): force ``</think> The final answer is $\\boxed{<gold>}$``
  after 0%, 10%, 25%, 50%, 75%, 100% of the reasoning and read log p(gold) at
  every layer through the logit lens and a tuned lens (Belrose et al. 2023)
  trained here on the model's own screening traces (``--lens-steps``; 0 skips).
* ``attn`` (analysis 5): on the dX trace, attention mass from each response
  token to the dX prompt tokens that differ from X (difflib on token ids),
  for every head of the 8 full-attention layers (the 24 Gated DeltaNet layers
  have no attention matrix). Receiver heads are found as in Thought Anchors
  (kurtosis of sentence-level vertical attention). Mass is reported per
  sentence, overall and in receiver heads, with the anchor sentences marked.
* ``steer`` (analysis 6): on the robust case's dX trace, at the start of
  reasoning and at its commitment point (the last prefix before P(correct)
  locks at >= 0.8), add alpha * median||h|| * u to the residual stream at the
  probe layer, u pointing toward "not robust", for every generated token, and
  resample. Control: a random unit direction at the largest alpha.

    uv run analysis/anchors/whitebox.py --stages probe lens attn steer
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

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "datagen"), str(HERE)]

from common import SOLVE_SUFFIX, extract_answer, read_jsonl  # noqa: E402
from resample import choose_points, curve_features, sentence_ends, thinking_of  # noqa: E402

WORK = Path(os.environ.get("ANCHORS_WORK", HERE / "work"))
FORCE = "\n</think>\n\nThe final answer is $\\boxed{"


def safe(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)


# ---------------------------------------------------------------- model plumbing
class Model:
    def __init__(self, model_id: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
        self.text = next(m for m in self.model.modules()
                         if hasattr(m, "layers") and hasattr(m, "norm") and hasattr(m, "embed_tokens"))
        self.layers = self.text.layers
        self.norm = self.text.norm
        self.lm_head = self.model.get_output_embeddings()
        self.device = self.lm_head.weight.device
        cfg = self.text.config
        types = getattr(cfg, "layer_types", None) or ["full_attention"] * len(self.layers)
        self.full_layers = [i for i, t in enumerate(types) if t == "full_attention"]
        self.model_type = cfg.model_type

    def prompt_text(self, problem: str) -> str:
        return self.tok.apply_chat_template([{"role": "user", "content": problem + SOLVE_SUFFIX}],
                                            tokenize=False, add_generation_prompt=True)

    def prompt_ids(self, problem: str) -> list[int]:
        return self.tok(self.prompt_text(problem), add_special_tokens=False)["input_ids"]

    def problem_mask(self, problem: str) -> np.ndarray:
        """Which prompt tokens lie inside the problem statement."""
        text = self.prompt_text(problem)
        start = text.find(problem)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return np.array([s >= start and e <= start + len(problem) and e > s for s, e in enc["offset_mapping"]])

    def encode_with_offsets(self, text: str) -> tuple[list[int], np.ndarray]:
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], np.array([e for _, e in enc["offset_mapping"]], dtype=np.int64)

    def residual_hook(self, layer_idx: int, store: dict, key: str, sl=slice(None)):
        """Capture the output of decoder layer ``layer_idx`` (= hidden_states[layer_idx + 1])."""
        def hook(_module, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            store[key] = h[0, sl].detach()
        return self.layers[layer_idx].register_forward_hook(hook)


def tok_index(ends_chars: np.ndarray, offset: int) -> int:
    """Number of trace tokens that end at or before a character offset."""
    return int(np.searchsorted(ends_chars, offset, side="right"))


def load_selection() -> list[dict]:
    return json.loads((WORK / "selection.json").read_text())


def load_anchor_rows() -> dict:
    return {(r["group"], r["curve"]): r for r in read_jsonl(WORK / "anchors.jsonl")}


def probe_direction(model_id: str, group: str, layer_frac: float):
    """Leave-``group``-out diff-in-means direction (robust - not robust) on resp_mean at the probe layer."""
    cases = [c for c in read_jsonl(ROOT / "data/cvbench/cases.jsonl") if c["model_id"] == model_id]
    keep = [c for c in cases if c["group"] != group]
    h = np.stack([np.load(ROOT / "data/cvbench/acts" / safe(model_id) / f"{c['group']}.npz")["resp_mean"]
                  .astype(np.float32) for c in keep])
    layer = int(round(layer_frac * (h.shape[1] - 1)))
    h = h[:, layer]
    y = np.array([c["is_robust"] for c in keep])
    mu_r, mu_n = h[y].mean(0), h[~y].mean(0)
    u = (mu_r - mu_n) / np.linalg.norm(mu_r - mu_n)
    return layer, u, float(0.5 * (mu_r + mu_n) @ u), float((h @ u).std())


def traces_for(sel) -> list[tuple[str, str, str]]:
    return [("X", sel["x_text"], sel["x_trace"]), ("dX", sel["dx_text"], sel["dx_trace"])]


# ---------------------------------------------------------------- analysis 3
@torch.no_grad()
def run_probe(m: Model, args, selection, anchors) -> dict:
    out = {}
    for sel in selection:
        layer, u_np, mid, std = probe_direction(args.model, sel["group"], args.layer_frac)
        u = torch.tensor(u_np, device=m.device, dtype=torch.float32)
        res = {}
        for curve, problem, trace in traces_for(sel):
            thinking = thinking_of(trace)
            p_ids = m.prompt_ids(problem)
            t_ids, t_ends = m.encode_with_offsets(thinking)
            t_ids, t_ends = t_ids[:args.max_trace_tokens], t_ends[:args.max_trace_tokens]
            store = {}
            handle = m.residual_hook(layer - 1, store, "h", slice(len(p_ids), None))
            m.model(input_ids=torch.tensor([p_ids + t_ids], device=m.device), logits_to_keep=1)
            handle.remove()
            proj = (store["h"].float() @ u - mid).cpu().numpy()
            running = np.cumsum(proj) / np.arange(1, len(proj) + 1)
            ends = sentence_ends(thinking)
            sent_tok = [min(tok_index(t_ends, e), len(proj)) for e in ends]
            starts = [0] + sent_tok[:-1]
            sent_mean = [float(proj[a:b].mean()) if b > a else float("nan") for a, b in zip(starts, sent_tok)]
            final = running[-1]
            settle = next((i for i in range(len(running)) if np.all(np.sign(running[i:]) == np.sign(final))), None)
            row = dict(n_tokens=len(proj), final_score=float(final), settle_token=settle,
                       running_at=[float(running[min(t, len(running)) - 1]) for t in (128, 256, 512, 1024, 2048, 4096)
                                   if t <= len(running)],
                       sentence_mean=sent_mean, sentence_token_ends=sent_tok)
            a = anchors.get((sel["group"], curve))
            if a:  # line the probe up with the resampling curve at its prefix points
                pts = a["points"]
                toks = [min(tok_index(t_ends, pt["offset"]), len(proj)) for pt in pts]
                row["curve_p"] = [pt["p"] for pt in pts]
                row["curve_probe_running"] = [float(running[t - 1]) if t > 0 else float("nan") for t in toks]
                row["curve_probe_local"] = [float(proj[max(0, t - 64):t].mean()) if t > 0 else float("nan")
                                            for t in toks]
                ok = [i for i, t in enumerate(toks) if t > 0]
                if len(ok) > 2:
                    row["corr_p_vs_running"] = float(np.corrcoef([row["curve_p"][i] for i in ok],
                                                                 [row["curve_probe_running"][i] for i in ok])[0, 1])
                feats = curve_features(pts, thinking, a["sentence_ends"])
                # Probe change across each anchor span vs the typical change between adjacent points.
                deltas = [row["curve_probe_local"][i] - row["curve_probe_local"][i - 1] for i in range(2, len(pts))]
                row["anchor_probe_delta"] = []
                for anc in feats["anchors"]:
                    i = next(j for j, pt in enumerate(pts) if pt["k"] == anc["k_to"])
                    if i >= 2:
                        row["anchor_probe_delta"].append(dict(k_to=anc["k_to"], dp=anc["delta"],
                                                              dprobe=row["curve_probe_local"][i]
                                                              - row["curve_probe_local"][i - 1]))
                row["typical_abs_dprobe"] = float(np.nanmedian(np.abs(deltas))) if deltas else None
            np.save(WORK / f"wb_probe_{sel['group']}_{curve}.npy", proj.astype(np.float16))
            res[curve] = row
        out[sel["group"]] = dict(role=sel["role"], layer=layer, mid=mid, proj_std=std, curves=res)
        print(f"[probe] {sel['role']} {sel['group']}: " + "; ".join(
            f"{c}: final {r['final_score']:+.3f}, settles@{r['settle_token']}, corr(P,probe)={r.get('corr_p_vs_running', float('nan')):+.2f}"
            for c, r in res.items()), flush=True)
    (WORK / "wb_probe.json").write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- analysis 4
@torch.enable_grad()
def train_tuned_lens(m: Model, args) -> list[torch.nn.Linear] | None:
    path = WORK / "tuned_lens.pt"
    n_layers = len(m.layers)
    d = m.norm.weight.shape[0]
    lens = [torch.nn.Linear(d, d, device=m.device, dtype=torch.float32) for _ in range(n_layers - 1)]
    for t in lens:
        torch.nn.init.zeros_(t.weight)
        torch.nn.init.zeros_(t.bias)
    if path.is_file():
        state = torch.load(path, map_location=m.device)
        for t, s in zip(lens, state):
            t.load_state_dict(s)
        print(f"[lens] loaded {path}")
        return lens
    if args.lens_steps <= 0:
        return None
    texts = []
    for r in read_jsonl(WORK / "screen.jsonl"):
        for t in r["texts"]:
            texts.append((r["text"], t))
    if not texts:
        print("[lens] no screen.jsonl traces; skipping the tuned lens")
        return None
    rng = np.random.default_rng(0)
    params = [p for t in lens for p in t.parameters()]
    opt = torch.optim.Adam(params, lr=args.lens_lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / args.lens_steps))
    for p in m.model.parameters():
        p.requires_grad_(False)
    start = time.time()
    for step in range(args.lens_steps):
        problem, trace = texts[rng.integers(len(texts))]
        ids = (m.prompt_ids(problem) + m.tok(trace, add_special_tokens=False)["input_ids"])[:args.lens_seq]
        pos = torch.tensor(np.sort(rng.choice(len(ids), size=min(args.lens_positions, len(ids)), replace=False)),
                           device=m.device)
        store = {}
        handles = [m.residual_hook(i, store, i) for i in range(n_layers - 1)]
        with torch.no_grad():
            logits = m.model(input_ids=torch.tensor([ids], device=m.device)).logits[0, pos].float()
            target = torch.log_softmax(logits, -1)
        for h in handles:
            h.remove()
        total = 0.0
        for i, t in enumerate(lens):
            h = store[i][pos].float()
            pred = torch.log_softmax(m.lm_head(m.norm((h + t(h)).to(torch.bfloat16))).float(), -1)
            loss = torch.sum(target.exp() * (target - pred), -1).mean() / len(lens)
            loss.backward()
            total += loss.item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 25 == 0 or step == args.lens_steps - 1:
            print(f"[lens] step {step}/{args.lens_steps} mean KL {total:.3f} ({time.time() - start:.0f}s)", flush=True)
    torch.save([t.state_dict() for t in lens], path)
    return lens


@torch.no_grad()
def run_lens(m: Model, args, selection) -> dict:
    lens = train_tuned_lens(m, args)
    n_layers = len(m.layers)
    out = {}
    for sel in selection:
        gold_ids = m.tok(str(sel["gold"]), add_special_tokens=False)["input_ids"]
        res = {}
        for curve, problem, trace in traces_for(sel):
            thinking = thinking_of(trace)
            ends = sentence_ends(thinking)
            p_ids = m.prompt_ids(problem)
            rows = []
            for frac in args.lens_fracs:
                k = int(round(frac * len(ends)))
                prefix = thinking[:ends[k - 1]] if k > 0 else ""
                pre_ids = m.tok(prefix, add_special_tokens=False)["input_ids"][:args.max_trace_tokens]
                force_ids = m.tok(FORCE, add_special_tokens=False)["input_ids"]
                ids = p_ids + pre_ids + force_ids + gold_ids
                first = len(ids) - len(gold_ids) - 1  # position predicting the first gold token
                pos = slice(first, first + len(gold_ids))
                store = {}
                handles = [m.residual_hook(i, store, i, pos) for i in range(n_layers)]
                logits = m.model(input_ids=torch.tensor([ids], device=m.device), logits_to_keep=len(gold_ids) + 1).logits
                for h in handles:
                    h.remove()
                target = torch.tensor(gold_ids, device=m.device)
                def read(z):
                    logp = torch.log_softmax(m.lm_head(m.norm(z)).float(), -1)
                    lp = logp.gather(-1, target[:, None]).squeeze(-1)
                    return dict(logp_gold=float(lp.sum()), p_first=float(lp[0].exp()),
                                rank_first=int((logp[0] > logp[0, target[0]]).sum()))
                per_layer = {"logit": [], "tuned": []}
                for i in range(n_layers):
                    h = store[i]
                    per_layer["logit"].append(read(h))
                    if lens is None:
                        per_layer["tuned"].append(None)
                    elif i == n_layers - 1:  # the last layer is the model's own output
                        per_layer["tuned"].append(per_layer["logit"][-1])
                    else:
                        per_layer["tuned"].append(read((h.float() + lens[i](h.float())).to(h.dtype)))
                final_top = m.tok.decode(logits[0, -len(gold_ids) - 1:-1].argmax(-1))
                # earliest layer where the lens already ranks the gold first digit top-1 and keeps it
                def first_layer(kind):
                    xs = per_layer[kind]
                    for i in range(n_layers):
                        if all(x is not None and x["rank_first"] == 0 for x in xs[i:]):
                            return i + 1
                    return None
                rows.append(dict(frac=frac, k=k, prefix_tokens=len(pre_ids), final_greedy=final_top,
                                 logit=per_layer["logit"], tuned=per_layer["tuned"],
                                 first_layer_logit=first_layer("logit"),
                                 first_layer_tuned=first_layer("tuned") if lens is not None else None))
            res[curve] = rows
            print(f"[lens] {sel['role']} {sel['group']} {curve}: " + " ".join(
                f"{r['frac']:.2f}:p={r['logit'][-1]['p_first']:.2f}/L{r['first_layer_logit']}"
                f"{'/T' + str(r['first_layer_tuned']) if r['first_layer_tuned'] is not None else ''}"
                for r in rows), flush=True)
        out[sel["group"]] = dict(role=sel["role"], gold=sel["gold"], curves=res)
    (WORK / "wb_lens.json").write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- analysis 5
def edited_positions(x_ids: list[int], dx_ids: list[int]) -> np.ndarray:
    """Positions in dx_ids not matched to x_ids (inserted or replaced tokens)."""
    mask = np.zeros(len(dx_ids), dtype=bool)
    for op, _, _, j1, j2 in difflib.SequenceMatcher(None, x_ids, dx_ids, autojunk=False).get_opcodes():
        if op in ("replace", "insert"):
            mask[j1:j2] = True
    return mask


def qk_states(mod, hs, cos, sin) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotated queries and (GQA-expanded) keys of a Qwen3.5 full-attention layer, [H, L, hd] float32.

    Mirrors ``Qwen3_5Attention.forward`` up to the attention scores.
    """
    rope = sys.modules[type(mod).__module__].apply_rotary_pos_emb
    shape = (*hs.shape[:-1], -1, mod.head_dim)
    q, _ = torch.chunk(mod.q_proj(hs).view(*hs.shape[:-1], -1, mod.head_dim * 2), 2, dim=-1)
    q = mod.q_norm(q.view(shape)).transpose(1, 2)
    k = mod.k_norm(mod.k_proj(hs).view(shape)).transpose(1, 2)
    q, k = rope(q, k, cos, sin)
    k = k.repeat_interleave(mod.num_key_value_groups, dim=1)
    return q[0].float(), k[0].float()


def kurtosis(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 4 or x.std() == 0:
        return float("nan")
    z = (x - x.mean()) / x.std()
    return float((z ** 4).mean() - 3)


@torch.no_grad()
def run_attn(m: Model, args, selection, anchors) -> dict:
    out = {}
    for sel in selection:
        x_ids, dx_ids = m.prompt_ids(sel["x_text"]), m.prompt_ids(sel["dx_text"])
        edit = edited_positions(x_ids, dx_ids)
        prob = m.problem_mask(sel["dx_text"])
        thinking = thinking_of(sel["dx_trace"])
        t_ids, t_ends = m.encode_with_offsets(thinking)
        t_ids, t_ends = t_ids[:args.attn_max_tokens], t_ends[:args.attn_max_tokens]
        P, T = len(dx_ids), len(t_ids)
        ends = sentence_ends(thinking)
        sent_tok = [min(tok_index(t_ends, e), T) for e in ends]
        sent_tok = [t for i, t in enumerate(sent_tok) if t > (sent_tok[i - 1] if i else 0)]
        sent_of = np.zeros(T, dtype=np.int64)
        for i, (a, b) in enumerate(zip([0] + sent_tok[:-1], sent_tok)):
            sent_of[a:b] = i
        S = len(sent_tok)
        sent_len = np.bincount(sent_of, minlength=S).astype(np.float32)
        onehot = torch.zeros(T, S, device=m.device)
        onehot[torch.arange(T), torch.tensor(sent_of, device=m.device)] = 1.0
        key_avg = onehot / torch.tensor(sent_len, device=m.device).clamp(min=1)

        captured = {}

        def make_hook(li):
            def hook(module, args_, kwargs, output):
                captured[li] = (kwargs["hidden_states"].detach(), [t.detach() for t in kwargs["position_embeddings"]])
            return hook
        attn_mods = {li: m.layers[li].self_attn for li in m.full_layers}
        handles = [attn_mods[li].register_forward_hook(make_hook(li), with_kwargs=True) for li in m.full_layers]
        m.model(input_ids=torch.tensor([dx_ids + t_ids], device=m.device), logits_to_keep=1)
        for h in handles:
            h.remove()

        edit_t = torch.tensor(edit, device=m.device, dtype=torch.float32)
        prob_t = torch.tensor(prob, device=m.device, dtype=torch.float32)
        res_edit, res_prob, sent_mat = [], [], []
        for li in m.full_layers:
            mod = attn_mods[li]
            hs, (cos, sin) = captured.pop(li)
            q, k = qk_states(mod, hs, cos, sin)
            H = q.shape[0]
            e_mass = torch.zeros(H, T, device=m.device)
            p_mass = torch.zeros(H, T, device=m.device)
            smat = torch.zeros(H, S, S, device=m.device)
            for r0 in range(0, T, args.attn_chunk):
                r1 = min(T, r0 + args.attn_chunk)
                rows = torch.arange(P + r0, P + r1, device=m.device)
                scores = q[:, P + r0:P + r1] @ k[:, :P + r1].transpose(1, 2) * mod.scaling
                causal = torch.arange(P + r1, device=m.device)[None, :] > rows[:, None]
                probs = torch.softmax(scores.masked_fill(causal, float("-inf")), dim=-1)  # [H, c, P+r1]
                e_mass[:, r0:r1] = probs[:, :, :P] @ edit_t
                p_mass[:, r0:r1] = probs[:, :, :P] @ prob_t
                to_sent = probs[:, :, P:P + r1] @ key_avg[:r1]  # [H, c, S] mean attention to each sentence's tokens
                q_sent = torch.tensor(sent_of[r0:r1], device=m.device)
                smat.index_add_(1, q_sent, to_sent)
            smat /= torch.tensor(sent_len, device=m.device).clamp(min=1)[None, :, None]
            res_edit.append(e_mass.cpu().numpy())
            res_prob.append(p_mass.cpu().numpy())
            sent_mat.append(smat.cpu().numpy())
            del q, k, hs
        e_mass = np.stack(res_edit)  # [Lf, H, T]
        p_mass = np.stack(res_prob)
        smat = np.stack(sent_mat)  # [Lf, H, S, S]

        # receiver heads: kurtosis over key sentences of mean attention received from sentences >= 4 later
        Lf, H = smat.shape[:2]
        kurt = np.full((Lf, H), np.nan)
        for li in range(Lf):
            for h in range(H):
                vert = np.array([smat[li, h, j + 4:, j].mean() if j + 4 < S else np.nan for j in range(S)])
                kurt[li, h] = kurtosis(vert)
        order = np.argsort(-np.nan_to_num(kurt, nan=-1e9), axis=None)[:args.receiver_heads]
        receivers = [(int(i // H), int(i % H)) for i in order]

        frac_edit = edit[prob].mean() if prob.any() else float("nan")
        def per_sentence(mass, heads=None):
            x = mass if heads is None else np.stack([mass[l, h] for l, h in heads])[None]
            x = x.reshape(-1, T).mean(0)
            return [float(x[sent_of == s].mean()) for s in range(S)]
        e_all, p_all = per_sentence(e_mass), per_sentence(p_mass)
        e_rec, p_rec = per_sentence(e_mass, receivers), per_sentence(p_mass, receivers)
        enrich = lambda e, p: [float((a / b) / frac_edit) if b > 0 and frac_edit > 0 else float("nan")
                               for a, b in zip(e, p)]
        anchor_sents = []
        a = anchors.get((sel["group"], "dX"))
        if a:
            feats = curve_features(a["points"], thinking, a["sentence_ends"])
            for anc in feats["anchors"]:
                anchor_sents += list(range(anc["k_from"], min(anc["k_to"], S)))
        anchor_sents = sorted(set(anchor_sents))
        others = [s for s in range(S) if s not in anchor_sents]
        mean_of = lambda xs, idx: float(np.nanmean([xs[i] for i in idx])) if idx else float("nan")
        row = dict(role=sel["role"], dx_key=sel["dx_key"], n_edited=int(edit.sum()), n_problem=int(prob.sum()),
                   edited_tokens=m.tok.convert_ids_to_tokens([t for t, e in zip(dx_ids, edit) if e]),
                   frac_edited=float(frac_edit), n_sentences=S, n_tokens=T,
                   receiver_heads=[dict(layer=m.full_layers[l], head=h, kurtosis=float(kurt[l, h]))
                                   for l, h in receivers],
                   sentence_edit_mass=e_all, sentence_problem_mass=p_all, sentence_enrichment=enrich(e_all, p_all),
                   sentence_edit_mass_receivers=e_rec, sentence_enrichment_receivers=enrich(e_rec, p_rec),
                   anchor_sentences=anchor_sents,
                   anchor_vs_rest=dict(edit_mass=(mean_of(e_all, anchor_sents), mean_of(e_all, others)),
                                       enrichment=(mean_of(enrich(e_all, p_all), anchor_sents),
                                                   mean_of(enrich(e_all, p_all), others)),
                                       enrichment_receivers=(mean_of(enrich(e_rec, p_rec), anchor_sents),
                                                             mean_of(enrich(e_rec, p_rec), others))),
                   per_layer_edit_mass=[float(e_mass[l].mean()) for l in range(Lf)])
        np.savez_compressed(WORK / f"wb_attn_{sel['group']}.npz", edit_mass=e_mass.astype(np.float16),
                            problem_mass=p_mass.astype(np.float16), sentence_matrix=smat.astype(np.float16),
                            kurtosis=kurt, full_layers=np.array(m.full_layers))
        out[sel["group"]] = row
        av = row["anchor_vs_rest"]
        print(f"[attn] {sel['role']} {sel['group']} dX={sel['dx_key']}: {row['n_edited']} edited tokens; "
              f"enrichment anchors {av['enrichment'][0]:.2f} vs rest {av['enrichment'][1]:.2f}; "
              f"receivers {av['enrichment_receivers'][0]:.2f} vs {av['enrichment_receivers'][1]:.2f}", flush=True)
        torch.cuda.empty_cache()
    (WORK / "wb_attn.json").write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- analysis 6
@torch.no_grad()
def run_steer(m: Model, args, selection, anchors) -> dict:
    out = {}
    targets = [s for s in selection if s["role"] in args.steer_roles]
    for sel in targets:
        layer, u_np, mid, _ = probe_direction(args.model, sel["group"], args.layer_frac)
        u = torch.tensor(u_np, device=m.device, dtype=torch.float32)
        thinking = thinking_of(sel["dx_trace"])
        ends = sentence_ends(thinking)
        a = anchors.get((sel["group"], "dX"))
        locations = {}
        if "start" in args.steer_locations:
            locations["start"] = 0
        if "anchor" in args.steer_locations:
            k = 0
            if a:
                feats = curve_features(a["points"], thinking, a["sentence_ends"])
                ks, ps = feats["k"], feats["p"]
                lock = next((i for i in range(len(ps)) if all(p >= 0.8 for p in ps[i:])), None)
                k = ks[max(0, lock - 1)] if lock else ks[int(np.argmax(np.abs(np.diff(ps))))] if len(ps) > 1 else 0
            if k > 0 or "start" not in locations:
                locations["anchor"] = k
        p_ids = m.prompt_ids(sel["dx_text"])
        stop_ids = sorted({m.tok.eos_token_id, m.tok.convert_tokens_to_ids("<|im_end|>"),
                           m.tok.convert_tokens_to_ids("<|endoftext|>")} - {None})
        res = {}
        for loc, k in locations.items():
            prefix = thinking[:ends[k - 1]] if k > 0 else ""
            ids = p_ids + m.tok(prefix, add_special_tokens=False)["input_ids"]
            store = {}
            handle = m.residual_hook(layer - 1, store, "h", slice(4, None))  # skip the high-norm first tokens
            m.model(input_ids=torch.tensor([ids], device=m.device), logits_to_keep=1)
            handle.remove()
            norm = float(store["h"].float().norm(dim=-1).median())
            gen = torch.Generator(device="cpu").manual_seed(0)
            rand = torch.randn(len(u_np), generator=gen).to(m.device)
            rand /= rand.norm()
            conditions = [(f"probe_a{al}", -u, al) for al in args.alphas]
            conditions.append((f"random_a{max(args.alphas)}", rand, max(args.alphas)))
            res[loc] = dict(k=k, prefix_tokens=len(ids) - len(p_ids), mean_norm=norm, conditions={})
            for name, direction, alpha in conditions:
                vec = (alpha * norm * direction).to(torch.bfloat16)
                readout = []

                def hook(_module, _args, output):
                    h = output[0] if isinstance(output, tuple) else output
                    readout.append(float((h[:, -1].float() @ u).mean() - mid))
                    h[:, -1] += vec
                    return output
                handle = m.layers[layer - 1].register_forward_hook(hook)
                torch.manual_seed(args.seed)
                inp = torch.tensor([ids] * args.n, device=m.device)
                max_new = max(16, min(args.budget - (len(ids) - len(p_ids)), args.steer_max_new))
                t0 = time.time()
                gen_ids = m.model.generate(inp, attention_mask=torch.ones_like(inp), do_sample=True, temperature=1.0,
                                           top_p=1.0, top_k=0, max_new_tokens=max_new, eos_token_id=stop_ids,
                                           pad_token_id=m.tok.pad_token_id or m.tok.eos_token_id)
                handle.remove()
                new = gen_ids[:, inp.shape[1]:]
                texts = m.tok.batch_decode(new, skip_special_tokens=True)
                finished = [bool(torch.isin(g, torch.tensor(stop_ids, device=g.device)).any()) for g in new]
                answers = [extract_answer(t) if f else None for t, f in zip(texts, finished)]
                correct = [a_ is not None and a_ == sel["gold"] for a_ in answers]
                res[loc]["conditions"][name] = dict(alpha=alpha, acc=float(np.mean(correct)), answers=answers,
                                                    finished=float(np.mean(finished)),
                                                    probe_readout=float(np.mean(readout[1:])) if len(readout) > 1 else None,
                                                    seconds=time.time() - t0)
                print(f"[steer] {sel['role']} {sel['group']} @{loc}(k={k}) {name}: acc {np.mean(correct):.2f} "
                      f"finished {np.mean(finished):.2f} probe readout {res[loc]['conditions'][name]['probe_readout']} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                (WORK / "wb_steer.json").write_text(json.dumps({**out, sel["group"]: dict(role=sel["role"], locations=res)}, indent=1))
        out[sel["group"]] = dict(role=sel["role"], layer=layer, locations=res)
    (WORK / "wb_steer.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", default=["probe", "lens", "attn", "steer"],
                        choices=["probe", "lens", "attn", "steer"])
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--layer-frac", type=float, default=0.5)
    parser.add_argument("--max-trace-tokens", type=int, default=16384)
    parser.add_argument("--lens-fracs", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--lens-steps", type=int, default=300, help="Tuned-lens training steps (0: logit lens only).")
    parser.add_argument("--lens-lr", type=float, default=1e-3)
    parser.add_argument("--lens-seq", type=int, default=2048)
    parser.add_argument("--lens-positions", type=int, default=256)
    parser.add_argument("--attn-max-tokens", type=int, default=8192)
    parser.add_argument("--attn-chunk", type=int, default=256)
    parser.add_argument("--receiver-heads", type=int, default=16)
    parser.add_argument("--steer-roles", nargs="+", default=["robust"])
    parser.add_argument("--steer-locations", nargs="+", default=["start", "anchor"], choices=["start", "anchor"])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0],
                        help="Steering strength as a multiple of the median residual norm at the probe layer.")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--budget", type=int, default=16384)
    parser.add_argument("--steer-max-new", type=int, default=12288)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    selection = load_selection()
    anchors = load_anchor_rows()
    m = Model(args.model)
    if "probe" in args.stages:
        run_probe(m, args, selection, anchors)
    if "lens" in args.stages:
        run_lens(m, args, selection)
    if "attn" in args.stages:
        if m.model_type.startswith("qwen3_5"):  # qk_states mirrors Qwen3.5's gated attention only
            run_attn(m, args, selection, anchors)
        else:
            print(f"[attn] skipped: qk_states supports Qwen3.5 only (model_type {m.model_type})", flush=True)
    if "steer" in args.stages:
        run_steer(m, args, selection, anchors)


if __name__ == "__main__":
    main()
