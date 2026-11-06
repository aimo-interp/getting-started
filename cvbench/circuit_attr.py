#!/usr/bin/env python3
"""Which components write the seg-probe direction? (Skywork-OR1-Math-7B, 2026-11-02)

The seg probe reads the residual stream after layer 14 of 28 (hidden_states[14], layer frac .5), averaged over
response deciles 0-4 (seg_0_5) and 5-8 (seg_5_9) of the greedy 4k trace, and projects it on the
robust-minus-fragile diff-in-means direction d. That residual is exactly

    hs[14] = embed + sum_{l<14} (attn_out_l + mlp_out_l),   attn_out_l = sum_h W_O[:, h] z_{l,h}

so the probe score splits additively over 14 x 28 heads + 14 MLPs + the embedding.

* ``collect`` (GPU): recompute each trace exactly as generate.py run_segs did, stop after layer 13, and save per
  case: per-decile means of every head's z (pre-o_proj, [10, 14, 28, 128]), every MLP output ([10, 14, d]), the
  embedding ([10, d]), hs[14] (to check the decomposition), and per-token projections of every component on the
  full-data directions (for token-level reading) -> data/cvbench/circuit/<model>/<group>.npz
* ``analyze`` (CPU): share of the robust-fragile gap per component (full-data direction), honest per-component
  AUC with out-of-fold directions, where along depth the signal emerges, and the tokens the top components fire on.

    python cvbench/circuit_attr.py collect     # cvbench/circuit.slurm
    python cvbench/circuit_attr.py analyze
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "cvbench"), str(ROOT / "solutions" / "uncertainty-profiling")]
import cv  # noqa: E402

MODEL = "Skywork/Skywork-OR1-Math-7B"
LAYER = 14            # hidden_states index read by the probe: whitebox.layer_of(29, .5)
SEGMENTS = 10
WINDOWS = {"w05": (0, 5), "w59": (5, 9)}
OUT = cv.DATA / "circuit" / cv.safe(MODEL)


def skywork_cases():
    c = cv.load_cases(list(cv.CASE_FILES))
    return c[c.model_id == MODEL]


def seg_windows(group):
    seg = np.load(cv.DATA / "segs" / cv.safe(MODEL) / f"{group}.npz")["seg_mean"].astype(np.float32)  # [S, L+1, d]
    return {w: seg[a:b, LAYER].mean(0) for w, (a, b) in WINDOWS.items()}


def full_directions(cases):
    y = cases.is_robust.to_numpy(bool)
    H = {w: np.stack([seg_windows(g)[w] for g in cases.group]) for w in WINDOWS}
    out = {}
    for w, X in H.items():
        d = X[y].mean(0) - X[~y].mean(0)
        out[w] = d / np.linalg.norm(d)
    return out


# --------------------------------------------------------------------------- collect
class Stop(Exception):
    pass


def cmd_collect(args):
    import torch
    from uncertainty_profile.extraction import format_problem, load_model_and_tokenizer

    cases = skywork_cases()
    dirs = full_directions(cases)
    greedy = {r["group"]: r for r in cv.read_jsonl(cv.DATA / "greedy" / f"{cv.safe(MODEL)}.jsonl")}
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [g for g in cases.group if not (OUT / f"{g}.npz").is_file()]
    print(f"[collect] {len(todo)} to do", flush=True)
    if not todo:
        return
    model, tok = load_model_and_tokenizer(MODEL, local_files_only=True)
    model.eval()
    layers = model.model.layers
    nH, hd = model.config.num_attention_heads, model.config.hidden_size // model.config.num_attention_heads
    dev = model.get_input_embeddings().weight.device
    D = torch.tensor(np.stack([dirs[w] for w in WINDOWS]), device=dev, dtype=torch.float32)  # [2, d]
    # per-head readout of each direction through W_O: u[l, h] = W_O[:, h]^T d  -> [2, nH, hd]
    U = [torch.einsum("kd,dhe->khe", D, layers[l].self_attn.o_proj.weight.float().view(-1, nH, hd))
         for l in range(LAYER)]
    cap = {}

    def pre_o(l):
        def f(mod, inp):
            cap[("z", l)] = inp[0][0].float()            # [T, nH*hd]
        return f

    def post_mlp(l):
        def f(mod, inp, out):
            cap[("mlp", l)] = out[0].float()            # [T, d]
        return f

    def stop(mod, inp, out):
        cap["resid"] = (out[0] if isinstance(out, tuple) else out)[0].float()
        raise Stop

    hooks = [layers[l].self_attn.o_proj.register_forward_pre_hook(pre_o(l)) for l in range(LAYER)]
    hooks += [layers[l].mlp.register_forward_hook(post_mlp(l)) for l in range(LAYER)]
    hooks += [layers[LAYER - 1].register_forward_hook(stop)]
    hooks += [model.model.embed_tokens.register_forward_hook(lambda m, i, o: cap.__setitem__("emb", o[0].float()))]
    try:
        for k, g in enumerate(todo, 1):
            problem = cases.loc[cases.group == g, "problem"].iloc[0]
            prompt_ids = tok(format_problem(tok, problem), add_special_tokens=False, truncation=True,
                             max_length=2048)["input_ids"]
            resp_ids = tok(greedy[g]["generated_text"], add_special_tokens=False)["input_ids"][:4096]
            p, n = len(prompt_ids), len(resp_ids)
            cap.clear()
            with torch.no_grad():
                try:
                    model.model(input_ids=torch.tensor([prompt_ids + resp_ids], device=dev))
                except Stop:
                    pass
                bounds = np.linspace(0, n, SEGMENTS + 1).round().astype(int)
                seg = lambda x: torch.stack([x[p + a:p + b].mean(0) for a, b in zip(bounds[:-1], bounds[1:])])
                z = torch.stack([cap[("z", l)].view(-1, nH, hd) for l in range(LAYER)], 1)        # [T, L, nH, hd]
                mlp = torch.stack([cap[("mlp", l)] for l in range(LAYER)], 1)                    # [T, L, d]
                # per-token projections on each direction: heads [2, T, L, nH], mlp [2, T, L], emb [2, T]
                tok_heads = torch.einsum("tlhe,lkhe->ktlh", z[p:], torch.stack(U, 0))
                tok_mlp = torch.einsum("tld,kd->ktl", mlp[p:], D)
                tok_emb = torch.einsum("td,kd->kt", cap["emb"][p:], D)
                rec = dict(
                    z=seg(z).cpu().numpy().astype(np.float16),
                    mlp=seg(mlp).cpu().numpy().astype(np.float16),
                    emb=seg(cap["emb"]).cpu().numpy().astype(np.float16),
                    resid=seg(cap["resid"]).cpu().numpy().astype(np.float16),
                    tok_heads=tok_heads.cpu().numpy().astype(np.float16),
                    tok_mlp=tok_mlp.cpu().numpy().astype(np.float16),
                    tok_emb=tok_emb.cpu().numpy().astype(np.float16),
                    resp_ids=np.array(resp_ids, dtype=np.int32), n_prompt=p)
            np.savez_compressed(OUT / f"{g}.npz", **rec)
            del z, mlp, tok_heads
            cap.clear()
            torch.cuda.empty_cache()
            print(f"  {k}/{len(todo)} {g}: {n} resp tokens", flush=True)
    finally:
        for h in hooks:
            h.remove()
    # W_O per layer, to map head z into the residual offline
    np.save(OUT / "W_O.npy", np.stack([layers[l].self_attn.o_proj.weight.detach().float().cpu().numpy().astype(np.float16)
                                       for l in range(LAYER)]))


# --------------------------------------------------------------------------- analyze
def cmd_analyze(args):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    cases = skywork_cases()
    y = cases.is_robust.to_numpy(bool)
    Z = [np.load(OUT / f"{g}.npz") for g in cases.group]
    if not (OUT / "W_O.npy").is_file():  # read o_proj weights straight from the checkpoint (CPU)
        import glob, json
        from safetensors import safe_open
        snap = glob.glob(f"/data/gpfs/projects/punim0478/bansaab/hub/models--Skywork--Skywork-OR1-Math-7B/snapshots/*")[0]
        wmap = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
        ws = []
        for l in range(LAYER):
            k = f"model.layers.{l}.self_attn.o_proj.weight"
            with safe_open(f"{snap}/{wmap[k]}", "pt") as f:
                ws.append(f.get_tensor(k).float().numpy().astype(np.float16))
        np.save(OUT / "W_O.npy", np.stack(ws))
    W_O = np.load(OUT / "W_O.npy").astype(np.float32)                # [L, d, nH*hd]
    L, d, _ = W_O.shape
    nH = 28
    hd = W_O.shape[2] // nH
    W = W_O.reshape(L, d, nH, hd)
    names = ["emb"] + [f"L{l}.H{h}" for l in range(L) for h in range(nH)] + [f"L{l}.MLP" for l in range(L)]

    def components(w):
        """[n, C, d] window-mean residual-space output of every component (C = 1 + L*nH + L)."""
        a, b = WINDOWS[w]
        out = []
        for z in Z:
            zz = z["z"][a:b].astype(np.float32).mean(0)              # [L, nH, hd]
            heads = np.einsum("ldhe,lhe->lhd", W, zz).reshape(L * nH, d)
            out.append(np.concatenate([z["emb"][a:b].astype(np.float32).mean(0)[None], heads,
                                       z["mlp"][a:b].astype(np.float32).mean(0)], 0))
        return np.stack(out)

    print(f"Skywork n={len(y)} robust={y.sum()} fragile={(~y).sum()}; components={len(names)}")
    report = {}
    for w in WINDOWS:
        C = components(w)                                            # [n, Cn, d]
        recon = C.sum(1)
        stored = np.stack([seg_windows(g)[w] for g in cases.group])
        rel = np.linalg.norm(recon - stored, axis=1) / np.linalg.norm(stored, axis=1)
        print(f"\n===== window {w}: decomposition check, rel err vs stored segs median {np.median(rel):.4f} max {rel.max():.4f}")
        # full-data direction: share of the gap
        dvec = recon[y].mean(0) - recon[~y].mean(0)
        dvec /= np.linalg.norm(dvec)
        S = C @ dvec                                                 # [n, Cn]
        gap = S[y].mean(0) - S[~y].mean(0)
        tot = gap.sum()
        # out-of-fold direction per case (5-fold x 10 seeds), per-component scores -> AUC
        oof = np.zeros((10, len(y), C.shape[1]))
        for s in range(10):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(y, y):
                dv = recon[tr][y[tr]].mean(0) - recon[tr][~y[tr]].mean(0)
                dv /= np.linalg.norm(dv)
                oof[s, te] = C[te] @ dv
        auc = np.array([np.mean([roc_auc_score(y, oof[s, :, k]) for s in range(10)]) for k in range(C.shape[1])])
        auc_total = np.mean([roc_auc_score(y, oof[s].sum(1)) for s in range(10)])
        # depth profile: cumulative residual up to layer l, OOF AUC
        layer_of_comp = np.array([-1] + [l for l in range(L) for _ in range(nH)] + list(range(L)))
        cum_auc = []
        for l in range(L):
            m = layer_of_comp <= l
            cum_auc.append(np.mean([roc_auc_score(y, oof[s][:, m].sum(1)) for s in range(10)]))
        per_layer_attn = [gap[(layer_of_comp == l)][:nH].sum() / tot for l in range(L)]
        per_layer_mlp = [gap[1 + L * nH + l] / tot for l in range(L)]
        print(f"total gap (full-data dir) {tot:.3f}; OOF AUC of full sum {auc_total:.3f}")
        print("per layer share of gap  attn: " + " ".join(f"{v:+.2f}" for v in per_layer_attn))
        print("                         mlp: " + " ".join(f"{v:+.2f}" for v in per_layer_mlp))
        print("                         emb: " + f"{gap[0] / tot:+.2f}")
        print("OOF AUC of residual up to layer l: " + " ".join(f"{v:.2f}" for v in cum_auc))
        attn_share = sum(per_layer_attn)
        print(f"attention total {attn_share:+.2f}, MLP total {sum(per_layer_mlp):+.2f}")
        order = np.argsort(-np.abs(gap))
        conc = np.cumsum(np.abs(gap[order])) / np.abs(gap).sum()
        print(f"concentration: top 5 comps {conc[4]:.2f}, top 20 {conc[19]:.2f}, top 50 {conc[49]:.2f} of |gap|")
        print(f"{'component':10s} {'gap share':>9s} {'OOF AUC':>8s}")
        for k in order[:20]:
            print(f"{names[k]:10s} {gap[k] / tot:+9.3f} {auc[k]:8.3f}")
        report[w] = dict(order=order, gap=gap, tot=tot, auc=auc, names=names)

    # token level: which response tokens do the top components fire on (full-data direction, per token)?
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    for wi, w in enumerate(WINDOWS):
        r = report[w]
        top = [k for k in r["order"][:6] if r["names"][k] != "emb"][:5]
        print(f"\n===== tokens, window {w} (per-token projection on full-data direction; + = robust-looking)")
        for k in top:
            nm = r["names"][k]
            vals, toks, lab = [], [], []
            for z, yy in zip(Z, y):
                if nm.endswith("MLP"):
                    v = z["tok_mlp"][wi][:, int(nm[1:].split(".")[0])]
                else:
                    l, h = int(nm[1:].split(".")[0]), int(nm.split(".H")[1])
                    v = z["tok_heads"][wi][:, l, h]
                n = len(z["resp_ids"])
                a, b = (np.array(WINDOWS[w]) * n / SEGMENTS).round().astype(int)
                vals.append(v[a:b].astype(np.float32)); toks.append(z["resp_ids"][a:b]); lab.append(np.full(b - a, yy))
            v, t, lb = np.concatenate(vals), np.concatenate(toks), np.concatenate(lab)
            df = __import__("pandas").DataFrame({"v": v, "t": t, "rob": lb})
            g = df.groupby("t").agg(n=("v", "size"), mean=("v", "mean"), tot=("v", "sum"), rob=("rob", "mean"))
            g = g[g.n >= 30]
            spread = np.abs(v).mean()
            pos = g.sort_values("mean").tail(8)
            neg = g.sort_values("mean").head(8)
            fmt = lambda G: ", ".join(f"{tok.decode([i])!r}({m / spread:+.1f})" for i, m in zip(G.index, G["mean"]))
            print(f"{nm:9s} gap share {r['gap'][k] / r['tot']:+.2f}  mean|v|={spread:.3f}")
            print(f"   robust-pushing tokens: {fmt(pos[::-1])}")
            print(f"   fragile-pushing tokens: {fmt(neg)}")
            # is the gap from WHICH tokens appear (composition) or how the component treats the same tokens?
            fr, rb = df[~df.rob.astype(bool)], df[df.rob.astype(bool)]
            print(f"   mean per token: robust {rb.v.mean():+.4f} fragile {fr.v.mean():+.4f}")


# --------------------------------------------------------------------------- attention patterns
FOCUS = ["L13.H4", "L11.H24", "L13.H20", "L10.H19", "L6.H9", "L13.H17", "L13.H16", "L11.H27",
         "L13.H18", "L11.H22", "L10.H14"]          # top OOF-AUC heads (first 8 robust-writing, last 3 inverted)
REGIONS = ["sink", "template", "problem", "cot_far", "cot_recent"]
RECENT = 64
ATT = OUT.parent.parent / "circuit_attn" / cv.safe(MODEL)


def cmd_attn(args):
    """Eager pass over prompt + greedy trace; per head and response decile, the attention mass on each region
    (BOS sink, chat template, problem text, CoT older than 64 tokens, last 64 CoT tokens), plus for FOCUS heads
    the attention each problem token receives -> data/cvbench/circuit_attn/<model>/<group>.npz"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from uncertainty_profile.extraction import format_problem

    cases = skywork_cases()
    greedy = {r["group"]: r for r in cv.read_jsonl(cv.DATA / "greedy" / f"{cv.safe(MODEL)}.jsonl")}
    ATT.mkdir(parents=True, exist_ok=True)
    todo = [g for g in cases.group if not (ATT / f"{g}.npz").is_file()]
    print(f"[attn] {len(todo)} to do", flush=True)
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16,
                                                 attn_implementation="eager", device_map="cuda:0").eval()
    layers = model.model.layers
    focus = [(int(f[1:].split(".")[0]), int(f.split(".H")[1])) for f in FOCUS]
    st = {}

    def hook(l):
        def f(mod, inp, out):
            A = out[1][0]                                   # [H, T, T] bf16
            p, n, ps, pe = st["p"], st["n"], st["ps"], st["pe"]
            Q = A[:, p:p + n].float()                       # response queries [H, n, T]
            sink = Q[..., 0]
            prob = Q[..., ps:pe].sum(-1)
            templ = Q[..., 1:p].sum(-1) - prob
            C = Q[..., p:].cumsum(-1)                       # [H, n, n]
            i = torch.arange(n, device=Q.device)
            lo = (i - RECENT).clamp(min=0)
            recent = C[:, i, i] - torch.where(i - RECENT - 1 >= 0, C[:, i, (i - RECENT - 1).clamp(min=0)], torch.zeros_like(C[:, i, i]))
            far = C[:, i, i] - recent
            R = torch.stack([sink, templ, prob, far, recent], -1)          # [H, n, 5]
            b = st["bounds"]
            st["reg"][l] = torch.stack([R[:, a:bb].mean(1) for a, bb in zip(b[:-1], b[1:])], 1).cpu().numpy()
            for j, (fl, fh) in enumerate(focus):
                if fl == l:
                    for w, (a, bb) in WINDOWS.items():
                        st["ptok"][w][j] = Q[fh, b[a]:b[bb], ps:pe].mean(0).cpu().numpy()
            del Q, C, R
        return f

    hooks = [layers[l].self_attn.register_forward_hook(hook(l)) for l in range(LAYER)]
    hooks.append(layers[LAYER - 1].register_forward_hook(lambda m, i, o: (_ for _ in ()).throw(Stop())))
    try:
        for k, g in enumerate(todo, 1):
            problem = cases.loc[cases.group == g, "problem"].iloc[0]
            prompt = format_problem(tok, problem)
            enc = tok(prompt, add_special_tokens=False, truncation=True, max_length=2048, return_offsets_mapping=True)
            prompt_ids, offs = enc["input_ids"], enc["offset_mapping"]
            c0 = prompt.find(problem)
            c1 = c0 + len(problem)
            inside = [j for j, (a, b) in enumerate(offs) if a >= c0 and b <= c1 and b > a]
            resp_ids = tok(greedy[g]["generated_text"], add_special_tokens=False)["input_ids"][:4096]
            p, n = len(prompt_ids), len(resp_ids)
            st.update(p=p, n=n, ps=inside[0], pe=inside[-1] + 1, reg={},
                      bounds=np.linspace(0, n, SEGMENTS + 1).round().astype(int),
                      ptok={w: {} for w in WINDOWS})
            with torch.no_grad():
                try:
                    model.model(input_ids=torch.tensor([prompt_ids + resp_ids], device="cuda:0"))
                except Stop:
                    pass
            np.savez_compressed(ATT / f"{g}.npz",
                                regions=np.stack([st["reg"][l] for l in range(LAYER)]).astype(np.float32),  # [L,H,10,5]
                                ptok_w05=np.stack([st["ptok"]["w05"][j] for j in range(len(focus))]).astype(np.float32),
                                ptok_w59=np.stack([st["ptok"]["w59"][j] for j in range(len(focus))]).astype(np.float32),
                                problem_ids=np.array(prompt_ids[st["ps"]:st["pe"]], dtype=np.int32))
            torch.cuda.empty_cache()
            print(f"  {k}/{len(todo)} {g}: prompt {p} (problem {st['pe'] - st['ps']}) resp {n}", flush=True)
    finally:
        for h in hooks:
            h.remove()


def cmd_attn_analyze(args):
    from sklearn.metrics import roc_auc_score
    from transformers import AutoTokenizer
    import pandas as pd
    cases = skywork_cases()
    y = cases.is_robust.to_numpy(bool)
    Z = [np.load(ATT / f"{g}.npz") for g in cases.group]
    R = np.stack([z["regions"] for z in Z])                  # [n, L, H, 10, 5]
    L, H = R.shape[1], R.shape[2]
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    # attribution AUC per head from the component analysis (window-averaged), if available
    for w, (a, b) in WINDOWS.items():
        Rw = R[:, :, :, a:b].mean(3)                         # [n, L, H, 5]
        print(f"\n===== window {w}: attention mass by region, mean over all 392 heads: "
              + " ".join(f"{r}={Rw[..., i].mean():.3f}" for i, r in enumerate(REGIONS)))
        print("robust vs fragile, all heads pooled: " + " ".join(
            f"{r}: R {Rw[y][..., i].mean():.3f} F {Rw[~y][..., i].mean():.3f}" for i, r in enumerate(REGIONS)))
        # heads whose region mass itself separates robust/fragile
        auc = np.array([[[roc_auc_score(y, Rw[:, l, h, i]) for i in range(5)] for h in range(H)] for l in range(L)])
        print("heads where a region mass predicts robustness (|AUC-.5| largest), AUC>.5 = more mass -> robust:")
        flat = [(abs(auc[l, h, i] - .5), l, h, i) for l in range(L) for h in range(H) for i in range(5)]
        for _, l, h, i in sorted(flat, reverse=True)[:12]:
            print(f"   L{l}.H{h:<2d} {REGIONS[i]:10s} AUC {auc[l, h, i]:.2f}  mass R {Rw[y, l, h, i].mean():.3f} F {Rw[~y, l, h, i].mean():.3f}")
        print(f"(null: with 392x5 tests at n=48, |AUC-.5| > .25 by chance is common; treat as hypotheses)")
        print("FOCUS heads: region profile (mean over cases) | robust-minus-fragile")
        for j, f in enumerate(FOCUS):
            l, h = int(f[1:].split(".")[0]), int(f.split(".H")[1])
            prof = Rw[:, l, h].mean(0)
            diff = Rw[y, l, h].mean(0) - Rw[~y, l, h].mean(0)
            print(f"   {f:8s} " + " ".join(f"{r}={v:.2f}" for r, v in zip(REGIONS, prof)) + " | "
                  + " ".join(f"{d:+.3f}" for d in diff))
        print("FOCUS heads: problem tokens receiving most attention (pooled over cases, per-token mean share)")
        for j, f in enumerate(FOCUS):
            rows = []
            for z, yy in zip(Z, y):
                v = z[f"ptok_{w}"][j]
                v = v / (v.sum() + 1e-9)
                rows += [(t, x, yy) for t, x in zip(z["problem_ids"], v)]
            df = pd.DataFrame(rows, columns=["t", "v", "rob"])
            g = df.groupby("t").agg(n=("v", "size"), v=("v", "mean"))
            g = g[g.n >= 8].sort_values("v", ascending=False).head(10)
            print(f"   {f:8s} " + ", ".join(f"{tok.decode([int(t)])!r}({v:.3f})" for t, v in zip(g.index, g.v)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "analyze", "attn", "attn_analyze"])
    a = ap.parse_args()
    {"collect": cmd_collect, "analyze": cmd_analyze, "attn": cmd_attn, "attn_analyze": cmd_attn_analyze}[a.cmd](a)
