#!/usr/bin/env python3
"""Stack cvbench signal blocks in one logistic (autoresearch loop, 2026-10-30).

Each block yields per-case feature columns; supervised blocks (diff-in-means probes) are
fitted on the training folds only. Blocks:

* ``seg``: seg_probe 0-5 / 5-9 on the greedy 4k trace (``segs/``).
* ``traj``: seg_traj step_mean + norm_slope on the greedy trace (label-free).
* ``tseg``: seg_probe 0-5 / 5-9 on the 8 sampled 16k traces (``tsegs/``), activations averaged
  over samples (= the mean of the per-sample projections).
* ``ttraj``: seg_traj features per sampled trace, averaged over samples (label-free).
* ``tx``: transcript-probe features (pre-answer length, anchor, answer agreement, length) of the
  same 8 samples (label-free).
* ``lb``: Lookback Lens ratio, windows 0-5 / 5-9 (label-free).

cv.py conventions: z-score within model, logistic C=0.3, 20x5-fold StratifiedGroupKFold by problem.

    uv run cvbench/stack.py --blocks seg traj        # the seg_traj row
    uv run cvbench/stack.py --blocks tseg tx
"""

from __future__ import annotations

import argparse
from collections import Counter
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv  # noqa: E402
import traj  # noqa: E402
import whitebox as wb  # noqa: E402

ROOT = cv.ROOT
C = 0.1
TWINDOWS = ((0, 5), (5, 9))
COMMIT_FEATS = ["c_mean", "c_commit"]  # commit block: chosen before seeing labels
TX_VAR = False  # tx: add agree*(1-agree)


def tsegs_file(model_id, group):
    return cv.DATA / "tsegs" / cv.safe(model_id) / f"{group}.npz"


@lru_cache(None)
def _tsegs(model_id, group, budget):
    z = np.load(tsegs_file(model_id, group))
    return z[f"seg_mean_{budget}"].astype(np.float32), z["n_resp"]  # [8, 10, L+1, d]


def load_tsegs(cases, budget=16384, samples=None):
    """wb.diffmean_scores-style acts: {"t_a_b": [L+1, d]} averaged over the chosen samples."""
    acts = []
    for c in tqdm(list(cases.itertuples()), desc="load tsegs", leave=False):
        S, _ = _tsegs(c.model_id, c.group, budget)
        S = S if samples is None else S[list(samples)]
        acts.append({f"t_{a}_{b}": S[:, a:b].mean((0, 1)) for a, b in TWINDOWS})
    return acts


def ttraj_feats(cases, budget=16384):
    rows = []
    for c in cases.itertuples():
        S, _ = _tsegs(c.model_id, c.group, budget)
        per = [traj.traj_features(s.astype(np.float64)) for s in S]
        rows.append(pd.DataFrame(per).mean().to_dict())
    return pd.DataFrame(rows, index=cases.index)[traj.FEATS]


TXP_FEATS = ["pre_answer_prov", "anchor", "agree_prov", "length"]   # unfinished samples -> provisional answer


def txp_feats(cases, budget=16384, n=8):
    sys.path.insert(0, str(ROOT / "solutions" / "transcript-probe" / "research"))
    import train as tp
    tcases = {(c["model_id"], c["group"]): c for c in tp.load_cases(oss=True)}
    rows = [tp.case_features(tcases[(c.model_id, c.group)], budget, n) if (c.model_id, c.group) in tcases else {}
            for c in cases.itertuples()]
    return pd.DataFrame(rows, index=cases.index)[TXP_FEATS]


TXX_EXTRA = ["answered", "n_distinct", "ans_entropy", "len_sd", "agree_prov", "fin_len", "first_frac"]


def txx_feats(cases, budget=16384, n=8, extra=None):
    """tx features plus extra transcript statistics (loop 2026-10-31): answered share, distinct answers and
    their entropy, sd of log length, provisional agreement, mean log length of finished samples, and where the
    eventual answer first appears as a fraction of the reasoning."""
    import math
    from collections import Counter
    sys.path.insert(0, str(ROOT / "solutions" / "transcript-probe" / "research"))
    import train as tp
    from transcript_probe.features import reasoning_of
    tcases = {(c["model_id"], c["group"]): c for c in tp.load_cases(oss=True)}
    rows = []
    for c in cases.itertuples():
        key = (c.model_id, c.group)
        if key not in tcases:
            rows.append({})
            continue
        tc = tcases[key]
        f = tp.case_features(tc, budget, n)
        x = tc["screen"]
        toks = [min(int(t), budget) for t in x["n_tokens"][:n]]
        ans = [a for a in f["answers"] if a is not None]
        cnt = Counter(ans)
        ent = -sum(v / len(ans) * math.log(v / len(ans)) for v in cnt.values()) if ans else math.log(n)
        fin = [math.log(t + 1) for t, a in zip(toks, f["answers"]) if a is not None]
        fracs = []
        for text, a, t in zip(x["texts"][:n], f["answers"], x["n_tokens"][:n]):
            if a is None:
                continue
            text = tp.clean(text)
            t = int(t)
            if t > budget:
                text = text[: int(budget * len(text) / max(t, 1))]
            think = reasoning_of(text)
            import re
            hit = re.search(r"(?<!\d)%s(?!\d)" % re.escape(str(a)), think)
            fracs.append((hit.start() if hit else len(think)) / max(len(think), 1))
        rows.append({**{k: f[k] for k in list(tp.FEATURES) + ["length"]},
                     "answered": len(ans) / n, "n_distinct": len(cnt), "ans_entropy": ent,
                     "len_sd": float(np.std([math.log(t + 1) for t in toks])), "agree_prov": f["agree_prov"],
                     "fin_len": float(np.mean(fin)) if fin else float(np.log(budget + 1)),
                     "first_frac": float(np.mean(fracs)) if fracs else 1.0})
    cols = list(tp.FEATURES) + ["length"] + (extra or TXX_EXTRA)
    return pd.DataFrame(rows, index=cases.index)[cols]


def gagree_feats(cases, budget=16384, n=8):
    """Greedy 4k trace vs the 8 samples: share of samples (final or provisional answers) that give the greedy
    trace's final-or-provisional answer, and whether the greedy trace has any answer at all."""
    import json
    sys.path.insert(0, str(ROOT / "solutions" / "transcript-probe" / "research"))
    import re
    import train as tp
    from transcript_probe.features import extract_answer, provisional_answer
    tcases = {(c["model_id"], c["group"]): c for c in tp.load_cases(oss=True)}
    greedy = {}
    for f in (cv.DATA / "greedy").glob("*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            greedy[(r["model_id"], r["group"])] = r.get("generated_text") or ""
    rows = []
    for c in cases.itertuples():
        key = (c.model_id, c.group)
        if key not in tcases or key not in greedy:
            rows.append({})
            continue
        f = tp.case_features(tcases[key], budget, n)
        in_problem = set(re.findall(r"\d+", c.problem))
        g = extract_answer(greedy[key]) or provisional_answer(greedy[key], in_problem)
        prov = f["provisional"]
        rows.append({"g_share": float(np.mean([a == g for a in prov])) if g is not None else 0.0,
                     "g_has": float(g is not None)})
    return pd.DataFrame(rows, index=cases.index, columns=["g_share", "g_has"])


def tx_feats(cases, budget=16384, n=8):
    sys.path.insert(0, str(ROOT / "solutions" / "transcript-probe" / "research"))
    import train as tp
    tcases = {(c["model_id"], c["group"]): c for c in tp.load_cases(oss=True)}
    rows = []
    for c in cases.itertuples():
        key = (c.model_id, c.group)
        rows.append(tp.case_features(tcases[key], budget, n) if key in tcases else {})
    T = pd.DataFrame(rows, index=cases.index)[list(tp.FEATURES) + ["length"]]
    if TX_VAR:  # fragile cases sit at intermediate agreement: Bernoulli variance of the majority answer
        T["agree_var"] = T.agree * (1 - T.agree)
    return T


def pert_feats(cases):
    """Self-perturbation (upper-bound study, too slow for the LB): 8x16k samples on our own perturbed
    variants (typos/distract/rephrase/rename, data/anchors screens). Per variant, the share of samples
    giving the original's majority answer; mean and min over variants. Small models only (NaN elsewhere).
    Our variants may be the ones labelgen labels were built from, so score on train-main-v2 only."""
    sys.path.insert(0, str(ROOT / "solutions" / "transcript-probe" / "research"))
    import train as tp
    from collections import Counter
    rows = {}
    for model_id, d in tp.SCREENS.items():
        by = {}
        for r in tp.read_jsonl(d / "screen.jsonl"):
            by.setdefault(r["group"], {})[r["variant"]] = r["answers"]
        for g, v in by.items():
            if "X" not in v:
                continue
            given = [a for a in v["X"] if a is not None]
            maj = Counter(given).most_common(1)[0][0] if given else None
            keep = [np.mean([a == maj for a in ans]) for k, ans in v.items() if k != "X"] if maj is not None else []
            rows[(model_id, g)] = {"pert_keep_mean": np.mean(keep), "pert_keep_min": np.min(keep)} if keep else {}
    return pd.DataFrame([rows.get((c.model_id, c.group), {}) for c in cases.itertuples()], index=cases.index,
                        columns=["pert_keep_mean", "pert_keep_min"])


def prob_feats(cases):
    """Problem text only: length, how many numbers, how much LaTeX."""
    import re
    P = cases.problem
    return pd.DataFrame({"p_logchars": np.log(P.str.len()), "p_numbers": P.map(lambda t: len(re.findall(r"\d+", t))),
                         "p_dollars": P.str.count(r"\$")}, index=cases.index)


def hedge_file(model_id, group):
    return cv.DATA / "hedge" / cv.safe(model_id) / f"{group}.npz"


def load_hedge(cases, site):
    """wb.diffmean_scores-style acts {site: [L+1, d]}: the hedge-localised state, or the CoT mean without its
    wrap-up (seg 0-9) when the trace has no hedge."""
    acts = wb.load_acts(cases, cv)
    out = []
    for c, a in zip(cases.itertuples(), acts):
        f = hedge_file(c.model_id, c.group)
        z = np.load(f) if f.is_file() else {}
        out.append({site: z[site].astype(np.float32) if site in z else a["seg_0_9"]})
    return out


def commit_feats(cases):
    """Early-answer commitment curve (label-free): log p of the final forced answer when the trace is cut at
    10%..100% and closed with the forcing suffix; match = the forced argmax equals the final answer."""
    rows = []
    for c in cases.itertuples():
        f = hedge_file(c.model_id, c.group)
        if not f.is_file():
            rows.append({})
            continue
        z = np.load(f)
        lp, m = z["commit_logp"].astype(float), z["commit_match"].astype(bool)
        settled = np.flatnonzero(~m[:-1])  # last boundary where the answer differed from the final one
        rows.append({"c_mean": lp[:-1].mean(), "c_half": lp[4], "c_early": lp[:3].mean(),
                     "c_commit": (settled[-1] + 1) / len(m) if len(settled) else 0.0,
                     "c_flips": float(np.abs(np.diff(m.astype(int))).sum()), "c_final": lp[-1]})
    return pd.DataFrame(rows, index=cases.index, columns=["c_mean", "c_half", "c_early", "c_commit", "c_flips", "c_final"])


LIREF_SITES = {"liref_last": "last_prompt", "liref_prob": "prob_mean", "liref_cot": "seg_0_5", "liref_cot2": "seg_5_9"}


def liref_feats(cases, frac, sites=("liref_last", "liref_cot", "liref_cot2")):
    """Label-free projection of each case's cached states onto the model's Linear Reasoning Feature
    r = mean h(reasoning prompts) - mean h(memory prompts) at the last prompt token, layer ``frac``
    (Hong et al., 2025; liref/<model>.npz). NaN for models without a contrast set."""
    acts = wb.load_acts(cases, cv)
    dirs = {}
    for m in cases.model_id.unique():
        p = cv.DATA / "liref" / f"{cv.safe(m)}.npz"
        if not p.is_file():
            continue
        z = np.load(p)
        H, kind = z["h_last"].astype(np.float32), z["kind"]
        k = kind == "reasoning"

        def direction(layer):
            r = H[k, layer].mean(0) - H[~k, layer].mean(0)
            return r / (np.linalg.norm(r) + 1e-8)

        def separation(layer):  # d' of the contrast prompts along the layer's own direction (label-free)
            proj = H[:, layer] @ direction(layer)
            return (proj[k].mean() - proj[~k].mean()) / np.sqrt((proj[k].var() + proj[~k].var()) / 2 + 1e-8)
        L = H.shape[1]
        layer = max(range(1, L), key=separation) if frac is None else wb.layer_of(L, frac)
        dirs[m] = (layer, direction(layer))
    rows = []
    for c, a in zip(cases.itertuples(), acts):
        if c.model_id not in dirs:
            rows.append({})
            continue
        layer, r = dirs[c.model_id]
        rows.append({name: float(a[site][layer] @ r) for name, site in LIREF_SITES.items() if site in a})
    return pd.DataFrame(rows, index=cases.index, columns=list(sites))


def lb_feats(cases):
    import lookback
    _, S = lookback.load(cases)
    return S[[f"ratio_{a}_{b}" for a, b in lookback.WINDOWS]]


def crossfit(score, y, groups, tr, inner=5, seed=0):
    """score(tr) -> [n, k] with honest training rows: each training row is scored by a probe fitted on the
    other inner folds of tr, test rows by the probe from all of tr (as lookback.crossfit_scores)."""
    out = score(tr)
    folds = StratifiedGroupKFold(min(inner, len(np.unique(groups[tr]))), shuffle=True, random_state=seed)
    for itr, ite in folds.split(tr, y[tr], groups[tr]):
        out[tr[ite]] = score(tr[itr])[tr[ite]]
    return out


def lda_scores(acts, models, y, tr, site, frac, k=None):
    """Per model: PCA (fit on training rows) to min(n_tr-1, 32) dims, Ledoit-Wolf shrinkage LDA direction,
    projection scaled by the training sd. Same contract as wb.diffmean_scores_layer."""
    from sklearn.covariance import LedoitWolf
    feat = np.zeros(len(acts))
    for m in np.unique(models):
        idx = np.where(models == m)[0]
        tr_m = np.intersect1d(idx, tr)
        y_tr = y[tr_m]
        if y_tr.all() or (~y_tr).all():
            continue
        L = acts[idx[0]][site].shape[0]
        layer = wb.layer_of(L, frac)
        H = np.stack([acts[i][site][layer] for i in idx]).astype(np.float64)
        H_tr = np.stack([acts[i][site][layer] for i in tr_m]).astype(np.float64)
        mu = H_tr.mean(0)
        U, S, Vt = np.linalg.svd(H_tr - mu, full_matrices=False)
        kk = k or min(len(tr_m) - 1, 32)
        P = Vt[:kk]
        Z_tr, Z = (H_tr - mu) @ P.T, (H - mu) @ P.T
        Zc = np.concatenate([Z_tr[y_tr] - Z_tr[y_tr].mean(0), Z_tr[~y_tr] - Z_tr[~y_tr].mean(0)])
        cov = LedoitWolf().fit(Zc).covariance_
        w = np.linalg.solve(cov, Z_tr[y_tr].mean(0) - Z_tr[~y_tr].mean(0))
        proj_tr, proj = Z_tr @ w, Z @ w
        feat[idx] = proj / (proj_tr.std() + 1e-8)
    return feat[:, None]


def pooled_seg_scores(acts, models, y, tr, sites=("seg_0_5", "seg_5_9"), frac=0.5):
    """seg probe with the DIM direction fitted per *base* model (gpt-oss efforts pooled: one model, one hidden
    space, 37 cases instead of 7-15), centred and scaled per effort on its own training rows."""
    base = np.array([m.split(":")[0] for m in models])
    out = np.zeros((len(y), len(sites)))
    for j, site in enumerate(sites):
        for b in np.unique(base):
            idx = np.where(base == b)[0]
            tr_b = np.intersect1d(idx, tr)
            yb = y[tr_b]
            if yb.all() or (~yb).all():
                continue
            L = acts[idx[0]][site].shape[0]
            layer = wb.layer_of(L, frac)
            # within-effort centring before pooling, so the direction is not the effort offset
            H = np.stack([acts[i][site][layer] for i in idx]).astype(np.float64)
            Hc = H.copy()
            for m in np.unique(models[idx]):
                k = models[idx] == m
                trk = np.isin(idx, tr) & k
                Hc[k] -= H[trk].mean(0) if trk.any() else H[k].mean(0)
            trmask = np.isin(idx, tr)
            d = Hc[trmask & y[idx]].mean(0) - Hc[trmask & ~y[idx]].mean(0)
            d /= np.linalg.norm(d) + 1e-8
            proj = Hc @ d
            for m in np.unique(models[idx]):
                k = models[idx] == m
                sd = proj[k & trmask].std() if (k & trmask).sum() > 1 else proj[k].std()
                out[idx[k], j] = proj[k] / (sd + 1e-8)
    return out


SAFE_MIN_AUC = 0.6


def safe_seg_scores(acts, models, y, groups, tr, sites=("seg_0_5", "seg_5_9"), frac=0.5, min_auc=None, soft=False):
    """seg probe scores with a per-model reliability check (loop 2026-10-31): for each model and window, the
    out-of-fold AUC of the DIM score on that model's *training* rows (grouped 5-fold, 2 seeds). Windows whose
    inner AUC is below ``min_auc`` score 0 for that model (no information) instead of a possibly inverted
    direction; with ``soft`` the score is scaled by 2*(AUC-0.5) clipped to [0, 1]."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    min_auc = SAFE_MIN_AUC if min_auc is None else min_auc
    full = np.concatenate([wb.diffmean_scores(acts, models, y, tr, None, s, frac) for s in sites], 1)
    inner = np.zeros((len(y), len(sites)))
    for seed in range(2):
        for itr, ite in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(tr, y[tr], groups[tr]):
            sc = np.concatenate([wb.diffmean_scores(acts, models, y, tr[itr], None, s, frac) for s in sites], 1)
            inner[tr[ite]] += sc[tr[ite]] / 2
    for m in np.unique(models):
        idx = np.where(models == m)[0]
        tr_m = np.intersect1d(idx, tr)
        for j in range(len(sites)):
            yy = y[tr_m]
            auc = roc_auc_score(yy, inner[tr_m, j]) if 0 < yy.sum() < len(yy) else 0.5
            w = float(np.clip(2 * (auc - 0.5), 0, 1)) if soft else float(auc >= min_auc)
            full[idx, j] *= w
    return full


def make_blocks(cases, names, args):
    """name -> callable(tr) -> [n, k] features."""
    y = cases.is_robust.astype(bool).to_numpy()
    models, groups = cases.model_id.to_numpy(), cases.group.to_numpy()
    out = {}
    for name in names:
        if name == "seg":
            acts = wb.load_acts(cases, cv)
            out[name] = lambda tr, acts=acts: np.concatenate(
                [wb.diffmean_scores(acts, models, y, tr, None, s, args.frac) for s in ("seg_0_5", "seg_5_9")], 1)
        elif name in ("seg_ml", "seg_ml2"):  # DIM score averaged over several layers (fracs)
            acts = wb.load_acts(cases, cv)
            fr = (0.3, 0.4, 0.5, 0.6, 0.7) if name == "seg_ml" else (0.4, 0.5, 0.6)
            out[name] = lambda tr, acts=acts, fr=fr: np.concatenate(
                [np.mean([wb.diffmean_scores(acts, models, y, tr, None, s, f) for f in fr], 0)
                 for s in ("seg_0_5", "seg_5_9")], 1)
        elif name in ("seg_lda", "seg_lda_ml"):  # shrinkage LDA direction in a PCA basis instead of plain DIM
            acts = wb.load_acts(cases, cv)
            fr = (0.5,) if name == "seg_lda" else (0.4, 0.5, 0.6)
            out[name] = lambda tr, acts=acts, fr=fr: np.concatenate(
                [np.mean([lda_scores(acts, models, y, tr, s, f) for f in fr], 0) for s in ("seg_0_5", "seg_5_9")], 1)
        elif name.startswith("segw_"):  # other windows, e.g. segw_0_3+3_7+7_10
            acts = wb.load_acts(cases, cv)
            sites = ["seg_" + w for w in name[5:].split("+")]
            out[name] = lambda tr, acts=acts, sites=sites: np.concatenate(
                [wb.diffmean_scores(acts, models, y, tr, None, s, args.frac) for s in sites], 1)
        elif name in ("seg_safe", "seg_soft", "seg_safe5", "seg_safe7"):
            acts = wb.load_acts(cases, cv)
            kw = {"seg_safe": {}, "seg_soft": {"soft": True}, "seg_safe5": {"min_auc": 0.5},
                  "seg_safe7": {"min_auc": 0.7}}[name]
            out[name] = lambda tr, acts=acts, kw=kw: safe_seg_scores(acts, models, y, groups, tr, frac=args.frac, **kw)
        elif name == "seg_pool":
            acts = wb.load_acts(cases, cv)
            out[name] = lambda tr, acts=acts: pooled_seg_scores(acts, models, y, tr, frac=args.frac)
        elif name == "seg1":  # one window 0-9 (fewer parameters)
            acts = wb.load_acts(cases, cv)
            out[name] = lambda tr, acts=acts: wb.diffmean_scores(acts, models, y, tr, None, "seg_0_9", args.frac)
        elif name == "effort":
            import effort
            eacts, t = effort.load_sites(cases), effort.target_length(cases)
            out[name] = lambda tr, eacts=eacts, t=t: cv.zscore_within_model(
                pd.DataFrame(effort.ridge_scores(eacts, models, t, tr, effort.SITE, effort.FRAC, effort.ALPHA),
                             index=cases.index), cases)
        elif name == "effort2":  # probe trained on the unlabelled effort-stage prompts
            import effort
            eacts = effort.load_sites(cases)
            out[name] = lambda tr, eacts=eacts: cv.zscore_within_model(
                pd.DataFrame(effort.unlabelled_scores(cases, eacts, tr), index=cases.index), cases)
        elif name in ("hedge_at", "hedge_pre", "hedge_post"):
            acts = load_hedge(cases, name)
            out[name] = lambda tr, acts=acts, site=name: wb.diffmean_scores(acts, models, y, tr, None, site, args.frac)
        elif name in ("tseg", "tseg8k", "tseg1", "tseg1_8k"):
            budget = 8192 if name.endswith("8k") else 16384
            acts = load_tsegs(cases, budget, samples=[args.sample] if name.startswith("tseg1") else None)
            out[name] = lambda tr, acts=acts: np.concatenate(
                [wb.diffmean_scores(acts, models, y, tr, None, f"t_{a}_{b}", args.frac) for a, b in TWINDOWS], 1)
        else:
            F = {"traj": lambda: pd.DataFrame([traj.traj_features(s) for s in traj.load_segs(cases)],
                                              index=cases.index)[traj.FEATS],
                 "ttraj": lambda: ttraj_feats(cases),
                 "tx": lambda: tx_feats(cases),
                 "tx4": lambda: tx_feats(cases, 4096), "tx8": lambda: tx_feats(cases, 8192),
                 "txx": lambda: txx_feats(cases),
                 "gagree": lambda: gagree_feats(cases),
                 **{f"txx_{e}": (lambda e=e: txx_feats(cases, extra=[e])) for e in TXX_EXTRA},
                 "txx8": lambda: txx_feats(cases, 8192), "txx4": lambda: txx_feats(cases, 4096),
                 "txp": lambda: txp_feats(cases), "txp4": lambda: txp_feats(cases, 4096),
                 "txp8": lambda: txp_feats(cases, 8192), "txp2": lambda: txp_feats(cases, 2048),
                 "lb": lambda: lb_feats(cases),
                 "pert": lambda: pert_feats(cases),
                 "prob": lambda: prob_feats(cases),
                 "chal": lambda: cv.m_challenge(cases, args)[0],
                 "commit": lambda: commit_feats(cases)[COMMIT_FEATS],
                 "liref": lambda: liref_feats(cases, args.frac),
                 "liref_last": lambda: liref_feats(cases, args.frac, ("liref_last",)),
                 "liref_auto": lambda: liref_feats(cases, None, ("liref_last",)),  # layer by contrast d' per model
                 "liref_cot": lambda: liref_feats(cases, args.frac, ("liref_cot", "liref_cot2")),
                 "diff": lambda: __import__("difficulty").diff_feats(cases, ("diff_last",)),      # E2H ridge @ last prompt token
                 "diff_prob": lambda: __import__("difficulty").diff_feats(cases, ("diff_prob",)),
                 "dim": lambda: __import__("difficulty").diff_feats(cases, ("dim_last",)),         # hardest-minus-easiest projection
                 "diff_all": lambda: __import__("difficulty").diff_feats(cases),
                 "cot": lambda: cv.m_cot_text(cases, args)[0],
                 "len": lambda: cv.m_length(cases, args)[0]}[name]()
            Z = cv.zscore_within_model(F, cases)
            out[name] = lambda tr, Z=Z: Z
            continue
        if args.crossfit:
            out[name] = lambda tr, f=out[name]: crossfit(f, y, groups, tr)
    return out


def run(cases, blocks, repeats, mask=None, avg=False, model_bias=False, tune=False):
    """avg: one logistic per block, probabilities averaged (instead of one stacked logistic).
    model_bias: add one-hot model columns, i.e. a per-model intercept (the model's base rate)."""
    y = cases.is_robust.astype(bool).to_numpy()
    if model_bias:
        onehot = pd.get_dummies(cases.model_id).to_numpy(float)
        blocks = {k: (lambda tr, b=b: np.c_[b(tr), onehot]) for k, b in blocks.items()}

    groups = cases.group.to_numpy()

    def tuned(F, tr, te):
        """Logistic, threshold chosen on inner out-of-fold probabilities of tr to maximise accuracy."""
        oof = np.zeros(len(tr))
        for itr, ite in StratifiedGroupKFold(5, shuffle=True, random_state=1).split(tr, y[tr], groups[tr]):
            oof[ite] = wb._predict(wb._fit_logistic(F[tr[itr]], y[tr[itr]], C), F[tr[ite]], True)[1]
        grid = np.linspace(0.3, 0.7, 41)
        acc = np.array([((oof >= t) == y[tr]).mean() for t in grid])
        t = grid[acc >= acc.max() - 1e-9].mean()
        p = wb._predict(wb._fit_logistic(F[tr], y[tr], C), F[te], True)[1]
        return p >= t, p

    def predict(tr, te):
        Fs = [b(tr) for b in blocks.values()]
        if tune:
            return tuned(np.concatenate(Fs, 1), tr, te)
        if avg:
            p = np.mean([wb._predict(wb._fit_logistic(F[tr], y[tr], C), F[te], True)[1] for F in Fs], 0)
            return p >= 0.5, p
        F = np.concatenate(Fs, 1)
        return wb._predict(wb._fit_logistic(F[tr], y[tr], C), F[te], True)
    return cv.evaluate(cases, predict, repeats, score_mask=mask)


def run_gated(cases, options, repeats, mask=None):
    """options: {name: blocks}. Per model, pick the option with the best inner-CV accuracy on that model's
    training cases (one logistic per option fitted on all models' training rows), then use it for that
    model's test cases."""
    y = cases.is_robust.astype(bool).to_numpy()
    models, groups = cases.model_id.to_numpy(), cases.group.to_numpy()
    picks = Counter()

    def fit_pred(F, tr, te):
        return wb._predict(wb._fit_logistic(F[tr], y[tr], C), F[te], True)[1]

    def predict(tr, te):
        Fs = {k: np.concatenate([b(tr) for b in bl.values()], 1) for k, bl in options.items()}
        oof = {k: np.zeros(len(y)) for k in options}
        for itr, ite in StratifiedGroupKFold(5, shuffle=True, random_state=1).split(tr, y[tr], groups[tr]):
            for k, F in Fs.items():
                Fi = np.concatenate([b(tr[itr]) for b in options[k].values()], 1)
                oof[k][tr[ite]] = fit_pred(Fi, tr[itr], tr[ite])
        p = np.zeros(len(te))
        for m in np.unique(models[te]):
            mtr, mte = tr[models[tr] == m], models[te] == m
            best = max(options, key=lambda k: (((oof[k][mtr] >= 0.5) == y[mtr]).mean(), k == next(iter(options))))
            picks[(m.split("/")[-1], best)] += 1
            p[mte] = fit_pred(Fs[best], tr, te[mte])
        return p >= 0.5, p
    res = cv.evaluate(cases, predict, repeats, score_mask=mask)
    return res, picks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="+", required=True)
    ap.add_argument("--gate", nargs="+", default=None,
                    help="Per-model choice among block sets, each 'a+b' (e.g. tx seg+traj seg+traj+tx).")
    ap.add_argument("--avg", action="store_true")
    ap.add_argument("--tune", action="store_true", help="Threshold from inner-CV accuracy instead of 0.5.")
    ap.add_argument("--crossfit", action="store_true", help="Cross-fit the probe scores of training rows.")
    ap.add_argument("--model-bias", action="store_true", help="Per-model intercept (one-hot model columns).")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--sample", type=int, default=0, help="tseg1: which sampled trace")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--score-sources", nargs="+", default=None)
    ap.add_argument("--cases", nargs="+", default=list(cv.CASE_FILES), help="Case files under data/cvbench.")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    cases = cv.load_cases(args.cases)
    mask = cases.source.isin(args.score_sources).to_numpy() if args.score_sources else None
    blocks = make_blocks(cases, args.blocks, args)
    if args.gate:
        options = {g: {b: blocks[b] for b in g.split("+")} for g in args.gate}
        res, picks = run_gated(cases, options, args.repeats, mask)
        print(f"gate {' | '.join(args.gate):22s} acc={res['cv_score']:.4f}±{res['cv_std']:.4f} auc={res['auc']:.4f} "
              f"bacc={res['balanced_acc']:.4f} | {res['per_model']}")
        print("picks:", dict(sorted(picks.items())))
        print(f"METRIC {res['cv_score']:.4f}")
        return
    res = run(cases, blocks, args.repeats, mask, args.avg, args.model_bias, args.tune)
    tag = ("avg " if args.avg else "") + "+".join(args.blocks) + (" +bias" if args.model_bias else "") + (" crossfit" if args.crossfit else "") + (" tuned" if args.tune else "")
    print(f"{tag:28s} acc={res['cv_score']:.4f}±{res['cv_std']:.4f} auc={res['auc']:.4f} "
          f"bacc={res['balanced_acc']:.4f} | {res['per_model']}")
    print(f"METRIC {res['cv_score']:.4f}")
    if args.write:
        hyper = f"stack {tag}; layer frac {args.frac}; logistic C={C}"
        if args.score_sources:
            hyper += f"; SCORED ON sources={','.join(args.score_sources)} ({int(mask.sum())} cases)"
        cv.write_rows([{"method": "stack", "type": "whitebox", "hyperparams": hyper, "n_cases": len(cases), **res}])


if __name__ == "__main__":
    main()
