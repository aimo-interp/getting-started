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
import re
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


TXE_FEATS = ["anchor_agree", "anchor_churn", "wait_rate", "anchor_top"]


def txe_feats(cases, budget=4096, n=8, base=True):
    """Early-trace transcript features that need no finished sample (loop 2026-10-31), at ``budget`` tokens:
    anchor_agree: share of samples whose most repeated ``= N`` value (not from the problem) is the modal one
    across samples; anchor_churn: mean number of distinct such values per 1k chars; wait_rate: hesitation
    words per 1k chars; anchor_top: mean count of each sample's top value. With ``base`` the tx features too."""
    import re
    from collections import Counter
    sys.path.insert(0, str(ROOT / "solutions" / "transcript-probe" / "research"))
    import train as tp
    from transcript_probe.features import _ANCHOR
    hes = re.compile(r"\b(wait|hmm+|alternatively|actually|let me (?:re-?check|verify|reconsider))\b", re.I)
    tcases = {(c["model_id"], c["group"]): c for c in tp.load_cases(oss=True)}
    rows = []
    for c in cases.itertuples():
        key = (c.model_id, c.group)
        if key not in tcases:
            rows.append({})
            continue
        tc = tcases[key]
        x = tc["screen"]
        in_problem = set(re.findall(r"\d+", c.problem))
        tops, churn, waits, topn = [], [], [], []
        for text, t in list(zip(x["texts"], x["n_tokens"]))[:n]:
            text, t = tp.clean(text), int(t)
            if t > budget:
                text = text[: int(budget * len(text) / max(t, 1))]
            vals = [m.group(1) for m in _ANCHOR.finditer(text) if m.group(1) not in in_problem]
            cnt = Counter(vals)
            k = max(len(text), 1) / 1000
            tops.append(cnt.most_common(1)[0][0] if vals else None)
            topn.append(cnt.most_common(1)[0][1] if vals else 0)
            churn.append(len(cnt) / k)
            waits.append(len(hes.findall(text)) / k)
        given = [v for v in tops if v is not None]
        row = {"anchor_agree": Counter(given).most_common(1)[0][1] / n if given else 0.0,
               "anchor_churn": float(np.mean(churn)), "wait_rate": float(np.mean(waits)), "anchor_top": float(np.mean(topn))}
        if base:
            f = tp.case_features(tc, budget, n)
            row.update({k: f[k] for k in list(tp.FEATURES) + ["length"]})
        rows.append(row)
    cols = (list(tp.FEATURES) + ["length"] if base else []) + TXE_FEATS
    return pd.DataFrame(rows, index=cases.index)[cols]


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


NT_FEATS = ["se", "top_share", "na_rate", "first_tok_entropy"]


def nt_feats(cases, mode="direct", feats=NT_FEATS):
    """No-reasoning semantic entropy (generate.py --stages nothink): answer-cluster entropy, majority share,
    no-answer rate and first answer-token entropy over n samples with the thinking block closed."""
    rows = {}
    for f in (cv.DATA / "nothink").glob(f"*__{mode}.jsonl"):
        for r in cv.read_jsonl(f):
            rows[(r["model_id"], r["group"])] = {k: r[k] for k in NT_FEATS}
    return pd.DataFrame([rows.get((c.model_id, c.group), {}) for c in cases.itertuples()], index=cases.index,
                        columns=NT_FEATS)[list(feats)]


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


def sign_safe(Z, models, y, tr, min_gap=0.0):
    """Zero a feature for a model when its training-row AUC there points the other way from the pooled
    training-row AUC of all other models (a per-model sign flip the shared logistic cannot represent)."""
    from sklearn.metrics import roc_auc_score
    Z = Z.copy()
    for m in np.unique(models):
        tr_m = tr[models[tr] == m]
        tr_o = tr[models[tr] != m]
        if not (0 < y[tr_m].sum() < len(tr_m)):
            continue
        for j in range(Z.shape[1]):
            a_m = roc_auc_score(y[tr_m], Z[tr_m, j]) - 0.5
            a_o = roc_auc_score(y[tr_o], Z[tr_o, j]) - 0.5
            if a_m * a_o < 0 and abs(a_m) > min_gap:
                Z[models == m, j] = 0.0
    return Z


def dim_variant_scores(acts, models, y, tr, site, frac, pcs=0, boot=0, seed=0):
    """Diff-in-means probe score (as wb.diffmean_scores_layer) with optional denoising (loop 2026-10-31):
    ``pcs``: remove the top-k principal components of the training rows first (shared problem variance);
    ``boot``: average the unit direction over that many bootstrap resamples of the training rows."""
    rng = np.random.default_rng(seed)
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
        Hc, Hc_tr = H - mu, H_tr - mu
        if pcs:
            V = np.linalg.svd(Hc_tr, full_matrices=False)[2][:pcs]
            Hc, Hc_tr = Hc - Hc @ V.T @ V, Hc_tr - Hc_tr @ V.T @ V

        def direction(rows):
            d = Hc_tr[rows][y_tr[rows]].mean(0) - Hc_tr[rows][~y_tr[rows]].mean(0)
            return d / (np.linalg.norm(d) + 1e-8)
        if boot:
            ds = []
            for _ in range(boot):
                r = rng.integers(0, len(tr_m), len(tr_m))
                if y_tr[r].all() or (~y_tr[r]).all():
                    continue
                ds.append(direction(r))
            d = np.mean(ds, 0) if ds else direction(np.arange(len(tr_m)))
            d /= np.linalg.norm(d) + 1e-8
        else:
            d = direction(np.arange(len(tr_m)))
        proj, proj_tr = Hc @ d, Hc_tr @ d
        feat[idx] = proj / (proj_tr.std() + 1e-8)
    return feat[:, None]


def pc_scores(acts, models, tr, loglen, sites=("seg_0_5", "seg_5_9"), frac=0.5, k=1):
    """Label-free (loop 2026-10-31): per model and window, projection on the top-k principal components of the
    training rows, each oriented so it correlates positively with the greedy trace's log length (no labels)."""
    out = []
    for site in sites:
        F = np.zeros((len(acts), k))
        for m in np.unique(models):
            idx = np.where(models == m)[0]
            tr_m = np.intersect1d(idx, tr)
            L = acts[idx[0]][site].shape[0]
            layer = wb.layer_of(L, frac)
            H = np.stack([acts[i][site][layer] for i in idx]).astype(np.float64)
            H_tr = np.stack([acts[i][site][layer] for i in tr_m]).astype(np.float64)
            mu = H_tr.mean(0)
            V = np.linalg.svd(H_tr - mu, full_matrices=False)[2][:k]
            P, P_tr = (H - mu) @ V.T, (H_tr - mu) @ V.T
            for j in range(k):
                r = np.corrcoef(P_tr[:, j], loglen[tr_m])[0, 1] if loglen[tr_m].std() > 0 else 0.0
                sgn = 1.0 if not np.isfinite(r) or r == 0 else float(np.sign(r))
                F[idx, j] = sgn * P[:, j] / (P_tr[:, j].std() + 1e-8)
        out.append(F)
    return np.concatenate(out, 1)


def load_reader(cases, reader="Qwen_Qwen3.5-4B"):
    """Reader-model states (cvbench/reader.py): per case {seg_0_5, seg_5_9, seg_0_9, last_prompt} [L+1, d]."""
    out = []
    for c in cases.itertuples():
        z = np.load(cv.DATA / "reader" / reader / cv.safe(c.model_id) / f"{c.group}.npz")
        seg = z["seg_mean"].astype(np.float32)
        out.append({"seg_0_5": seg[0:5].mean(0), "seg_5_9": seg[5:9].mean(0), "seg_0_9": seg[0:9].mean(0),
                    "last_prompt": z["last_prompt"].astype(np.float32)})
    return out


def reader_scores(racts, models, y, tr, sites=("seg_0_5", "seg_5_9"), frac=0.5, pooled=True):
    """One diff-in-means direction on the reader's states. ``pooled``: fitted on all models' training rows
    (each model centred on its own training mean first); else per model. Scaled per model by training sd."""
    out = np.zeros((len(y), len(sites)))
    L = racts[0][sites[0]].shape[0]
    layer = wb.layer_of(L, frac)
    for j, site in enumerate(sites):
        H = np.stack([a[site][layer] for a in racts]).astype(np.float64)
        Hc = H.copy()
        for m in np.unique(models):
            k = models == m
            trk = np.zeros(len(y), bool); trk[tr] = True; trk &= k
            Hc[k] -= H[trk].mean(0) if trk.any() else H[k].mean(0)
        trm = np.zeros(len(y), bool); trm[tr] = True
        groups_ = [np.ones(len(y), bool)] if pooled else [models == m for m in np.unique(models)]
        for g in groups_:
            r = trm & g
            if y[r].all() or (~y[r]).all():
                continue
            d = Hc[r & y].mean(0) - Hc[r & ~y].mean(0)
            d /= np.linalg.norm(d) + 1e-8
            proj = Hc @ d
            for m in np.unique(models[g]):
                k = (models == m) & g
                sd = proj[k & trm].std() if (k & trm).sum() > 1 else proj[k].std()
                out[k, j] = proj[k] / (sd + 1e-8)
    return out


SAFE_MIN_AUC = 0.6


_DROP = None   # cases.max_drop, set in main (continuous robustness target for cdim / rpca probes)


def probe_scores(acts, models, y, tr, site, frac, kind, C):
    """Per-model probe on the same site/layer as the DIM seg probe (drop-in for wb.diffmean_scores).
    kind = <base>[k][w<win>]: base lr (L2 logistic, centred), lrz (standardised), pca<k> (PCA(k) + logistic),
    wdim<k> (whitened DIM in k PCs), dim (DIM), cdim (direction = covariance with -max_drop),
    rpca<k> (ridge on k PCs to -max_drop, alpha = C). w<win>: average the score over layers +-win.
    Score = training-row z-scored."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.decomposition import PCA
    m_ = re.match(r"([a-z]+?)(\d*)(?:w(\d+))?$", kind)
    base, k, win = m_.group(1), int(m_.group(2) or 5), int(m_.group(3) or 0)
    feat = np.zeros(len(acts))
    for m in np.unique(models):
        if m == "__other__":
            continue
        idx = np.where(models == m)[0]
        tr_m = np.intersect1d(idx, tr)
        y_tr = y[tr_m]
        if y_tr.all() or (~y_tr).all():
            continue
        L = acts[idx[0]][site].shape[0]
        acc = np.zeros(len(idx))
        offs = range(-win, win + 1)
        for off in offs:
            layer = min(max(wb.layer_of(L, frac) + off, 0), L - 1)
            H = np.stack([acts[i][site][layer] for i in idx]).astype(np.float64)
            H_tr = np.stack([acts[i][site][layer] for i in tr_m]).astype(np.float64)
            mu = H_tr.mean(0)
            if base == "lrz":
                sd = H_tr.std(0) + 1e-6
            else:
                sd = np.sqrt(((H_tr - mu) ** 2).sum(1).mean()) + 1e-8
            Xtr, X = (H_tr - mu) / sd, (H - mu) / sd
            if base in ("pca", "wdim", "rpca"):
                p = PCA(min(k, len(tr_m) - 1), random_state=0).fit(Xtr)
                Xtr, X = p.transform(Xtr), p.transform(X)
            if base == "dim":
                w = Xtr[y_tr].mean(0) - Xtr[~y_tr].mean(0)
                f, f_tr = X @ w, Xtr @ w
            elif base == "wdim":
                w = (Xtr[y_tr].mean(0) - Xtr[~y_tr].mean(0)) / (Xtr.std(0) + 1e-8) ** 2
                f, f_tr = X @ w, Xtr @ w
            elif base == "cdim":
                t = -_DROP[tr_m]
                w = Xtr.T @ (t - t.mean())
                f, f_tr = X @ w, Xtr @ w
            elif base == "rpca":
                t = -_DROP[tr_m]
                r = Ridge(alpha=C).fit(Xtr, t)
                f, f_tr = r.predict(X), r.predict(Xtr)
            else:
                clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(Xtr, y_tr)
                f, f_tr = clf.decision_function(X), clf.decision_function(Xtr)
            acc += (f - f_tr.mean()) / (f_tr.std() + 1e-8)
        feat[idx] = acc / len(offs)
    return feat[:, None]


_UNLAB = {}


def unlabelled(model_id, site, frac=0.5):
    """Unlabelled CoT windows from the effort stage (sep2 prompts, 4k greedy traces): {key: vec} at layer frac.
    seg_0_5 stored; seg_5_9 = (9*seg_0_9 - 5*seg_0_5)/4 (equal-length segments). None if unavailable."""
    key = (model_id, site)
    if key not in _UNLAB:
        import glob
        fs = sorted(glob.glob(str(cv.DATA / "effort" / cv.safe(model_id) / "*.npz")))
        out = {}
        for f in fs:
            z = np.load(f)
            if "seg_0_5" not in z.files:
                break
            j = int(np.argmin(np.abs(z["fracs"] - frac)))
            a5, a9 = z["seg_0_5"][j].astype(np.float64), z["seg_0_9"][j].astype(np.float64)
            out[Path(f).stem] = a5 if site == "seg_0_5" else (9 * a9 - 5 * a5) / 4
        _UNLAB[key] = out or None
    return _UNLAB[key]


def upca_scores(acts, models, y, tr, site, frac, k, C, mode="lr"):
    """Per-model probe in a PCA basis fitted on the model's UNLABELLED CoT windows (effort stage, hundreds to
    ~2k prompts); models without unlabelled windows fall back to the train-row PCA probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    feat = np.zeros(len(acts))
    for m in np.unique(models):
        U = unlabelled(m, site, frac)
        if U is None:
            feat += probe_scores(acts, np.where(models == m, m, "__other__"), y, tr, site, frac, f"pca{min(k, 5)}", C)[:, 0] * (models == m)
            continue
        idx = np.where(models == m)[0]
        tr_m = np.intersect1d(idx, tr)
        y_tr = y[tr_m]
        if y_tr.all() or (~y_tr).all():
            continue
        ck = ("upca", m, site, frac, k)
        if ck not in _UNLAB:   # label-free: the unlabelled PCA and every case's coordinates, computed once
            L = acts[idx[0]][site].shape[0]
            layer = wb.layer_of(L, frac)
            Uo = np.stack(list(U.values()))
            mu = Uo.mean(0)
            sd = np.sqrt(((Uo - mu) ** 2).sum(1).mean()) + 1e-8
            p = PCA(k, random_state=0).fit((Uo - mu) / sd)
            _UNLAB[ck] = p.transform((np.stack([acts[i][site][layer] for i in idx]).astype(np.float64) - mu) / sd)
        H = _UNLAB[ck]
        Htr = H[np.isin(idx, tr_m)]
        if mode == "wdim":
            w = (Htr[y_tr].mean(0) - Htr[~y_tr].mean(0)) / (Htr.std(0) + 1e-8) ** 2
            f, f_tr = H @ w, Htr @ w
        else:
            clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(Htr, y_tr)
            f, f_tr = clf.decision_function(H), clf.decision_function(Htr)
        feat[idx] = (f - f_tr.mean()) / (f_tr.std() + 1e-8)
    return feat[:, None]


def align_scores(acts, models, y, tr, site, frac, k, C, ref="Skywork/Skywork-OR1-Math-7B"):
    """Shared probe across the small models: each model's CoT window -> its unlabelled-PCA(k) coordinates ->
    orthogonal Procrustes onto the reference model's coordinates, fitted on the unlabelled prompts both models
    answered (same problem id / variant). One logistic on all aligned training rows; per-model z-scored score.
    Models without unlabelled windows (gpt-oss) use their own DIM."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    from scipy.linalg import orthogonal_procrustes
    feat = np.zeros(len(acts))
    coords, Z = {}, {}
    ck = ("align", site, frac, k, ref)
    if ck in _UNLAB:
        raw = _UNLAB[ck]
    else:
        raw = None
    Uref = unlabelled(ref, site, frac)
    basis = {}
    for m in (np.unique(models) if raw is None else []):
        U = unlabelled(m, site, frac)
        if U is None:
            continue
        Uo = np.stack(list(U.values()))
        mu = Uo.mean(0); sd = np.sqrt(((Uo - mu) ** 2).sum(1).mean()) + 1e-8
        p = PCA(k, random_state=0).fit((Uo - mu) / sd)
        basis[m] = (mu, sd, p, U)
    if raw is None:
        raw = {}
        rm, rs, rp, _ = basis[ref]
    for m, (mu, sd, p, U) in basis.items():
        shared = sorted(set(U) & set(Uref))
        A = p.transform((np.stack([U[s] for s in shared]) - mu) / sd)
        B = rp.transform((np.stack([Uref[s] for s in shared]) - rm) / rs)
        R = np.eye(k) if m == ref else orthogonal_procrustes(A, B)[0]
        idx = np.where(models == m)[0]
        layer = wb.layer_of(acts[idx[0]][site].shape[0], frac)
        raw[m] = (idx, p.transform((np.stack([acts[i][site][layer] for i in idx]).astype(np.float64) - mu) / sd) @ R)
    _UNLAB[ck] = raw
    for m, (idx, X) in raw.items():
        trm = np.isin(idx, tr)
        X = (X - X[trm].mean(0)) / (X[trm].std(0) + 1e-8)     # per-model standardised in the shared basis
        coords[m] = (idx, X)
    I = np.concatenate([v[0] for v in coords.values()]); X = np.concatenate([v[1] for v in coords.values()])
    trmask = np.isin(I, tr)
    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(X[trmask], y[I[trmask]])
    f = clf.decision_function(X)
    for m, (idx, _) in coords.items():
        sel = np.isin(I, idx); ff = f[sel]; ftr = ff[np.isin(idx, tr)]
        feat[idx] = (ff - ftr.mean()) / (ftr.std() + 1e-8)
    others = [m for m in np.unique(models) if m not in coords]
    if others:
        d = wb.diffmean_scores(acts, models, y, tr, None, site, frac)[:, 0]
        for m in others:
            feat[models == m] = d[models == m]
    return feat[:, None]


def safe_seg_scores(acts, models, y, groups, tr, sites=("seg_0_5", "seg_5_9"), frac=0.5, min_auc=None, soft=False,
                    scorer=None):
    """seg probe scores with a per-model reliability check (loop 2026-10-31): for each model and window, the
    out-of-fold AUC of the DIM score on that model's *training* rows (grouped 5-fold, 2 seeds). Windows whose
    inner AUC is below ``min_auc`` score 0 for that model (no information) instead of a possibly inverted
    direction; with ``soft`` the score is scaled by 2*(AUC-0.5) clipped to [0, 1]."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    min_auc = SAFE_MIN_AUC if min_auc is None else min_auc
    score = scorer or (lambda trr, s: wb.diffmean_scores(acts, models, y, trr, None, s, frac))
    full = np.concatenate([score(tr, s) for s in sites], 1)
    inner = np.zeros((len(y), len(sites)))
    for seed in range(2):
        for itr, ite in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(tr, y[tr], groups[tr]):
            sc = np.concatenate([score(tr[itr], s) for s in sites], 1)
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


TOPIC_KW = {
    "geometry": r"triangle|circle|square|angle|polygon|radius|coordinate|axis|segment|perpendicular|tangent|area|"
                r"rectangle|hexagon|plane|sphere|parallel|chord|diameter|\\cos|\\sin",
    "number": r"prime|divisor|divisible|remainder|gcd|integer|digit|mod|factor|multiple of",
    "combinatorics": r"probability|ways|choose|arrange|random|count|subsets?|sequences?|permutation|game",
}


def problem_topic(problems, four=False):
    """Keyword topic per problem (no robustness labels): the category with the most keyword hits."""
    import re
    keys = list(TOPIC_KW) if four else ["geometry", "number"]
    hits = np.array([[len(re.findall(TOPIC_KW[k], p.lower())) for k in keys] for p in problems])
    lab = np.where(hits.max(1) > 0, hits.argmax(1), len(keys))   # len(keys) = other / no hit
    if not four:
        lab = np.where((hits[:, 0] == hits[:, 1]), 2, lab)        # tie -> other
    return lab


def topic_projected(acts, cases, four=False, sites=("seg_0_5", "seg_5_9")):
    """Copy of acts with, per model, site and layer, the span of topic-class mean differences projected out."""
    lab = problem_topic(cases.problem.tolist(), four)
    models = cases.model_id.to_numpy()
    out = [dict(a) for a in acts]
    for m in np.unique(models):
        idx = np.where(models == m)[0]
        for s in sites:
            H = np.stack([acts[i][s] for i in idx])                      # [n, L+1, d]
            classes = [c for c in np.unique(lab[idx]) if (lab[idx] == c).sum() >= 2]
            if len(classes) < 2:
                continue
            mu = H.mean(0)
            D = np.stack([H[lab[idx] == c].mean(0) - mu for c in classes], 1)   # [L+1, k, d]
            for l in range(H.shape[1]):
                Q, _ = np.linalg.qr(D[l].T)                                       # [d, k]
                Q = Q[:, :len(classes) - 1]
                H[:, l] -= (H[:, l] @ Q) @ Q.T
            for j, i in enumerate(idx):
                out[i][s] = H[j]
    return out


def sens_feats(cases, kind):
    """cvbench/sens_collect.py (small models; 0 elsewhere). lens: logit-lens commitment depth (first layer frac
    from which the first answer token stays top-prob > .5) and mean lens log p over the upper half of layers.
    pert2: forced-answer log p drop / flip / mid-layer CoT cosine shift under 4 problem perturbations."""
    cols = {"lens": ["ll_depth", "ll_upper"], "pert2": ["pt_drop", "pt_flip", "pt_cos"]}[kind]
    rows = []
    for c in cases.itertuples():
        f = cv.DATA / "sens" / cv.safe(c.model_id) / f"{c.group}.npz"
        if not f.is_file():
            rows.append({}); continue
        z = np.load(f)
        if kind == "lens":
            a = z["lens_logp"][:, 0]
            ok = a > np.log(0.5)
            bad = np.flatnonzero(~ok)
            first = (bad[-1] + 1) if len(bad) else 0
            rows.append({"ll_depth": first / (len(a) - 1), "ll_upper": z["lens_logp"][len(a) // 2:].mean()})
        else:
            drop = z["logp_clean"].mean() - z["pert_logp"].mean()
            rows.append({"pt_drop": np.sign(drop) * np.log1p(abs(drop)), "pt_flip": z["pert_flip"].mean(),
                         "pt_cos": np.log(z["pert_cos_cot"].mean() + 1e-6)})
    return pd.DataFrame(rows, index=cases.index, columns=cols).fillna(0.0)


def noise_feats(cases):
    """Latent-noise sensitivity (generate.py --stages noise, small models only; 0 elsewhere): forced answer after
    the greedy trace, noise sigma*||h|| at layer frac .5 on all CoT tokens. log p drop at sigma .3 / 1.0, flip rate."""
    rows = []
    for c in cases.itertuples():
        f = cv.DATA / "noise" / cv.safe(c.model_id) / f"{c.group}.npz"
        if not f.is_file():
            rows.append({}); continue
        z = np.load(f)
        clean = z["logp_clean"].mean()
        drop = clean - z["logp_noisy"][1].mean(-1)          # region 'all', per sigma
        rows.append({"nz_clean": clean, "nz_drop3": np.log1p(max(drop[2], 0)), "nz_drop10": np.log1p(max(drop[3], 0)),
                     "nz_flip": z["flip"][:, 2:].mean()})
    return pd.DataFrame(rows, index=cases.index, columns=["nz_clean", "nz_drop3", "nz_drop10", "nz_flip"]).fillna(0.0)


def attn_feats(cases):
    rows = []
    for c in cases.itertuples():
        f = cv.DATA / "circuit_attn" / cv.safe(c.model_id) / f"{c.group}.npz"
        if not f.is_file():
            rows.append({}); continue
        z = np.load(f)
        R = z["regions"] if "layers" in z else z["regions"][6:]       # [layers lo..hi, H, 10, 5] (NaN: no weights)
        rows.append({"attn_sink": np.nanmean(R[:, :, 5:9, 0] + R[:, :, 5:9, 1]), "attn_far": np.nanmean(R[:, :, 0:9, 3])})
    return pd.DataFrame(rows, index=cases.index, columns=["attn_sink", "attn_far"]).fillna(0.0)


SAE_PATH = Path("/data/gpfs/projects/punim0478/bansaab/sae/qwen3-8b-L18-t0/ae.pt")
SAE_LAYER = 19   # DeepSeek: hidden_states index = resid_post of layer 18 (the SAE's site)
# model -> (checkpoint, hidden_states index, decoder key, decoder is [d, F] (True) or [F, d])
SAE_CFG = {
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": (SAE_PATH, 19, "decoder.weight", True),
    "Skywork/Skywork-OR1-Math-7B": (Path("/data/gpfs/projects/punim0478/bansaab/sae/qwen25-7b-it-L15-t0/ae.pt"),
                                    16, "decoder.weight", True),      # Qwen2.5-7B-Instruct SAE, resid_post L15
    "Qwen/Qwen3.5-4B": (Path("/data/gpfs/projects/punim0478/bansaab/sae/qwen35-4b-L18-cv/sae_final.pt"),
                        int(__import__("os").environ.get("QW_SAE_IDX", "19")), "W_dec", False),  # third-party L18
}
_SAE = {}


def sae_decoder(model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"):
    if model not in _SAE:
        import torch
        path, _, key, d_first = SAE_CFG[model]
        sd = torch.load(path, map_location="cpu", weights_only=True)
        W = sd[key].float().numpy()
        W = W if d_first else W.T                                    # [d, F]
        _SAE[model] = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    return _SAE[model]


def sae_dir(H_tr, y_tr, k, model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", align=None):
    """DIM direction projected on the span of the k SAE decoder directions most aligned with it.
    ``align`` (= W.T @ d, optional) skips the [d, F] product; sae_dim_scores passes it from cached projections."""
    W = sae_decoder(model)
    d = H_tr[y_tr].mean(0) - H_tr[~y_tr].mean(0)
    top = np.argsort(-np.abs(W.T @ d if align is None else align))[:k]
    Q, _ = np.linalg.qr(W[:, top])
    ds = Q @ (Q.T @ d)
    return ds / (np.linalg.norm(ds) + 1e-12), top


_SAE_PROJ = {}


def _sae_proj(acts, idx, site, li, model):
    """[n_model_cases, F] decoder alignments H @ W of every case of the model, computed once: W.T @ d for a
    difference of class means d is then the same difference of rows (the 1.9 GB Skywork decoder made each
    fold's W.T @ d the CV's bottleneck)."""
    key = (model, site, li, id(acts), len(idx))
    if key not in _SAE_PROJ:
        H = np.stack([acts[i][site][li] for i in idx]).astype(np.float64)
        _SAE_PROJ[key] = (H.astype(np.float32) @ sae_decoder(model)).astype(np.float64)
    return _SAE_PROJ[key]


def sae_dim_scores(acts, models, y, tr, site, k, model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"):
    feat = np.zeros(len(acts))
    idx = np.where(models == model)[0]
    tr_m = np.intersect1d(idx, tr)
    y_tr = y[tr_m]
    if y_tr.all() or (~y_tr).all():
        return feat[:, None]
    li = SAE_CFG[model][1]
    H = np.stack([acts[i][site][li] for i in idx]).astype(np.float64)
    H_tr = np.stack([acts[i][site][li] for i in tr_m]).astype(np.float64)
    mu = H_tr.mean(0)
    G = _sae_proj(acts, idx, site, li, model)[np.searchsorted(idx, tr_m)]
    d, _ = sae_dir(H_tr - mu, y_tr, k, model, align=G[y_tr].mean(0) - G[~y_tr].mean(0))
    feat[idx] = ((H - mu) @ d) / (((H_tr - mu) @ d).std() + 1e-8)
    return feat[:, None]


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
        elif name.startswith(("segu_", "segal_")):   # segu_<lr|wdim><k>_<C> / segal_<k>_<C>
            acts = wb.load_acts(cases, cv)
            parts = name.split("_")
            if parts[0] == "segu":
                mode, k = re.match(r"([a-z]+)(\d+)", parts[1]).groups()
                sc = lambda trr, s, acts=acts, mode=mode, k=int(k), Cp=float(parts[2]): upca_scores(acts, models, y, trr, s, args.frac, k, Cp, mode)
            else:
                sc = lambda trr, s, acts=acts, k=int(parts[1]), Cp=float(parts[2]): align_scores(acts, models, y, trr, s, args.frac, k, Cp)
            out[name] = lambda tr, acts=acts, sc=sc: safe_seg_scores(acts, models, y, groups, tr, frac=args.frac, scorer=sc)
        elif name.startswith("segp_"):   # segp_<kind>_<C>: gated seg with a logistic-type probe instead of DIM
            acts = wb.load_acts(cases, cv)
            _, kind, Cp = name.split("_")
            sc = lambda trr, s, acts=acts, kind=kind, Cp=float(Cp): probe_scores(acts, models, y, trr, s, args.frac, kind, Cp)
            out[name] = lambda tr, acts=acts, sc=sc: safe_seg_scores(acts, models, y, groups, tr, frac=args.frac, scorer=sc)
        elif name in ("seg_safe", "seg_soft", "seg_safe5", "seg_safe7"):
            acts = wb.load_acts(cases, cv)
            kw = {"seg_safe": {}, "seg_soft": {"soft": True}, "seg_safe5": {"min_auc": 0.5},
                  "seg_safe7": {"min_auc": 0.7}}[name]
            out[name] = lambda tr, acts=acts, kw=kw: safe_seg_scores(acts, models, y, groups, tr, frac=args.frac, **kw)
        elif name in ("seg_notopic", "seg_notopic4"):
            # gated seg after projecting out label-free topic directions (keyword topic of the problem text; per
            # model and layer, the span of topic-class mean differences). Circuit analysis 2026-11-02: the Skywork
            # DIM direction carries a topic axis (geometry vs number theory) that does not predict robustness.
            acts = topic_projected(wb.load_acts(cases, cv), cases, four=name.endswith("4"))
            out[name] = lambda tr, acts=acts: safe_seg_scores(acts, models, y, groups, tr, frac=args.frac)
        elif name.startswith("seg_sae_"):  # seg_sae_sky10 / seg_sae_qw10 / seg_sae_all10: SAE-projected DIM for those models
            acts = wb.load_acts(cases, cv)
            tag, k = name[8:].rstrip("0123456789"), int(name[8:].lstrip("abcdefghijklmnopqrstuvwxyz"))
            tags = {"sky": ["Skywork/Skywork-OR1-Math-7B"], "qw": ["Qwen/Qwen3.5-4B"],
                    "ds": ["deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"], "all": list(SAE_CFG)}[tag]
            def f(tr, acts=acts, k=k, tags=tags):
                out_ = safe_seg_scores(acts, models, y, groups, tr, frac=args.frac)
                for mm in tags:
                    b = safe_seg_scores(acts, models, y, groups, tr, frac=args.frac,
                                        scorer=lambda trr, s, mm=mm: sae_dim_scores(acts, models, y, trr, s, k, mm))
                    out_ = np.where((models == mm)[:, None], b, out_)
                return out_
            out[name] = f
        elif name.startswith("seg_dssae"):  # seg_dssae20: DeepSeek DIM direction projected on its top-k aligned SAE features
            acts = wb.load_acts(cases, cv)
            k = int(name[9:])
            ds = models == "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
            def f(tr, acts=acts, k=k):
                a = safe_seg_scores(acts, models, y, groups, tr, frac=args.frac)
                b = safe_seg_scores(acts, models, y, groups, tr, frac=args.frac,
                                    scorer=lambda trr, s: sae_dim_scores(acts, models, y, trr, s, k))
                return np.where(ds[:, None], b, a)
            out[name] = f
        elif name.startswith("seg_ds"):  # seg_ds5 / seg_ds3: gated seg, but DeepSeek's DIM after removing its top-k PCs
            acts = wb.load_acts(cases, cv)
            k = int(name[6:])
            ds = models == "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
            def f(tr, acts=acts, k=k):
                a = safe_seg_scores(acts, models, y, groups, tr, frac=args.frac)
                b = safe_seg_scores(acts, models, y, groups, tr, frac=args.frac,
                                    scorer=lambda trr, s: dim_variant_scores(acts, models, y, trr, s, args.frac, pcs=k))
                return np.where(ds[:, None], b, a)
            out[name] = f
        elif name == "seg_pool":
            acts = wb.load_acts(cases, cv)
            out[name] = lambda tr, acts=acts: pooled_seg_scores(acts, models, y, tr, frac=args.frac)
        elif name in ("traj_safe", "tx_safe"):  # per-model sign check of each feature against the pooled sign
            base = "traj" if name == "traj_safe" else "tx"
            Fb = {"traj": lambda: pd.DataFrame([traj.traj_features(sg) for sg in traj.load_segs(cases)],
                                               index=cases.index)[traj.FEATS],
                  "tx": lambda: tx_feats(cases)}[base]()
            Zb = np.asarray(cv.zscore_within_model(Fb, cases), dtype=float)
            out[name] = lambda tr, Zb=Zb: sign_safe(Zb, models, y, tr)
        elif name in ("seg_fill", "seg_fill_dim"):  # gated seg; gated-off windows filled by the label-free difficulty score
            acts = wb.load_acts(cases, cv)
            import difficulty
            col = "diff_last" if name == "seg_fill" else "dim_last"
            D = -np.asarray(cv.zscore_within_model(difficulty.diff_feats(cases, (col,)), cases), dtype=float)[:, 0]
            out[name] = lambda tr, acts=acts, D=D: np.where(
                (g := safe_seg_scores(acts, models, y, groups, tr, frac=args.frac)) == 0, D[:, None], g)
        elif name == "diff_fill":  # difficulty only for models where every seg window is gated off (separate column)
            acts = wb.load_acts(cases, cv)
            import difficulty
            D = -np.asarray(cv.zscore_within_model(difficulty.diff_feats(cases, ("diff_last",)), cases), dtype=float)[:, 0]
            out[name] = lambda tr, acts=acts, D=D: np.where(
                (safe_seg_scores(acts, models, y, groups, tr, frac=args.frac) == 0).all(1), D, 0.0)[:, None]
        elif name.startswith(("segpc", "segboot")):  # e.g. segpc1, segpc3, segboot20, gated per model
            acts = wb.load_acts(cases, cv)
            k = int(name.lstrip("segpcbot"))
            kw = {"pcs": k} if name.startswith("segpc") else {"boot": k}
            out[name] = lambda tr, acts=acts, kw=kw: safe_seg_scores(
                acts, models, y, groups, tr, frac=args.frac,
                scorer=lambda trr, s: dim_variant_scores(acts, models, y, trr, s, args.frac, **kw))
        elif name in ("pc1", "pc2", "pc3"):
            acts = wb.load_acts(cases, cv)
            import json as _json
            n_tok = {}
            for f in (cv.DATA / "greedy").glob("*.jsonl"):
                for line in open(f):
                    r = _json.loads(line)
                    n_tok[(r["model_id"], r["group"])] = r["generation_num_tokens"]
            loglen = np.log1p(np.array([n_tok.get((c.model_id, c.group), 4096) for c in cases.itertuples()], float))
            kk = int(name[2])
            out[name] = lambda tr, acts=acts, loglen=loglen, kk=kk: pc_scores(acts, models, tr, loglen, frac=args.frac, k=kk)
        elif name in ("seg_rfill", "seg_rfill_ds"):  # gated seg; gated-off windows use the pooled reader probe
            acts = wb.load_acts(cases, cv)
            racts = load_reader(cases, "Qwen_Qwen3.5-4B" if name == "seg_rfill" else "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B")
            def blk(tr, acts=acts, racts=racts):
                g = safe_seg_scores(acts, models, y, groups, tr, frac=args.frac)
                r = reader_scores(racts, models, y, tr)
                return np.where(g == 0, r, g)
            out[name] = blk
        elif name.startswith("rdrds"):  # DeepSeek-R1-Qwen3-8B as the reader
            racts = load_reader(cases, "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B")
            out[name] = lambda tr, racts=racts: reader_scores(racts, models, y, tr)
        elif name.startswith("rdr"):  # rdr (pooled, frac .5), rdr_pm (per model), rdr3 / rdr7 (frac .3/.7), rdr_safe
            racts = load_reader(cases)
            fr = {"rdr3": 0.3, "rdr7": 0.7, "rdr8": 0.8}.get(name, 0.5)
            pooled = not name.endswith("_pm")
            if name == "rdr_safe":
                out[name] = lambda tr, racts=racts: safe_seg_scores(
                    racts, models, y, groups, tr, scorer=lambda trr, s: reader_scores(racts, models, y, trr, (s,), 0.5))
            else:
                out[name] = lambda tr, racts=racts, fr=fr, pooled=pooled: reader_scores(racts, models, y, tr, frac=fr,
                                                                                    pooled=pooled)
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
                 "txe4": lambda: txe_feats(cases, 4096), "txe8": lambda: txe_feats(cases, 8192),
                 "txe16": lambda: txe_feats(cases, 16384), "txe2": lambda: txe_feats(cases, 2048),
                 "e4": lambda: txe_feats(cases, 4096, base=False), "e8": lambda: txe_feats(cases, 8192, base=False),
                 **{f"e4_{f}": (lambda f=f: txe_feats(cases, 4096, base=False)[[f]]) for f in TXE_FEATS},
                 "gagree": lambda: gagree_feats(cases),
                 **{f"txx_{e}": (lambda e=e: txx_feats(cases, extra=[e])) for e in TXX_EXTRA},
                 "txx8": lambda: txx_feats(cases, 8192), "txx4": lambda: txx_feats(cases, 4096),
                 "txp": lambda: txp_feats(cases), "txp4": lambda: txp_feats(cases, 4096),
                 "txp8": lambda: txp_feats(cases, 8192), "txp2": lambda: txp_feats(cases, 2048),
                 "lb": lambda: lb_feats(cases),
                 "pert": lambda: pert_feats(cases),
                 "prob": lambda: prob_feats(cases),
                 "nt": lambda: nt_feats(cases), "ntf": lambda: nt_feats(cases, "free"),
                 "nt_se": lambda: nt_feats(cases, feats=["se"]), "ntf_se": lambda: nt_feats(cases, "free", ["se"]),
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
                 "len": lambda: cv.m_length(cases, args)[0],
                 # constant column: intercept-only logistic = predict the pooled base rate (robust). A gate option
                 # that lets a model fall back to "always robust" when its features do worse in inner CV.
                 # attention look-back (circuit analysis 2026-11-02; Skywork only, 0 elsewhere): mean over heads of
                 # layers 6-13 of the attention mass on BOS+template (deciles 5-9) and on CoT > 64 tokens back (0-9)
                 "attn": lambda: attn_feats(cases),
                 "noise": lambda: noise_feats(cases),
                 "lens": lambda: sens_feats(cases, "lens"), "pert2": lambda: sens_feats(cases, "pert2"),
                 "prior": lambda: pd.DataFrame({"prior": np.zeros(len(cases))}, index=cases.index),
                 # competence probe (cvbench/pcorrect.py bench): ridge solve-rate from last prompt, trained on
                 # non-benchmark labelgen problems; Skywork/Olmo/DeepSeek only (others constant)
                 "pc": lambda: pd.read_csv(cv.DATA / "pcorrect" / "scores.csv", index_col=0).reindex(cases.index)[["pc_l2"]],
                 "pc1": lambda: pd.read_csv(cv.DATA / "pcorrect" / "scores.csv", index_col=0).reindex(cases.index)[["pc_l1"]]}[name]()
            Z = cv.zscore_within_model(F, cases)
            out[name] = lambda tr, Z=Z: Z
            continue
        if args.crossfit:
            out[name] = lambda tr, f=out[name]: crossfit(f, y, groups, tr)
    return out


def run(cases, blocks, repeats, mask=None, avg=False, model_bias=False, tune=False, per_model=None, per_model_C=None):
    """avg: one logistic per block, probabilities averaged (instead of one stacked logistic).
    model_bias: add one-hot model columns, i.e. a per-model intercept (the model's base rate)."""
    y = cases.is_robust.astype(bool).to_numpy()
    if model_bias:
        onehot = pd.get_dummies(cases.model_id).to_numpy(float)
        blocks = {k: (lambda tr, b=b: np.c_[b(tr), onehot]) for k, b in blocks.items()}

    groups = cases.group.to_numpy()
    models = cases.model_id.to_numpy()
    per_model_C = per_model_C or C

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
        if per_model:
            # one logistic per model on that model's training rows (falls back to pooled if one class only)
            p_pool = wb._predict(wb._fit_logistic(F[tr], y[tr], C), F[te], True)[1]
            p = p_pool.copy()
            for m in np.unique(models[te]):
                mtr, mte = tr[models[tr] == m], models[te] == m
                if len(np.unique(y[mtr])) == 2:
                    p[mte] = wb._predict(wb._fit_logistic(F[mtr], y[mtr], per_model_C), F[te[mte]], True)[1]
            if per_model == "blend":
                p = (p + p_pool) / 2
            return p >= 0.5, p
        return wb._predict(wb._fit_logistic(F[tr], y[tr], C), F[te], True)
    return cv.evaluate(cases, predict, repeats, score_mask=mask)


def run_ens(cases, sets, repeats, thr="0.5", mask=None):
    """sets: list of block dicts; one logistic per set, probabilities averaged. thr: fixed float, 'glob' (one
    threshold from inner OOF accuracy) or 'pm' (per-model threshold from inner OOF, shrunk to the global one with
    20 pseudo-cases)."""
    y = cases.is_robust.astype(bool).to_numpy()
    groups, models = cases.group.to_numpy(), cases.model_id.to_numpy()
    grid = np.linspace(0.3, 0.7, 41)

    def prob(tr, te):
        out = []
        for bl in sets:
            F = np.concatenate([b(tr) for b in bl.values()], 1)
            out.append(wb._predict(wb._fit_logistic(F[tr], y[tr], C), F[te], True)[1])
        return np.mean(out, 0)

    def best_t(p, yy):
        acc = np.array([((p >= t) == yy).mean() for t in grid])
        return grid[acc >= acc.max() - 1e-9].mean()

    def predict(tr, te):
        p = prob(tr, te)
        if thr not in ("glob", "pm"):
            return p >= float(thr), p
        oof = np.zeros(len(y))
        for itr, ite in StratifiedGroupKFold(5, shuffle=True, random_state=1).split(tr, y[tr], groups[tr]):
            oof[tr[ite]] = prob(tr[itr], tr[ite])
        tg = best_t(oof[tr], y[tr])
        if thr == "glob":
            return p >= tg, p
        pred = p >= tg
        for m in np.unique(models[te]):
            mtr = tr[models[tr] == m]
            tm = best_t(oof[mtr], y[mtr])
            t = (len(mtr) * tm + 20 * tg) / (len(mtr) + 20)
            pred[models[te] == m] = p[models[te] == m] >= t
        return pred, p
    return cv.evaluate(cases, predict, repeats, score_mask=mask)


def threshold_curve(cases, sets, repeats, grid=np.round(np.arange(0.30, 0.751, 0.01), 2)):
    """Out-of-fold probabilities (same splits as cv.evaluate, logistic per set, probabilities averaged), then
    accuracy at every threshold in grid (mean over repeats), plus a per-model threshold picked on the pooled OOF
    (shrunk to the global best with 20 pseudo-cases)."""
    from sklearn.model_selection import StratifiedGroupKFold as SGK
    y = cases.is_robust.astype(bool).to_numpy()
    groups, models = cases.group.to_numpy(), cases.model_id.to_numpy()
    S = []
    for r in range(repeats):
        sc = np.zeros(len(y))
        for tr, te in SGK(5, shuffle=True, random_state=r).split(y, y, groups):
            ps = []
            for bl in sets:
                F = np.concatenate([b(tr) for b in bl.values()], 1)
                ps.append(wb._predict(wb._fit_logistic(F[tr], y[tr], C), F[te], True)[1])
            sc[te] = np.mean(ps, 0)
        S.append(sc)
    S = np.array(S)
    acc = np.array([((S >= t) == y).mean() for t in grid])
    tg = grid[acc.argmax()]
    pm_t = {}
    pred = S >= tg
    for m in np.unique(models):
        k = models == m
        am = np.array([((S[:, k] >= t) == y[k]).mean() for t in grid])
        tm = grid[am >= am.max() - 1e-9].mean()
        pm_t[m.split("/")[-1]] = round(float((k.sum() * tm + 20 * tg) / (k.sum() + 20)), 3)
        pred[:, k] = S[:, k] >= pm_t[m.split("/")[-1]]
    acc_pm = (pred == y).mean()
    per = "; ".join(f"{m.split('/')[-1]}={((S[:, models == m] >= tg) == y[models == m]).mean():.3f}" for m in np.unique(models))
    curve = " ".join(f"{t:.2f}:{a:.3f}" for t, a in zip(grid, acc) if round(t * 100) % 5 == 0)
    return tg, acc.max(), acc[np.argmin(np.abs(grid - 0.5))], acc_pm, pm_t, curve, per


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


def make_fit(kind):
    """Drop-in for wb._fit_logistic(X, y, C): returns a fitted classifier with predict_proba."""
    from sklearn.svm import SVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier

    def fit(X, y, C=1.0):
        if len(set(y)) < 2:
            return None
        if kind == "xgb":
            from xgboost import XGBClassifier
            m = XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=2, reg_lambda=1.0, verbosity=0, n_jobs=4)
        elif kind == "lgbm":
            from lightgbm import LGBMClassifier
            m = LGBMClassifier(n_estimators=200, num_leaves=4, max_depth=2, learning_rate=0.05, min_child_samples=5,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
        elif kind == "cat":
            from catboost import CatBoostClassifier
            m = CatBoostClassifier(iterations=200, depth=3, learning_rate=0.05, l2_leaf_reg=5, verbose=0,
                                   random_seed=0, thread_count=4, allow_writing_files=False)
        else:
            m = {"svm": lambda: SVC(kernel="linear", C=C, probability=True, random_state=0),
                 "lda": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                 "nb": lambda: GaussianNB(),
                 "rf": lambda: RandomForestClassifier(300, max_depth=3, min_samples_leaf=3, random_state=0, n_jobs=4),
                 "gb": lambda: HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=150,
                                                              min_samples_leaf=5),
                 "knn": lambda: KNeighborsClassifier(15, weights="distance")}[kind]()
        return m.fit(X, np.asarray(y).astype(int))
    return fit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="+", required=True)
    ap.add_argument("--gate", nargs="+", default=None,
                    help="Per-model choice among block sets, each 'a+b' (e.g. tx seg+traj seg+traj+tx).")
    ap.add_argument("--avg", action="store_true")
    ap.add_argument("--tune", action="store_true", help="Threshold from inner-CV accuracy instead of 0.5.")
    ap.add_argument("--crossfit", action="store_true", help="Cross-fit the probe scores of training rows.")
    ap.add_argument("--model-bias", action="store_true", help="Per-model intercept (one-hot model columns).")
    ap.add_argument("--per-model", choices=["sep", "blend"], default=None,
                    help="sep: one logistic per model; blend: average of per-model and pooled probabilities")
    ap.add_argument("--clf", default=None, choices=["svm", "lda", "nb", "rf", "gb", "xgb", "lgbm", "cat", "knn"],
                    help="replace the logistic everywhere (wb._fit_logistic) with this classifier")
    ap.add_argument("--ens", nargs="+", default=None, help="block sets 'a+b' whose probabilities are averaged")
    ap.add_argument("--thr", default=None, help="with --ens (or alone): 0.5 / a float / glob / pm")
    ap.add_argument("--per-model-C", type=float, default=None)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--sample", type=int, default=0, help="tseg1: which sampled trace")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--score-sources", nargs="+", default=None)
    ap.add_argument("--cases", nargs="+", default=list(cv.CASE_FILES), help="Case files under data/cvbench.")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    if args.clf:
        wb._fit_logistic = make_fit(args.clf)
    cases = cv.load_cases(args.cases)
    mask = cases.source.isin(args.score_sources).to_numpy() if args.score_sources else None
    global _DROP
    _DROP = cases.max_drop.to_numpy(float)
    blocks = make_blocks(cases, args.blocks, args)
    if args.thr == "sweep":
        sets = args.ens or ["+".join(args.blocks)]
        tg, best, a50, apm, pmt, curve, per = threshold_curve(cases, [{b: blocks[b] for b in g.split("+")} for g in sets], args.repeats)
        print(f"thrsweep {' | '.join(sets)} best_thr={tg:.2f} acc={best:.4f} (at .50 {a50:.4f}) per-model-thr acc {apm:.4f} "
              f"{pmt} | curve {curve} | at best: {per}")
        return
    if args.ens or args.thr:
        sets = args.ens or ["+".join(args.blocks)]
        res = run_ens(cases, [{b: blocks[b] for b in g.split("+")} for g in sets], args.repeats, args.thr or "0.5", mask)
        print(f"ens {' | '.join(sets)} thr={args.thr or '0.5'} acc={res['cv_score']:.4f}±{res['cv_std']:.4f} "
              f"auc={res['auc']:.4f} bacc={res['balanced_acc']:.4f} | {res['per_model']}")
        print(f"METRIC {res['cv_score']:.4f}")
        return
    if args.gate:
        options = {g: {b: blocks[b] for b in g.split("+")} for g in args.gate}
        res, picks = run_gated(cases, options, args.repeats, mask)
        print(f"gate {' | '.join(args.gate):22s} acc={res['cv_score']:.4f}±{res['cv_std']:.4f} auc={res['auc']:.4f} "
              f"bacc={res['balanced_acc']:.4f} | {res['per_model']}")
        print("picks:", dict(sorted(picks.items())))
        print(f"METRIC {res['cv_score']:.4f}")
        return
    res = run(cases, blocks, args.repeats, mask, args.avg, args.model_bias, args.tune, args.per_model, args.per_model_C)
    tag = ("avg " if args.avg else "") + "+".join(args.blocks) + (" +bias" if args.model_bias else "") + (" crossfit" if args.crossfit else "") + (" tuned" if args.tune else "") + (f" pm-{args.per_model} C={args.per_model_C}" if args.per_model else "")
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
