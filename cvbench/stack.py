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
