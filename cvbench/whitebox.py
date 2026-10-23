"""White-box methods for ``cv.py``, built on ``data/cvbench/acts/`` (``generate.py --stages acts``).

Each case has, at every layer: ``last_prompt`` (last prompt token), ``resp_mean``
(mean over the greedy response) and ``pert_last`` (last prompt token of each of
4 code-made perturbations of the problem).

* ``act_probe``: difference-in-means probe. Per model and fold, the direction
  mean(h | robust) - mean(h | not robust) at one layer; the projection is
  z-scored within model with training-fold statistics, and a shared logistic
  regression maps it to a label. Site and layer depth are picked by an inner
  grouped CV on the training fold (nested), or fixed with ``--layer-frac``.
* ``perturb_direction``: the refusal-direction recipe (Arditi et al., 2024)
  applied to perturbations. Per model and layer, the label-free direction
  r = mean(h(dX)) - mean(h(X)) over all problems is the model's generic
  "this prompt was perturbed" direction. Per problem, the features are how far
  its perturbations move it along r (mean and max projection of
  h(dX) - h(X) on unit r) and off r (residual norm), relative to its norm.
  Hypothesis: a problem whose representation is pushed hard by surface edits
  is one the model is not robust on. Layer depth is picked by nested CV.
* ``perturb_direction_sup``: as above plus a supervised direction in shift
  space, diff-in-means of the shift vectors mean_k(h(dX_k) - h(X)) between
  robust and non-robust training problems.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

LAYER_FRACS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_CACHE: dict = {}


def load_acts(cases, cv):
    key = id(cases)
    if key not in _CACHE:
        acts = []
        for c in cases.itertuples():
            path = cv.acts_file(c.model_id, c.group)
            z = np.load(path)
            acts.append({k: z[k].astype(np.float32) for k in ("last_prompt", "resp_mean", "pert_last")})
        _CACHE[key] = acts
    return _CACHE[key]


def layer_of(n_layers_plus_1: int, frac: float) -> int:
    return int(round(frac * (n_layers_plus_1 - 1)))


def _fit_logistic(X, y, C=1.0):
    if len(set(y)) < 2:
        return None
    return LogisticRegression(C=C, max_iter=2000).fit(X, y)


def _predict(clf, X, default):
    if clf is None:
        return np.full(len(X), default, dtype=bool), np.zeros(len(X))
    prob = clf.predict_proba(X)[:, 1]
    return prob >= 0.5, prob


def diffmean_scores(acts, models, y, tr, te, site, frac):
    """1-D projection feature for train and test rows, z-scored within model on train stats."""
    feat = np.zeros(len(acts))
    for m in np.unique(models):
        idx = np.where(models == m)[0]
        tr_m, all_m = np.intersect1d(idx, tr), idx
        L = acts[idx[0]][site].shape[0]
        layer = layer_of(L, frac)
        H = np.stack([acts[i][site][layer] for i in all_m])
        H_tr = np.stack([acts[i][site][layer] for i in tr_m])
        y_tr = y[tr_m]
        if y_tr.all() or (~y_tr).all():
            continue  # no direction: feature stays 0 (the prior)
        mu = H_tr.mean(0)
        d = H_tr[y_tr].mean(0) - H_tr[~y_tr].mean(0)
        d /= np.linalg.norm(d) + 1e-8
        proj = (H - mu) @ d
        proj_tr = (H_tr - mu) @ d
        feat[all_m] = proj / (proj_tr.std() + 1e-8)
    return feat[:, None]


def shift_features(acts, models, frac, tr=None, y=None, supervised=False):
    feats = np.zeros((len(acts), 4 if supervised else 3))
    for m in np.unique(models):
        idx = np.where(models == m)[0]
        L = acts[idx[0]]["last_prompt"].shape[0]
        layer = layer_of(L, frac)
        hx = np.stack([acts[i]["last_prompt"][layer] for i in idx])               # [n, d]
        hdx = np.stack([acts[i]["pert_last"][:, layer] for i in idx])             # [n, k, d]
        shift = hdx - hx[:, None]                                                 # [n, k, d]
        r = shift.reshape(-1, shift.shape[-1]).mean(0)
        r /= np.linalg.norm(r) + 1e-8
        along = shift @ r                                                         # [n, k]
        resid = np.linalg.norm(shift - along[..., None] * r, axis=-1)            # [n, k]
        scale = np.linalg.norm(hx, axis=-1, keepdims=True) + 1e-8
        block = [along.mean(1) / scale[:, 0], along.max(1) / scale[:, 0], resid.mean(1) / scale[:, 0]]
        if supervised:
            mean_shift = shift.mean(1)
            tr_mask = np.isin(idx, tr)
            y_m = y[idx]
            col = np.zeros(len(idx))
            if tr_mask.any() and 0 < y_m[tr_mask].sum() < tr_mask.sum():
                d = mean_shift[tr_mask & y_m].mean(0) - mean_shift[tr_mask & ~y_m].mean(0)
                d /= np.linalg.norm(d) + 1e-8
                col = mean_shift @ d
            block.append(col)
        B = np.stack(block, 1)
        B = (B - B.mean(0)) / (B.std(0) + 1e-8)   # label-free within-model z-score
        feats[idx] = B
    return feats


def _inner_select(candidates, make_feats, y, groups, tr, C):
    """Pick the candidate with the best inner grouped-CV accuracy on the training rows."""
    best, best_key = None, None
    y_tr, g_tr = y[tr], groups[tr]
    n_splits = min(4, len(set(g_tr[y_tr])), len(set(g_tr[~y_tr])))
    for cand in candidates:
        correct, scores = [], []
        for itr, ite in StratifiedGroupKFold(n_splits, shuffle=True, random_state=0).split(tr, y_tr, g_tr):
            F = make_feats(cand, tr[itr])
            clf = _fit_logistic(F[tr[itr]], y_tr[itr], C)
            p, s = _predict(clf, F[tr[ite]], True)
            correct += list(p == y_tr[ite])
        key = (np.mean(correct),)
        if best_key is None or key > best_key:
            best, best_key = cand, key
    return best


def register(methods, cv):
    def act_probe(cases, args):
        acts = load_acts(cases, cv)
        y = cases.is_robust.astype(bool).to_numpy()
        models = cases.model_id.to_numpy()
        groups = cases.group.to_numpy()
        sites = ("last_prompt", "resp_mean") if args.site is None else (args.site,)
        fracs = LAYER_FRACS if args.layer_frac is None else (args.layer_frac,)
        cands = [(s, f) for s in sites for f in fracs]
        chosen = []

        def predict(tr, te):
            make = lambda cand, rows: diffmean_scores(acts, models, y, rows, None, *cand)
            cand = cands[0] if len(cands) == 1 else _inner_select(cands, make, y, groups, tr, args.C)
            chosen.append(cand)
            F = make(cand, tr)
            return _predict(_fit_logistic(F[tr], y[tr], args.C), F[te], True)

        sel = "fixed" if len(cands) == 1 else "nested-CV"
        return predict, (f"diff-in-means probe per model; site in {list(sites)}; layer frac in {list(fracs)} ({sel}); "
                         f"logistic C={args.C}")

    def perturb_direction(cases, args, supervised=False):
        acts = load_acts(cases, cv)
        y = cases.is_robust.astype(bool).to_numpy()
        models = cases.model_id.to_numpy()
        groups = cases.group.to_numpy()
        fracs = LAYER_FRACS if args.layer_frac is None else (args.layer_frac,)

        def predict(tr, te):
            make = lambda f, rows: shift_features(acts, models, f, rows, y, supervised)
            f = fracs[0] if len(fracs) == 1 else _inner_select(fracs, make, y, groups, tr, args.C)
            F = make(f, tr)
            return _predict(_fit_logistic(F[tr], y[tr], args.C), F[te], True)

        sel = "fixed" if len(fracs) == 1 else "nested-CV"
        return predict, (f"refusal-style perturbation direction r=mean(h(dX))-mean(h(X)), last prompt token; "
                         f"4 code perturbations; layer frac in {list(fracs)} ({sel}); "
                         f"{'+ supervised shift direction; ' if supervised else ''}logistic C={args.C}")

    methods["act_probe"] = ("whitebox", act_probe)
    methods["perturb_direction"] = ("whitebox", perturb_direction)
    methods["perturb_direction_sup"] = ("whitebox", lambda c, a: perturb_direction(c, a, True))


def layer_report(cases, cv, out_path):
    """Per-layer OOF AUCs: robust vs not (diff-in-means) and original vs perturbed (label-free probe)."""
    acts = load_acts(cases, cv)
    y = cases.is_robust.astype(bool).to_numpy()
    models = cases.model_id.to_numpy()
    groups = cases.group.to_numpy()
    rows = []
    for m in np.unique(models):
        idx = np.where(models == m)[0]
        L = acts[idx[0]]["last_prompt"].shape[0]
        for layer in range(L):
            hx = np.stack([acts[i]["last_prompt"][layer] for i in idx])
            hdx = np.stack([acts[i]["pert_last"][:, layer] for i in idx])
            # original vs perturbed: leave-one-problem-out diff-in-means on the paired shift
            shift = hdx - hx[:, None]
            scores_x, scores_dx = [], []
            for j in range(len(idx)):
                r = np.delete(shift, j, 0).reshape(-1, shift.shape[-1]).mean(0)
                scores_x.append(hx[j] @ r)
                scores_dx += list(hdx[j] @ r)
            lab = np.r_[np.zeros(len(scores_x)), np.ones(len(scores_dx))]
            auc_pert = roc_auc_score(lab, np.r_[scores_x, scores_dx])
            row = {"model": m, "layer": layer, "frac": layer / (L - 1), "auc_orig_vs_perturbed": auc_pert}
            for site in ("last_prompt", "resp_mean"):
                yi = y[idx]
                if yi.all() or (~yi).all():
                    row[f"auc_robust_{site}"] = np.nan
                    continue
                H = np.stack([acts[i][site][layer] for i in idx])
                s = np.zeros(len(idx))
                for j in range(len(idx)):
                    keep = np.delete(np.arange(len(idx)), j)
                    if yi[keep].all() or (~yi[keep]).all():
                        continue
                    d = H[keep][yi[keep]].mean(0) - H[keep][~yi[keep]].mean(0)
                    s[j] = (H[j] - H[keep].mean(0)) @ d
                row[f"auc_robust_{site}"] = roc_auc_score(yi, s)
            rows.append(row)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df
