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
            row = {k: z[k].astype(np.float32) for k in ("last_prompt", "resp_mean", "pert_last")}
            row["pert_types"] = z["pert_types"]
            pool = cv.DATA / "pool" / cv.safe(c.model_id) / f"{c.group}.npz"
            if pool.is_file():
                zp = np.load(pool)
                row.update({k: zp[k].astype(np.float32) for k in ("prob_mean", "tmpl_last", "pert_prob_mean",
                                                                   "pert_tmpl_last")})
            acts.append(row)
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


def diffmean_scores(acts, models, y, tr, te, site, frac, window=0):
    """1-D projection feature, z-scored within model on train stats; averaged over layer +-window."""
    if window:
        L = {m: acts[np.where(models == m)[0][0]][site].shape[0] - 1 for m in np.unique(models)}
        cols = []
        for k in range(-window, window + 1):
            # shift every model by k layers around its own frac
            cols.append(np.concatenate([diffmean_scores_layer(acts, models, y, tr, site, frac, k)], 1))
        return np.mean(cols, 0)
    return diffmean_scores_layer(acts, models, y, tr, site, frac, 0)


def diffmean_scores_layer(acts, models, y, tr, site, frac, offset):
    feat = np.zeros(len(acts))
    for m in np.unique(models):
        idx = np.where(models == m)[0]
        tr_m, all_m = np.intersect1d(idx, tr), idx
        L = acts[idx[0]][site].shape[0]
        layer = min(max(layer_of(L, frac) + offset, 0), L - 1)
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


def pca_fit_predict(acts, models, y, tr, te, site, frac, k, C):
    """Per model: PCA(k) on its training rows, then an L2 logistic probe on the components.

    A model whose training rows hold one class predicts that class. Returns (pred, prob) for ``te``.
    """
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pred, prob = np.ones(len(te), dtype=bool), np.full(len(te), 0.5)
    for m in np.unique(models[te]):
        tr_m, te_pos = tr[models[tr] == m], np.where(models[te] == m)[0]
        if not len(tr_m):
            continue
        layer = layer_of(acts[tr_m[0]][site].shape[0], frac)
        H_tr = np.stack([acts[i][site][layer] for i in tr_m])
        H_te = np.stack([acts[i][site][layer] for i in te[te_pos]])
        if len(set(y[tr_m])) < 2:
            pred[te_pos], prob[te_pos] = y[tr_m][0], float(y[tr_m][0])
            continue
        n_comp = min(k, len(tr_m) - 1)
        clf = make_pipeline(StandardScaler(), PCA(n_comp, random_state=0), StandardScaler(),
                            LogisticRegression(C=C, max_iter=2000)).fit(H_tr, y[tr_m])
        prob[te_pos] = clf.predict_proba(H_te)[:, 1]
        pred[te_pos] = prob[te_pos] >= 0.5
    return pred, prob


def register(methods, cv):
    def pca_probe(cases, args):
        acts = load_acts(cases, cv)
        y = cases.is_robust.astype(bool).to_numpy()
        models = cases.model_id.to_numpy()
        groups = cases.group.to_numpy()
        sites = ("last_prompt", "resp_mean") if args.site is None else (args.site,)
        fracs = LAYER_FRACS if args.layer_frac is None else (args.layer_frac,)
        ks = tuple(args.pca_k)
        Cs = tuple(args.pca_C or [args.C])
        cands = [(s, f, k, c) for s in sites for f in fracs for k in ks for c in Cs]

        def select(tr):
            y_tr, g_tr = y[tr], groups[tr]
            n_splits = min(4, len(set(g_tr[y_tr])), len(set(g_tr[~y_tr])))
            splits = list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=0).split(tr, y_tr, g_tr))
            best, best_acc = None, -1.0
            for cand in cands:
                correct = []
                for itr, ite in splits:
                    p, _ = pca_fit_predict(acts, models, y, tr[itr], tr[ite], *cand)
                    correct += list(p == y_tr[ite])
                if np.mean(correct) > best_acc:
                    best, best_acc = cand, np.mean(correct)
            return best

        def predict(tr, te):
            cand = cands[0] if len(cands) == 1 else select(tr)
            return pca_fit_predict(acts, models, y, tr, te, *cand)

        sel = "fixed" if len(cands) == 1 else "nested-CV"
        return predict, (f"per-model PCA + L2 logistic probe; site in {list(sites)}; layer frac in {list(fracs)}; "
                         f"k in {list(ks)}; C in {list(Cs)} ({sel})")

    methods["pca_probe"] = ("whitebox", pca_probe)

    def shift_resid(cases, args):
        acts = load_acts(cases, cv)
        y = cases.is_robust.astype(bool).to_numpy()
        models = cases.model_id.to_numpy()
        groups = cases.group.to_numpy()
        sites = tuple(args.shift_sites)
        fracs = LAYER_FRACS if args.layer_frac is None else (args.layer_frac,)
        cache = {}

        def make(cand, rows):
            if cand not in cache:
                cache[cand] = residual_features(acts, models, *cand, args.shift_feats)
            return cache[cand]

        def predict(tr, te):
            cands = [(s_, f) for s_ in sites for f in fracs]
            cand = cands[0] if len(cands) == 1 else _inner_select(cands, make, y, groups, tr, args.C)
            F = make(cand, tr)
            if args.with_probe:
                F = np.concatenate([F, diffmean_scores(acts, models, y, tr, None, "resp_mean", 0.5)], 1)
            return _predict(_fit_logistic(F[tr], y[tr], args.C), F[te], True)

        return predict, (f"shift d=h(dX)-h(X), r per model x perturbation type (label-free); feats {args.shift_feats}; "
                         f"site in {list(sites)}; layer frac nested in {list(fracs)}"
                         f"{'; + resp_mean diff-in-means probe @0.5' if args.with_probe else ''}; logistic C={args.C}")

    methods["shift_resid"] = ("whitebox", shift_resid)


SITE_PAIRS = {"last": ("last_prompt", "pert_last"), "prob_mean": ("prob_mean", "pert_prob_mean"),
              "tmpl_last": ("tmpl_last", "pert_tmpl_last")}


def residual_features(acts, models, site, frac, feats):
    """Label-free per-case shift scores (proj, resid, norm), r per model x perturbation type, z-scored within model."""
    x_key, dx_key = SITE_PAIRS[site]
    names = ("proj", "resid", "norm")
    out = np.zeros((len(acts), len(feats)))
    for m in np.unique(models):
        idx = np.where(models == m)[0]
        layer = layer_of(acts[idx[0]][x_key].shape[0], frac)
        acc = np.zeros((len(idx), 3))
        cnt = np.zeros(len(idx))
        for kind in sorted({t for i in idx for t in acts[i]["pert_types"]}):
            D = [acts[i][dx_key][acts[i]["pert_types"] == kind, layer] - acts[i][x_key][layer] for i in idx]
            have = [j for j in range(len(idx)) if len(D[j])]
            if len(have) < 3:
                continue
            for j in have:
                r = np.concatenate([D[o] for o in have if o != j]).mean(0)   # leave this problem out
                r /= np.linalg.norm(r) + 1e-8
                p = D[j] @ r
                scale = np.linalg.norm(acts[idx[j]][x_key][layer]) + 1e-8
                acc[j] += [p.mean() / scale, np.linalg.norm(D[j] - p[:, None] * r, axis=1).mean() / scale,
                           np.linalg.norm(D[j], axis=1).mean() / scale]
                cnt[j] += 1
        v = acc / np.maximum(cnt, 1)[:, None]
        v = (v - v.mean(0)) / (v.std(0) + 1e-8)
        out[idx] = v[:, [names.index(f) for f in feats]]
    return out

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
            make = lambda cand, rows: diffmean_scores(acts, models, y, rows, None, *cand, window=args.layer_window)
            cand = cands[0] if len(cands) == 1 else _inner_select(cands, make, y, groups, tr, args.C)
            chosen.append(cand)
            F = make(cand, tr)
            if extra is not None:
                F = np.concatenate([F, extra], 1)
            return _predict(_fit_logistic(F[tr], y[tr], args.C), F[te], True)

        extra = None
        if args.with_bb:
            blocks = [cv.zscore_within_model(cv.METHODS[name][1](cases, args)[0], cases) for name in args.with_bb]
            extra = np.concatenate(blocks, 1)
        sel = "fixed" if len(cands) == 1 else "nested-CV"
        return predict, (f"{'+'.join(['probe'] + (args.with_bb or []))}: " if args.with_bb else "") + (f"diff-in-means probe per model; site in {list(sites)}; layer frac in {list(fracs)} ({sel}); "
                         f"layer window +-{args.layer_window}; logistic C={args.C}")

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
