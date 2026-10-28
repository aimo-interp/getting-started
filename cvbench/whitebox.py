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
* ``seg_probe``: ``act_probe`` with the response pooled over a window of the CoT
  instead of all of it. ``data/cvbench/segs/`` (``generate.py --stages segs``) holds
  per-layer means over 10 equal token segments; a window ``a-b`` averages
  segments a..b-1 (``0-10`` = resp_mean up to rounding). Window and layer depth are
  picked by nested CV, or fixed with ``--seg-windows`` / ``--layer-frac``.
  Motivation: Thought Anchors finds that a few sentences (often a late commitment)
  decide the answer, so the robustness signal may sit in part of the trace.
* ``sep_probe``: semantic-entropy probe (Kossen et al., 2024). Per model and fold, a
  ridge regression from the last-prompt-token hidden state to the answer entropy of
  the model's own samples, trained on every labelgen prompt (originals and perturbed
  variants, ``data/cvbench/sep/``) whose problem is not in the test fold, then
  applied to each case's X prompt; feature = -predicted entropy. Models without
  labelgen samples get the prior (0). ``--sep-with-seg`` adds the seg_probe 0-5/5-9
  scores. Hypothesis: problems the model is internally unsure about are less robust.
* ``sep2_probe``: ``sep_probe`` in the paper's setup (``generate.py --stages sep2``):
  10 samples per prompt at T=1; target = discrete semantic entropy (answer clusters;
  ``--sep-none`` says whether a sample with no answer is a cluster or left out),
  binarized at the split minimizing within-class squared error; probe = logistic
  regression (default L2) on the hidden state at ``--sep-pos`` (TBG: token before
  generation; SLT: last token of the greedy response) over ``--sep-block`` adjacent
  layers, the block picked per model by held-out SE AUROC on the SEP data alone.
  Feature = -probe logit on each case, z-scored within model.
* ``perturb_direction_sup``: as above plus a supervised direction in shift
  space, diff-in-means of the shift vectors mean_k(h(dX_k) - h(X)) between
  robust and non-robust training problems.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

LAYER_FRACS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# Windows over the 10 response segments: all, first/last 10/30/50%, middle 40%, and the
# CoT without its final 10% (the answer wrap-up, where the probe is at chance).
SEG_WINDOWS = ((0, 10), (0, 1), (0, 3), (0, 5), (5, 10), (7, 10), (9, 10), (3, 7), (0, 9), (5, 9))
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
            segs = cv.DATA / "segs" / cv.safe(c.model_id) / f"{c.group}.npz"
            if segs.is_file():
                seg = np.load(segs)["seg_mean"].astype(np.float32)
                for a, b in SEG_WINDOWS:
                    row[f"seg_{a}_{b}"] = seg[a:b].mean(0)
            else:  # no segments (response too short): every window falls back to the full mean
                for a, b in SEG_WINDOWS:
                    row[f"seg_{a}_{b}"] = row["resp_mean"]
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

    def seg_probe(cases, args):
        acts = load_acts(cases, cv)
        y = cases.is_robust.astype(bool).to_numpy()
        models = cases.model_id.to_numpy()
        if getattr(args, "pool_efforts", False):
            models = np.array([m.partition(":")[0] for m in models])
        groups = cases.group.to_numpy()
        windows = [tuple(map(int, w.split("-"))) for w in args.seg_windows] if args.seg_windows else SEG_WINDOWS
        sites = [f"seg_{a}_{b}" for a, b in windows]
        fracs = LAYER_FRACS if args.layer_frac is None else (args.layer_frac,)
        cands = [(s_, f) for s_ in sites for f in fracs]

        def predict(tr, te):
            make = lambda cand, rows: diffmean_scores(acts, models, y, rows, None, *cand, window=args.layer_window)
            if args.seg_combine:  # one probe score per window, all fed to the logistic (layer fixed)
                frac = 0.5 if args.layer_frac is None else args.layer_frac
                F = np.concatenate([make((s_, frac), tr) for s_ in sites], 1)
            else:
                cand = cands[0] if len(cands) == 1 else _inner_select(cands, make, y, groups, tr, args.C)
                F = make(cand, tr)
            if extra is not None:
                F = np.concatenate([F, extra], 1)
            return _predict(_fit_logistic(F[tr], y[tr], args.C), F[te], True)

        extra = None
        if args.with_bb:
            extra = np.concatenate([cv.zscore_within_model(cv.METHODS[name][1](cases, args)[0], cases)
                                    for name in args.with_bb], 1)
        sel = "combined: one score per window" if args.seg_combine else ("fixed" if len(cands) == 1 else "nested-CV")
        return predict, (f"{'+'.join(['probe'] + args.with_bb)}: " if args.with_bb else "") + (
                         f"diff-in-means probe per model on a CoT window (10 segments); windows "
                         f"{['%d-%d' % w for w in windows]}; layer frac in {list(fracs)} ({sel}); logistic C={args.C}"
                         f"{'; efforts pooled' if getattr(args, 'pool_efforts', False) else ''}")

    def sep_probe(cases, args):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        acts = load_acts(cases, cv)
        y = cases.is_robust.astype(bool).to_numpy()
        models = cases.model_id.to_numpy()
        problems = cases.problem.to_numpy()
        frac = 0.5 if args.layer_frac is None else args.layer_frac
        sep = {}
        for m in np.unique(models):
            path = cv.DATA / "sep" / f"{cv.safe(m)}.npz"
            if path.is_file():
                z = np.load(path)
                sep[m] = dict(h=z["h"].astype(np.float32), e=z["entropy"], problem=z["problem"])

        def sep_feature(tr, te):
            feat = np.zeros(len(acts))
            held_out = set(problems[te])
            for m, d in sep.items():
                idx = np.where(models == m)[0]
                layer = layer_of(d["h"].shape[1], frac)
                keep = np.array([q not in held_out for q in d["problem"]])
                scaler = StandardScaler().fit(d["h"][keep, layer])
                reg = Ridge(alpha=args.sep_alpha).fit(scaler.transform(d["h"][keep, layer]), d["e"][keep])
                H = np.stack([acts[i]["last_prompt"][layer] for i in idx])
                pred = -reg.predict(scaler.transform(H))
                tr_m = np.isin(idx, tr)
                feat[idx] = (pred - pred[tr_m].mean()) / (pred[tr_m].std() + 1e-8)
            return feat[:, None]

        def predict(tr, te):
            F = sep_feature(tr, te)
            if args.sep_with_seg:
                F = np.concatenate([F] + [diffmean_scores(acts, models, y, tr, None, w, frac) for w in ("seg_0_5", "seg_5_9")], 1)
            return _predict(_fit_logistic(F[tr], y[tr], args.C), F[te], True)

        return predict, (f"semantic-entropy probe: ridge(alpha={args.sep_alpha}) last-prompt h -> answer entropy on labelgen "
                         f"prompts minus test-fold problems; models with SEP data {[m.split('/')[-1] for m in sep]}; "
                         f"layer frac {frac}{'; + seg_probe 0-5/5-9' if args.sep_with_seg else ''}; logistic C={args.C}")

    def sep2_probe(cases, args):
        acts = load_acts(cases, cv)
        y = cases.is_robust.astype(bool).to_numpy()
        models = cases.model_id.to_numpy()
        problems = cases.problem.to_numpy()
        sep = {}
        for m in np.unique(models):
            d = load_sep2(cv, m, args.sep_none, args.sep_answers)
            if d is None:
                continue
            if args.sep_pos == "tbg":
                d["case_h"] = np.stack([acts[i]["last_prompt"] for i in np.where(models == m)[0]])
            else:
                z = np.load(cv.DATA / "sep2" / f"{cv.safe(m)}__cases.npz")
                by_group = dict(zip(z["group"], z["h_slt"].astype(np.float32)))
                d["case_h"] = np.stack([by_group[g] for g in cases.group.to_numpy()[models == m]])
            d["layers"], d["layer_auc"] = sep2_select_layers(d, args.sep_pos, args.sep_block)
            sep[m] = d
            print(f"[sep2] {m.split('/')[-1]}: {len(d['se'])} prompts, {np.mean(d['se'] == 0):.0%} zero SE, "
                  f"layers {d['layers'][0]}-{d['layers'][-1]} (held-out SE AUROC {d['layer_auc']:.3f})", flush=True)

        def sep_feature(tr, te):
            feat = np.zeros(len(acts))
            held_out = set(problems[te])
            for m, d in sep.items():
                idx = np.where(models == m)[0]
                keep = np.array([q not in held_out for q in d["problem"]])
                X = sep2_features(d["h"][keep], d["layers"])
                target = d["se"][keep] > sep2_threshold(d["se"][keep])
                if target.all() or (~target).all():
                    continue
                clf = LogisticRegression(max_iter=1000).fit(X, target)
                score = -clf.decision_function(sep2_features(d["case_h"], d["layers"]))  # high = low entropy
                tr_m = np.isin(idx, tr)
                feat[idx] = (score - score[tr_m].mean()) / (score[tr_m].std() + 1e-8)
            return feat[:, None]

        def predict(tr, te):
            F = sep_feature(tr, te)
            if args.sep_with_seg:
                F = np.concatenate([F] + [diffmean_scores(acts, models, y, tr, None, w, 0.5) for w in ("seg_0_5", "seg_5_9")], 1)
            return _predict(_fit_logistic(F[tr], y[tr], args.C), F[te], True)

        layers = {m.split('/')[-1]: f"{d['layers'][0]}-{d['layers'][-1]} (SE AUROC {d['layer_auc']:.2f})" for m, d in sep.items()}
        return predict, (f"semantic-entropy probe, paper setup: logistic(L2, C=1) on {args.sep_pos.upper()} h, "
                         f"{args.sep_block} adjacent layers -> SE above the SSE split, 10 samples/prompt ({args.sep_answers} answers), "
                         f"no-answer {args.sep_none}; trained on sep2 prompts minus test-fold problems; layers {layers}"
                         f"{'; + seg_probe 0-5/5-9' if args.sep_with_seg else ''}; logistic C={args.C}")

    def len_probe(cases, args):
        """Ridge from the last-prompt hidden state to the mean log length of the model's sampled responses,
        trained on the sep2 prompts (originals + variants, 10 samples each, no labels). Each case is
        predicted by a ridge fitted without its own problem, so train and test cases get the same kind of
        out-of-sample score; the predicted length (short = robust) is the feature."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        acts = load_acts(cases, cv)
        y = cases.is_robust.astype(bool).to_numpy()
        models = cases.model_id.to_numpy()
        problems = np.array([_norm(t) for t in cases.problem])
        feat = np.zeros(len(acts))
        with_data = []
        for m in np.unique(models):
            d = load_len2(cv, m)
            if d is None:
                continue
            with_data.append(m.split("/")[-1])
            layer = layer_of(d["h"].shape[1], args.len_frac)
            X_all = d["h"][:, layer].astype(np.float32)
            idx = np.where(models == m)[0]
            score = np.zeros(len(idx))
            for q in np.unique(problems[idx]):
                keep = d["problem"] != q
                scaler = StandardScaler().fit(X_all[keep])
                reg = Ridge(alpha=args.len_alpha).fit(scaler.transform(X_all[keep]), d["y"][keep])
                rows = idx[problems[idx] == q]
                H = np.stack([acts[i]["last_prompt"][layer] for i in rows])
                score[np.isin(idx, rows)] = -reg.predict(scaler.transform(H))  # high = short responses = robust
            feat[idx] = (score - score.mean()) / (score.std() + 1e-8)  # label-free, so all of the model's cases
            print(f"[len2] {m.split('/')[-1]}: {len(d['y'])} prompts, layer {layer}, "
                  f"{len(np.unique(problems[idx]))} leave-one-problem-out fits", flush=True)
        data = with_data

        def len_feature(tr, te):
            return feat[:, None]

        def predict(tr, te):
            F = len_feature(tr, te)
            if args.len_with_seg:
                F = np.concatenate([F] + [diffmean_scores(acts, models, y, tr, None, w, 0.5) for w in ("seg_0_5", "seg_5_9")], 1)
            return _predict(_fit_logistic(F[tr], y[tr], args.C), F[te], True)

        return predict, (f"length probe: ridge(alpha={args.len_alpha}) last-prompt h @ layer frac {args.len_frac} -> mean log "
                         f"length of 10 sampled responses (8,192 cap), trained on sep2 prompts minus the case's own problem; "
                         f"models with data {data}"
                         f"{'; + seg_probe 0-5/5-9' if args.len_with_seg else ''}; logistic C={args.C}")

    def saplma(cases, args):
        """SAPLMA (Azaria & Mitchell, 2023): an MLP (256-128-64, ReLU) on one layer's hidden state.
        target "robust": fitted per model on the training cases' robustness labels (site --site, default
        resp_mean). target "correct": its own target, whether most of the model's sampled answers are
        correct, fitted on the sep2 prompt states (no robustness labels) and applied to the cases'
        last_prompt; each case scored by an MLP fitted without its problem's group (5 problem groups)."""
        from sklearn.model_selection import GroupKFold
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        acts = load_acts(cases, cv)
        y = cases.is_robust.astype(bool).to_numpy()
        models = cases.model_id.to_numpy()
        frac = 0.5 if args.layer_frac is None else args.layer_frac
        mlp = lambda: make_pipeline(StandardScaler(), MLPClassifier((256, 128, 64), alpha=args.saplma_alpha,
                                                                    max_iter=300, random_state=0))

        if args.saplma_target == "robust":
            site = args.site or "resp_mean"

            def predict(tr, te):
                feat = np.zeros(len(acts))
                for m in np.unique(models):
                    idx = np.where(models == m)[0]
                    tr_m = np.intersect1d(idx, tr)
                    if len(set(y[tr_m])) < 2:
                        continue
                    layer = layer_of(acts[idx[0]][site].shape[0], frac)
                    H = np.stack([acts[i][site][layer] for i in idx])
                    clf = mlp().fit(H[np.isin(idx, tr_m)], y[tr_m])
                    p = clf.predict_proba(H)[:, 1].clip(1e-4, 1 - 1e-4)
                    feat[idx] = np.log(p / (1 - p))
                F = feat[:, None]
                return _predict(_fit_logistic(F[tr], y[tr], args.C), F[te], True)

            return predict, f"SAPLMA MLP(256-128-64) per model on {site} @ layer frac {frac}, target robust; alpha={args.saplma_alpha}; logistic C={args.C}"

        problems = np.array([_norm(t) for t in cases.problem])
        feat = np.zeros(len(acts))
        with_data = []
        for m in np.unique(models):
            d = load_correct2(cv, m)
            if d is None:
                continue
            with_data.append(m.split("/")[-1])
            layer = layer_of(d["h"].shape[1], frac)
            X_all = d["h"][:, layer].astype(np.float32)
            idx = np.where(models == m)[0]
            case_q = np.unique(problems[idx])
            H = np.stack([acts[i]["last_prompt"][layer] for i in idx])
            for _, part in GroupKFold(5).split(case_q, groups=case_q):
                held = set(case_q[part])
                keep = np.array([q not in held for q in d["problem"]])
                clf = mlp().fit(X_all[keep], d["y"][keep])
                rows = np.isin(problems[idx], list(held))
                p = clf.predict_proba(H[rows])[:, 1].clip(1e-4, 1 - 1e-4)
                feat[idx[rows]] = np.log(p / (1 - p))
            print(f"[saplma] {m.split('/')[-1]}: {len(d['y'])} prompts ({d['y'].mean():.0%} majority-correct), layer {layer}", flush=True)

        def predict(tr, te):
            F = feat[:, None]
            if args.len_with_seg:
                F = np.concatenate([F] + [diffmean_scores(acts, models, y, tr, None, w, 0.5) for w in ("seg_0_5", "seg_5_9")], 1)
            return _predict(_fit_logistic(F[tr], y[tr], args.C), F[te], True)

        return predict, (f"SAPLMA MLP(256-128-64) on last-prompt h @ layer frac {frac}, target: majority of 10 sampled answers "
                         f"correct, fitted on sep2 prompts minus the case's problem group; models with data {with_data}; "
                         f"alpha={args.saplma_alpha}{'; + seg_probe 0-5/5-9' if args.len_with_seg else ''}; logistic C={args.C}")

    methods["saplma"] = ("whitebox", saplma)
    methods["len_probe"] = ("whitebox", len_probe)
    methods["sep2_probe"] = ("whitebox", sep2_probe)
    methods["act_probe"] = ("whitebox", act_probe)
    methods["seg_probe"] = ("whitebox", seg_probe)
    methods["sep_probe"] = ("whitebox", sep_probe)
    methods["perturb_direction"] = ("whitebox", perturb_direction)
    methods["perturb_direction_sup"] = ("whitebox", lambda c, a: perturb_direction(c, a, True))


def semantic_entropy(answers: list, none: str = "cluster") -> float | None:
    """Discrete semantic entropy: entropy of the fractions of samples per answer cluster. For integer
    answers, bidirectional entailment reduces to equality. ``none``: a sample with no parsed answer
    is one more cluster ("cluster") or is left out ("drop"; None if nothing is left)."""
    if none == "drop":
        answers = [a for a in answers if a is not None]
        if not answers:
            return None
    counts = np.unique(np.array(["NA" if a is None else str(a) for a in answers]), return_counts=True)[1]
    q = counts / counts.sum()
    return float(-(q * np.log(q)).sum())


def sep2_threshold(se: np.ndarray) -> float:
    """The paper's binarization: the split that minimizes the within-class squared error."""
    values = np.unique(se)
    if len(values) < 2:
        return float(values[0])
    best, gamma = np.inf, values[0]
    for lo, hi in zip(values[:-1], values[1:]):
        t = (lo + hi) / 2
        a, b = se[se <= t], se[se > t]
        sse = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        if sse < best:
            best, gamma = sse, t
    return float(gamma)


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def load_len2(cv, model_id: str):
    """Prompt states (every layer) with the mean log length of each prompt's sampled responses: the sep2
    prompts (originals + variants; rows aligned with sep2/<model>.npz the way generate.py run_sep2 built
    it, checked on the answers), else the originals of len/<model>.npz (generate.py --stages lens)."""
    key = ("len2", model_id)
    if key not in _CACHE:
        path = cv.DATA / "sep2" / f"{cv.safe(model_id)}.npz"
        samples = cv.DATA / "sep2" / "samples" / f"{cv.safe(model_id)}__default.jsonl"
        greedy_path = cv.DATA / "sep2" / "greedy" / f"{cv.safe(model_id)}__default.traces.jsonl"
        lens = cv.DATA / "len" / f"{cv.safe(model_id)}.npz"
        if not (path.is_file() and samples.is_file() and greedy_path.is_file()):
            if not lens.is_file():
                return None
            z = np.load(lens)
            _CACHE[key] = dict(h=z["h"], problem=np.array([_norm(t) for t in z["problem"]]), y=z["log_len"])
            return dict(_CACHE[key])
        labelgen = cv.ROOT / "data" / "labelgen"
        texts = {(r["problem_id"], "original", 0): r["problem"] for r in cv.read_jsonl(labelgen / "problems.jsonl")}
        texts.update({(r["problem_id"], r["type"], int(r["variant"])): r["text"] for r in cv.read_jsonl(labelgen / "variants.jsonl")})
        greedy = {(r["problem_id"], r["type"], int(r["variant"])) for r in cv.read_jsonl(greedy_path)}
        rows = [r for r in cv.read_jsonl(samples)
                if texts.get(k := (r["problem_id"], r["type"], int(r["variant"]))) and k in greedy]
        z = np.load(path)
        assert [json.loads(a) for a in z["answers"]] == [r["answers"] for r in rows], "sep2 row order changed"
        _CACHE[key] = dict(h=z["h_tbg"], problem=np.array([_norm(t) for t in z["problem"]]),
                           y=np.array([np.log1p(np.array(r["n_tokens"], float)).mean() for r in rows]),
                           correct=np.array([np.mean(r["correct"]) for r in rows]))
    return dict(_CACHE[key])


def load_correct2(cv, model_id: str):
    """sep2 prompt states with SAPLMA's target: whether most of the prompt's sampled answers are correct."""
    d = load_len2(cv, model_id)
    if d is None or "correct" not in d:
        return None
    return dict(h=d["h"], problem=d["problem"], y=d["correct"] >= 0.5)


def load_sep2(cv, model_id: str, none: str, answers: str = "sampled"):
    """``answers``: the sampled answers, or "forced" (truncated samples budget-forced, cvbench/force_answers.py)."""
    key = ("sep2", model_id, none, answers)
    if key not in _CACHE:
        path = cv.DATA / "sep2" / f"{cv.safe(model_id)}.npz"
        forced = cv.DATA / "sep2" / f"{cv.safe(model_id)}__forced.npz"
        if not path.is_file() or (answers == "forced" and not forced.is_file()):
            return None
        z = np.load(path)
        raw = np.load(forced)["answers"] if answers == "forced" else z["answers"]
        se = [semantic_entropy(json.loads(a), none) for a in raw]
        keep = np.array([s is not None for s in se])
        _CACHE[key] = dict(se=np.array([s for s in se if s is not None]), problem=z["problem"][keep],
                           h_tbg=z["h_tbg"][keep], h_slt=z["h_slt"][keep])
    d = dict(_CACHE[key])
    return d


def sep2_features(H: np.ndarray, layers: list[int]) -> np.ndarray:
    return np.concatenate([H[:, l].astype(np.float32) for l in layers], 1)


def sep2_select_layers(d: dict, pos: str, block: int) -> tuple[list[int], float]:
    """Per-layer probes, scored by held-out AUROC on the binarized SE (3-fold, grouped by problem, no
    robustness labels); the block of ``block`` adjacent layers with the best mean AUROC is kept."""
    d["h"] = d[f"h_{pos}"]
    target = d["se"] > sep2_threshold(d["se"])
    groups = d["problem"]
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=0).split(target, target, groups))
    n_layers = d["h"].shape[1]
    auc = np.full(n_layers, 0.5)
    for layer in range(1, n_layers):  # layer 0 is the embedding
        score = np.zeros(len(target))
        X = d["h"][:, layer].astype(np.float32)
        for tr, te in folds:
            score[te] = LogisticRegression(max_iter=1000).fit(X[tr], target[tr]).decision_function(X[te])
        auc[layer] = roc_auc_score(target, score)
    block = min(block, n_layers - 1)
    means = [auc[s:s + block].mean() for s in range(1, n_layers - block + 1)]
    start = 1 + int(np.argmax(means))
    layers = list(range(start, start + block))
    # held-out AUROC of the concatenated block itself
    score = np.zeros(len(target))
    X = sep2_features(d["h"], layers)
    for tr, te in folds:
        score[te] = LogisticRegression(max_iter=1000).fit(X[tr], target[tr]).decision_function(X[te])
    return layers, float(roc_auc_score(target, score))


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
