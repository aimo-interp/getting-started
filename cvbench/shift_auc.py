"""Per-(model, perturbation type) shift scores vs labels: s_proj = d.r, s_resid = |d - (d.r)r|, |d|.

d = h(dX) - h(X) at the last prompt token; r is the label-free mean shift of that model and
perturbation type, computed leaving the scored problem out. Reports pooled AUC (robust = positive,
so AUC < 0.5 means a bigger shift goes with NOT robust), the paired sign check P(d.r > 0), and the
same AUC with r replaced by the first PC of the shifts and by random unit directions.
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv, whitebox

cases = cv.load_cases()
acts = whitebox.load_acts(cases, cv)
types = [np.load(cv.acts_file(c.model_id, c.group))["pert_types"] for c in cases.itertuples()]
y = cases.is_robust.astype(bool).to_numpy()
models = cases.model_id.to_numpy()
rng = np.random.default_rng(0)


def scores(frac, per_type, direction="r"):
    out = {k: np.full(len(acts), np.nan) for k in ("proj", "resid", "norm")}
    signs = []
    for m in np.unique(models):
        idx = np.where(models == m)[0]
        l = whitebox.layer_of(acts[idx[0]]["last_prompt"].shape[0], frac)
        kinds = sorted({t for i in idx for t in types[i]}) if per_type else [None]
        acc = {k: np.zeros(len(idx)) for k in out}
        cnt = np.zeros(len(idx))
        for kind in kinds:
            rows = [(j, np.where(types[i] == kind)[0] if kind else np.arange(len(types[i]))) for j, i in enumerate(idx)]
            D = [acts[idx[j]]["pert_last"][ks, l] - acts[idx[j]]["last_prompt"][l] for j, ks in rows]
            for j, ks in rows:
                if not len(ks):
                    continue
                pool = [D[o] for o in range(len(idx)) if o != j and len(D[o])]
                if len(pool) < 2:
                    continue
                others = np.concatenate(pool)
                if direction == "r":
                    r = others.mean(0)
                elif direction == "pc1":
                    r = np.linalg.svd(others - others.mean(0), full_matrices=False)[2][0]
                else:
                    r = rng.standard_normal(others.shape[1])
                r /= np.linalg.norm(r) + 1e-8
                d = D[j]
                p = d @ r
                signs += list(p > 0)
                scale = np.linalg.norm(acts[idx[j]]["last_prompt"][l]) + 1e-8
                acc["proj"][j] += p.mean() / scale
                acc["resid"][j] += np.linalg.norm(d - p[:, None] * r, axis=1).mean() / scale
                acc["norm"][j] += np.linalg.norm(d, axis=1).mean() / scale
                cnt[j] += 1
        for k in out:
            v = acc[k] / np.maximum(cnt, 1)
            out[k][idx] = (v - v.mean()) / (v.std() + 1e-8)  # within-model z-score, label-free
    return out, np.mean(signs)


for frac in (0.25, 0.5, 0.75):
    for per_type in (False, True):
        s, sign = scores(frac, per_type)
        base = {d: scores(frac, per_type, d)[0]["proj"] for d in ("pc1", "rand")}
        print(f"frac={frac} r per {'type ' if per_type else 'model'}  P(d.r>0)={sign:.2f}  "
              f"AUC proj={roc_auc_score(y, s['proj']):.3f} resid={roc_auc_score(y, s['resid']):.3f} "
              f"norm={roc_auc_score(y, s['norm']):.3f} | proj on pc1={roc_auc_score(y, base['pc1']):.3f} "
              f"rand={roc_auc_score(y, base['rand']):.3f}")
