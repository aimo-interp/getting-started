"""Readout check: projection of the ORIGINAL prompt's (or CoT-mean) activation onto the
label-free perturbation direction r = mean(h(dX)) - mean(h(X)), per model and layer frac."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv, whitebox
from sklearn.metrics import roc_auc_score

cases = cv.load_cases()
acts = whitebox.load_acts(cases, cv)
y = cases.is_robust.astype(bool).to_numpy()
models = cases.model_id.to_numpy()
for site in ("last_prompt", "resp_mean"):
    for frac in (0.2, 0.4, 0.5, 0.6, 0.8):
        feat = np.zeros(len(acts))
        for m in np.unique(models):
            idx = np.where(models == m)[0]
            l = whitebox.layer_of(acts[idx[0]]["last_prompt"].shape[0], frac)
            hx = np.stack([acts[i]["last_prompt"][l] for i in idx])
            hdx = np.stack([acts[i]["pert_last"][:, l] for i in idx]).mean(1)
            r = (hdx - hx).mean(0); r /= np.linalg.norm(r)
            h = np.stack([acts[i][site][l] for i in idx])
            p = (h - h.mean(0)) @ r
            feat[idx] = p / (p.std() + 1e-8)
        # label-free score: higher projection -> "more perturbed-looking" -> less robust
        aucs = {m.split('/')[-1][:8]: roc_auc_score(y[models == m], -feat[models == m]) for m in np.unique(models)}
        print(f"{site:11s} frac={frac} pooled AUC={roc_auc_score(y, -feat):.3f}  " + " ".join(f"{k}={v:.2f}" for k, v in aucs.items()))
