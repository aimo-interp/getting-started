"""What do the top principal components of the seg activations encode, per model? (dead-end audit 2026-11-02)
- variance share of the top PCs (n cases per model is 7-48, so PCs are estimated from very few points)
- AUC of each PC for robustness, |corr| with greedy log length and with label-free difficulty
- share of the full-data DIM direction's squared norm inside the top-k PC subspace
- out-of-fold seg-score AUC per model when the top-k PCs are removed (no gate), k = 0..5
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cvbench"))
import cv, stack, whitebox as wb, difficulty

cases = cv.load_cases(list(cv.CASE_FILES))
y = cases.is_robust.astype(bool).to_numpy()
models, groups = cases.model_id.to_numpy(), cases.group.to_numpy()
acts = wb.load_acts(cases, cv)
n_tok = {}
for f in (cv.DATA / "greedy").glob("*.jsonl"):
    for line in open(f):
        r = json.loads(line); n_tok[(r["model_id"], r["group"])] = r["generation_num_tokens"]
loglen = np.log1p(np.array([n_tok.get((c.model_id, c.group), np.nan) for c in cases.itertuples()], float))
diff = difficulty.diff_feats(cases, ("diff_last",)).diff_last.to_numpy()

def auc(yy, s):
    return roc_auc_score(yy, s) if 0 < yy.sum() < len(yy) else np.nan
def cor(a, b):
    k = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[k], b[k])[0, 1] if k.sum() > 2 else np.nan

for site in ("seg_0_5", "seg_5_9"):
    print(f"\n######## {site} (layer frac .5)")
    print(f"{'model':28s} n  frag | var share PC1..5            | AUC(robust) PC1..5 (0.5=none) | |r| loglen PC1..3 | |r| diff PC1..3 | DIM norm² in top1/3/5")
    for m in np.unique(models):
        idx = np.where(models == m)[0]
        L = acts[idx[0]][site].shape[0]; layer = wb.layer_of(L, .5)
        H = np.stack([acts[i][site][layer] for i in idx]).astype(np.float64)
        Hc = H - H.mean(0)
        U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
        var = S**2 / (S**2).sum()
        P = Hc @ Vt[:5].T
        yy = y[idx]
        aucs = [auc(yy, P[:, j]) for j in range(min(5, P.shape[1]))]
        d = Hc[yy].mean(0) - Hc[~yy].mean(0)
        dn = (d @ Vt.T) ** 2 / (d @ d)
        print(f"{m.split('/')[-1]:28s} {len(idx):2d} {(~yy).sum():2d}  | " + " ".join(f"{v:.2f}" for v in var[:5]) +
              " | " + " ".join(f"{max(a,1-a):.2f}{'+' if a>=.5 else '-'}" for a in aucs) +
              " | " + " ".join(f"{abs(cor(P[:,j], loglen[idx])):.2f}" for j in range(3)) +
              " | " + " ".join(f"{abs(cor(P[:,j], diff[idx])):.2f}" for j in range(3)) +
              f" | {dn[:1].sum():.2f} {dn[:3].sum():.2f} {dn[:5].sum():.2f}")

print("\n######## out-of-fold seg-score AUC per model, top-k PCs removed (ungated; windows averaged); 10 repeats")
ks = [0, 1, 2, 3, 5]
res = {k: np.zeros((10, len(y))) for k in ks}
for r in range(10):
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(y, y, groups):
        for k in ks:
            s = np.mean([stack.dim_variant_scores(acts, models, y, tr, site, .5, pcs=k)[:, 0]
                         for site in ("seg_0_5", "seg_5_9")], 0)
            res[k][r, te] = s[te]
ntr = {m.split('/')[-1]: int(round((models == m).sum() * .8)) for m in np.unique(models)}
print(f"{'model':28s} n_train " + "  ".join(f"k={k}" for k in ks))
for m in np.unique(models):
    k_ = models == m
    print(f"{m.split('/')[-1]:28s} {ntr[m.split('/')[-1]]:5d}   " +
          "  ".join(f"{np.mean([auc(y[k_], res[k][r][k_]) for r in range(10)]):.2f}" for k in ks))
