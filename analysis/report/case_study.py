"""Group view of the blocks on stack outcomes, and a caught / missed pair on Skywork, decile by decile.
Seg: diff-in-means on Skywork with every case of the problem held out, windows d1-5 and d6-9 as in the probe; deciles projected on
those directions, z-scored on training projections. Traj: log step and norm, z-scored on the other Skywork cases. Attention: layers 6-13.

    python3 analysis/report/case_study.py > analysis/report/case_study.txt
"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as auc
HERE = Path(__file__).parent; ROOT = HERE.parents[1]; DATA = ROOT / "data/cvbench"
safe = lambda m: m.replace("/", "_").replace(":", "_")
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)
d = pd.read_csv(HERE / "errors_stack_cases.csv", index_col=0)
c_ = {json.loads(l)["case_id"]: json.loads(l) for f in ("cases.jsonl", "cases_oss.jsonl") for l in open(DATA / f)}
g = {}
for i in d.index:
    c = c_[i]; s = np.load(DATA / "segs" / safe(c["model_id"]) / f"{c['group']}.npz")["seg_mean"]; g[i] = s[:, int(round(.5 * (s.shape[1] - 1)))].astype(np.float64)
step = {i: np.log(np.linalg.norm(np.diff(v[:9], axis=0), axis=1) + 1e-12) for i, v in g.items()}
norm = {i: (lambda n: n / n.mean())(np.linalg.norm(v[:9], axis=1)) for i, v in g.items()}
d["step"] = [step[i].mean() for i in d.index]; d["slope"] = [np.polyfit(np.arange(9), norm[i], 1)[0] for i in d.index]
C = np.load(HERE / "errors_oof.npz", allow_pickle=True); 
def attn(i):
    c = c_[i]; f = DATA / "circuit_attn" / safe(c["model_id"]) / f"{c['group']}.npz"
    if not f.is_file(): return np.full((10, 5), np.nan)
    z = np.load(f); return np.nanmean(z["regions"] if "layers" in z else z["regions"][6:], (0, 1))
A = {i: attn(i) for i in d.index}
d["sink"] = [np.nanmean(A[i][5:9, 0] + A[i][5:9, 1]) if not np.isnan(A[i]).all() else np.nan for i in d.index]
d["far"] = [np.nanmean(A[i][:9, 3]) if not np.isnan(A[i]).all() else np.nan for i in d.index]
# held-out seg score, one per case
def direction(idx, a, b):
    X = np.stack([g[i][a:b].mean(0) for i in idx]); y = d.loc[idx, "robust"].values
    w = X[y].mean(0) - X[~y].mean(0); w /= np.linalg.norm(w) + 1e-12; p = X @ w; return w, p.mean(), p.std() + 1e-12
d["seg"] = np.nan
for m, h in d.groupby("model"):
    if h.frag.sum() < 2: continue
    for i in h.index:
        tr = [j for j in h.index if d.group[j] != d.group[i]]
        if d.loc[tr, "frag"].sum() and (~d.loc[tr, "frag"]).sum():
            v = []
            for a, b in ((0, 5), (5, 9)):
                w, mu, sd = direction(tr, a, b); v.append((g[i][a:b].mean(0) @ w - mu) / sd)
            d.loc[i, "seg"] = np.mean(v)
F = ["seg", "step", "slope", "sink", "far"]
Z = d[F].groupby(d.model).transform(lambda v: (v - v.mean()) / (v.std() + 1e-9))
cap = d.capped; org = d.source == "train-main-v2"
print("== mean within-model z by outcome, capped traces"); print(Z[cap].groupby(d.out[cap]).mean().round(2).reindex(["TP", "FN", "FP", "TN"]).assign(n=d[cap].groupby("out").size()).to_string())
print("== same, organiser cases only"); print(Z[cap & org].groupby(d.out[cap & org]).mean().round(2).reindex(["TP", "FN", "FP", "TN"]).assign(n=d[cap & org].groupby("out").size()).to_string())
print("== capped: FN vs TN, TP vs FN, FP vs TN AUC, first group higher")
for nm, a, b in (("FN vs TN", "FN", "TN"), ("TP vs TN", "TP", "TN"), ("TP vs FN", "TP", "FN"), ("FP vs TN", "FP", "TN")):
    print(f"  {nm}: " + "  ".join(f"{k} {auc(*(lambda s: ((d.out[s.index] == a).values, s.values))(Z[k][cap & d.out.isin([a, b])].dropna())):.2f}" for k in F))
for i, tag in (("Skywork-OR1-Math-7B:335cfaaf1b6e", "CAUGHT"), ("Skywork-OR1-Math-7B:7ee9d4437899", "MISSED")):
    h = d[d.model == d.model[i]]; tr = [j for j in h.index if d.group[j] != d.group[i]]
    print(f"\n#### {tag} {i}: p_stack {d.p_stack[i]:.2f} wrong rate {d.err_rate[i]:.2f} drop {d.max_drop[i]} solve {d.solve[i]} tokens {d.tokens[i]} chars {d.chars[i]}")
    print("block ranks among other Skywork cases (share of Skywork cases with a higher z; robust-looking is high):", {k: round(float((Z.loc[tr, k] > Z.loc[i, k]).mean()), 2) for k in F})
    print("z:", Z.loc[i].round(2).to_dict())
    for lab, (a, b) in {"early": (0, 5), "late": (5, 9)}.items():
        w, mu, sd = direction(tr, a, b); print(f"  seg {lab} per decile:", " ".join(f"{(g[i][k] @ w - mu) / sd:+.2f}" for k in range(10)))
    S = np.stack([step[j] for j in tr]); print("  step z:", " ".join(f"{q:+.2f}" for q in (step[i] - S.mean(0)) / S.std(0)))
    N = np.stack([norm[j] for j in tr]); print("  norm z:", " ".join(f"{q:+.2f}" for q in (norm[i] - N.mean(0)) / N.std(0)))
    Aq = np.stack([A[j] for j in tr]); 
    for r, nm in (((0, 1), "BOS+template"), ((2,), "problem"), ((3,), "far"), ((4,), "last64")):
        v = A[i][:, list(r)].sum(1); m_ = Aq[:, :, list(r)].sum(2); print(f"  attn {nm:13s}", " ".join(f"{q:.3f}" for q in v), "| z", " ".join(f"{q:+.1f}" for q in (v - m_.mean(0)) / m_.std(0)))
d[F + ["out", "model", "source", "capped"]].join(Z, rsuffix="_z").to_csv(HERE / "case_study_blocks.csv")
