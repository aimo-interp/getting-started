"""Attention on cases the stack gets right: caught fragile (TP) vs correctly passed robust (TN).
Outcomes come from the stack WITHOUT attention (ablation 'attn -attn (tx)'), so selection does not use attention.
Capped traces only, since TN otherwise includes easy finished traces. Features z-scored within model on all capped cases.
Null: shuffle TP/TN within model, max |AUC-.5| across features. FN shown for comparison.

    python3 analysis/report/attn_tp_tn.py <fig_dir> > analysis/report/attn_tp_tn.txt
"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as auc
HERE = Path(__file__).parent; ROOT = HERE.parents[1]; DATA = ROOT / "data/cvbench"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "figures"
safe = lambda m: m.replace("/", "_").replace(":", "_")
c_ = {json.loads(l)["case_id"]: json.loads(l) for f in ("cases.jsonl", "cases_oss.jsonl") for l in open(DATA / f)}
d = pd.read_csv(HERE / "errors_stack_cases.csv", index_col=0)
A = np.load(HERE / "ablation_candidates.npz", allow_pickle=True); Z = np.load(HERE / "errors_oof.npz", allow_pickle=True)
ids = Z["case_id"].astype(str); assert (A["y"] == Z["y"]).all()
P = pd.DataFrame(A["attn -attn (tx)"].T, index=ids).reindex(d.index)
err = ((P.values >= .5) != d.robust.values[:, None]).mean(1); wrong = err >= .5
d["out_na"] = np.where(d.frag, np.where(wrong, "FN", "TP"), np.where(wrong, "FP", "TN"))
print("stack without attention:", d.out_na.value_counts().to_dict(), "| same outcome as with attention on", int((d.out_na == d.out).sum()), "of", len(d))
REG = {"BOS + template": lambda R: R[..., 0] + R[..., 1], "problem": lambda R: R[..., 2], "far reasoning": lambda R: R[..., 3], "last 64": lambda R: R[..., 4]}
rows = []; curves = {r: [] for r in REG}
for i, r in d[d.attn_data & d.capped].iterrows():
    c = c_[i]; z = np.load(DATA / "circuit_attn" / safe(c["model_id"]) / f"{c['group']}.npz")
    Q = np.nanmean(z["regions"] if "layers" in z else z["regions"][6:], (0, 1)); o = {"case": i, "model": r.model, "out": r.out_na, "chars": r.chars}
    for k, f in REG.items():
        v = f(Q); curves[k].append(v)
        for w, (a, b) in {"d1-5": (0, 5), "d6-9": (5, 9), "all": (0, 10)}.items(): o[f"{k} {w}"] = v[a:b].mean()
    rows.append(o)
t = pd.DataFrame(rows).set_index("case"); F = [c for c in t if c not in ("model", "out", "chars")]
print(pd.crosstab(t.model, t.out).reindex(columns=["TP", "FN", "FP", "TN"], fill_value=0).to_string())
z = t[F].groupby(t.model).transform(lambda v: (v - v.mean()) / (v.std() + 1e-9))
s = t.out.isin(["TP", "TN"]); rng = np.random.default_rng(0)


def run(mask, tag, X=z):
    x = X[mask]; l = (t.out[mask] == "TN").values; m = t.model[mask].values
    obs = np.array([auc(l, x[c]) for c in F]); N = 5000; mx = np.zeros(N); ind = np.zeros((N, len(F)))
    Xv = x[F].values
    from scipy.stats import rankdata
    R = rankdata(Xv, axis=0); n1 = l.sum(); n0 = len(l) - n1
    for p in range(N):
        q = l.copy()
        for g in np.unique(m):
            j = m == g; q[j] = rng.permutation(q[j])
        a = (R[q].sum(0) - n1 * (n1 + 1) / 2) / (n0 * n1); ind[p] = a; mx[p] = np.abs(a - .5).max()
    print(f"\n== {tag}: TN {n1}, TP {n0}; AUC = TN higher; p raw two-sided, p family across {len(F)} features")
    for k, c in enumerate(F):
        pr = (np.abs(ind[:, k] - .5) >= abs(obs[k] - .5) - 1e-12).mean(); pf = (mx >= abs(obs[k] - .5) - 1e-12).mean()
        print(f"  {c:24s} AUC {obs[k]:.3f}  p {pr:.3f}  family p {pf:.3f}")


run(s, "all four models, capped")
run(s & (t.model.str.startswith("Skywork")), "Skywork only, capped")
run(s & ~t.model.str.startswith("Skywork"), "Olmo + DeepSeek + Qwen3.5, capped")
L = np.log(t.chars); R_ = z.copy()
for g, h in t.groupby("model"):
    D = np.c_[L[h.index], np.ones(len(h))]; R_.loc[h.index, F] = z.loc[h.index, F].values - D @ np.linalg.lstsq(D, z.loc[h.index, F].values, rcond=None)[0]
run(s, "all four, capped, log problem length removed", R_)
print("\n== mean within-model z by outcome, including misses and false alarms")
print(z.groupby(t.out)[["BOS + template d6-9", "far reasoning all", "problem all", "last 64 all"]].mean().round(2).reindex(["TP", "FN", "FP", "TN"]).to_string())
print(t.groupby(["model", "out"])[["BOS + template d6-9", "far reasoning all", "problem all", "last 64 all"]].mean().round(3).to_string())

INK, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e6e5e0", "#a3a29b"; ROB, FRAG = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "pdf.fonttype": 42, "text.color": INK, "axes.edgecolor": NEUTRAL, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.linewidth": .6, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": .5,
    "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 7, "xtick.major.size": 0, "ytick.major.size": 0, "lines.linewidth": 1.4})
fig, ax = plt.subplots(1, 4, figsize=(5.5, 1.9), sharey=True, gridspec_kw={"wspace": .14}); x = np.arange(1, 11)
for a, (k, C) in zip(ax, curves.items()):
    C = pd.DataFrame(np.stack(C), index=t.index); C = C.groupby(t.model).transform(lambda v: (v - v.mean()) / (v.std() + 1e-9))
    for o, col, mk, nm in (("TN", ROB, "o", "Robust, passed (TN)"), ("TP", FRAG, "^", "Fragile, caught (TP)")):
        v = C[t.out == o].values; lo, hi = np.quantile(v[rng.integers(0, len(v), (4000, len(v)))].mean(1), [.025, .975], axis=0)
        a.fill_between(x, lo, hi, color=col, alpha=.14, lw=0); a.plot(x, v.mean(0), color=col, marker=mk, ms=2.8, mec="white", mew=.4, label=f"{nm}, $n$={len(v)}")
    a.axhline(0, color=NEUTRAL, lw=.7); a.set_title(k); a.set_xticks([1, 5, 10]); a.set_xlabel("Decile")
ax[0].set_ylabel("Attention (within-model SD)"); h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.24), columnspacing=1.2, handletextpad=.4)
fig.savefig(OUT / "fig_attn_tp_tn.pdf", bbox_inches="tight"); fig.savefig(OUT / "fig_attn_tp_tn.png", dpi=240, bbox_inches="tight", facecolor="white")

# ---------------------------------------------------------------- two regions, attention share, no bands
fig, ax = plt.subplots(1, 2, figsize=(4.6, 2.0), gridspec_kw={"wspace": .36}); x = np.arange(1, 11)
print("\n== mean attention share by decile, capped; model mean removed, +- share points x 100")
for a, k in zip(ax, ("BOS + template", "problem")):
    C = pd.DataFrame(np.stack(curves[k]), index=t.index)
    C = C - C.groupby(t.model).transform("mean").mean(1).values[:, None]      # remove each model's level, keep the decile shape
    C = C - C.mean(0)                                                          # shared time course removed: gap around the average trace
    for o, col, mk, nm in (("TN", ROB, "o", "Robust, passed (TN)"), ("TP", FRAG, "^", "Fragile, caught (TP)")):
        v = C[t.out == o].mean(0).values * 100
        a.plot(x, v, color=col, marker=mk, ms=3, mec="white", mew=.4, label=f"{nm}, $n$={(t.out == o).sum()}"); print(f"{k:16s} {o} " + " ".join(f"{q:+.2f}" for q in v))
    a.axhline(0, color=NEUTRAL, lw=.7); a.set_title(k[0].upper() + k[1:]); a.set_xticks([1, 3, 5, 7, 9]); a.set_xlabel("Decile of the chain of thought")
ax[0].set_ylabel("Attention share, points from mean"); h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.2), columnspacing=1.2, handletextpad=.4)
fig.savefig(OUT / "fig_attn_tp_tn_two.pdf", bbox_inches="tight"); fig.savefig(OUT / "fig_attn_tp_tn_two.png", dpi=240, bbox_inches="tight", facecolor="white")
o = t[t.out.isin(["TP", "TN"])]
print("\nproblem chars median", o.groupby("out").chars.median().to_dict(), "| by model"); print(o.groupby(["model", "out"]).chars.median().to_string())
print("within-model corr with log chars, capped:", {k: round(float(np.corrcoef(np.log(t.chars) - np.log(t.chars).groupby(t.model).transform("mean"), z[k])[0, 1]), 2) for k in ("BOS + template all", "problem all")})
lc = np.log(o.chars) - np.log(o.chars).groupby(o.model).transform("mean"); print("within-model log chars AUC, TN higher %.3f" % auc(o.out == "TN", lc))
print("raw whole-trace share by model:"); print(o.groupby(["model", "out"])[["BOS + template all", "problem all"]].mean().round(3).to_string())
