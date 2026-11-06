"""Group test of the link between self-correction and the blocks on Skywork, and the case-pair figure.

    /usr/bin/python3 analysis/report/case_figure.py <fig_dir> > analysis/report/case_figure.txt
"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import report_figstyle as rs
HERE = Path(__file__).parent; ROOT = HERE.parents[1]; DATA = ROOT / "data/cvbench"; OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "figures"
safe = lambda m: m.replace("/", "_").replace(":", "_"); SK = "Skywork/Skywork-OR1-Math-7B"
d = pd.read_csv(HERE / "errors_stack_cases.csv", index_col=0); s = d[d.model.str.startswith("Skywork")]
W = pd.read_csv(HERE / "case_wait_counts.csv", index_col=0).reindex(s.group).values.astype(float); y = s.robust.values; grp = s.group.values
X = []
for g in grp:
    v = np.load(DATA / "segs" / safe(SK) / f"{g}.npz")["seg_mean"]; X.append(v[:, int(round(.5 * (v.shape[1] - 1)))].astype(np.float64))
X = np.stack(X)                                                       # [cases, deciles, d]
P = np.zeros((len(y), 10))                                            # held-out projection, one direction per window, averaged
for i in range(len(y)):
    tr = np.arange(len(y)) != i; v = []
    for a, b in ((0, 5), (5, 9)):
        A = X[tr][:, a:b].mean(1); w = A[y[tr]].mean(0) - A[~y[tr]].mean(0); w /= np.linalg.norm(w); p = A @ w; v.append((X[i] @ w - p.mean()) / p.std())
    P[i] = np.mean(v, 0)
St = np.log(np.linalg.norm(np.diff(X[:, :9], axis=1), axis=2)); Wm = (W[:, :-2] + W[:, 1:-1]) / 2        # step k -> k+1, waits in the two deciles
def within(a, b):
    a = a - a.mean(0, keepdims=True); b = b - b.mean(0, keepdims=True)       # remove position
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)       # remove case
    return np.corrcoef(a.ravel(), b.ravel())[0, 1]
def perm(a, b, n=5000, rng=np.random.default_rng(0)):
    o = within(a, b); nl = np.array([within(a, b[rng.permutation(len(b))]) for _ in range(n)]); return o, (np.abs(nl) >= abs(o)).mean()
def within_case(a, b):
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True); return np.corrcoef(a.ravel(), b.ravel())[0, 1]
print("Skywork, 48 traces: decile-level correlation with Wait count, removing case and decile means")
print("  seg score: r %.3f p %.3f (shuffling traces) | within case only r %.3f" % (perm(P[:, :9], W[:, :9]) + (within_case(P[:, :9], W[:, :9]),)))
print("  log step : r %.3f p %.3f | within case only r %.3f" % (perm(St, Wm) + (within_case(St, Wm),)))
tot = W[:, :9].sum(1); print("across cases: Wait total vs mean seg r %.3f, vs mean step r %.3f; Wait total robust %.1f fragile %.1f" % (np.corrcoef(tot, P[:, :9].mean(1))[0, 1], np.corrcoef(tot, St.mean(1))[0, 1], tot[y].mean(), tot[~y].mean()))
from sklearn.metrics import roc_auc_score as auc
print("Wait total AUC for robust (fewer = robust) %.3f, capped %.3f" % (auc(y, -tot), auc(y[s.capped.values], -tot[s.capped.values])))
Sz = (St - St.mean(0)) / St.std(0)
CASES = (("335cfaaf1b6e", "Fragile, predicted fragile", rs.FRAG, "^"), ("7ee9d4437899", "Fragile, predicted robust", rs.ROB, "o"))
rs.apply()
fig, ax = plt.subplots(2, 2, figsize=(rs.WIDTH, 2.75), sharex=True, gridspec_kw={"wspace": .48, "hspace": .14}); ax = ax.ravel(); x = np.arange(1, 11)
for g, nm, col, mk in CASES:
    i = np.where(grp == g)[0][0]; kw = dict(color=col, marker=mk)
    ax[0].plot(x, W[i], label=nm, **kw); ax[1].plot(x[:9], P[i, :9], **kw); ax[2].plot(np.arange(1.5, 9.5), Sz[i], **kw)
    z = np.load(DATA / "circuit_attn" / safe(SK) / f"{g}.npz")["regions"][6:].mean((0, 1)); ax[3].plot(x, z[:, 2], **kw)
    print(nm, "seg", np.round(P[i], 2), "step z", np.round(Sz[i], 2))
ax[3].plot(x, np.mean([np.load(DATA / "circuit_attn" / safe(SK) / f"{g}.npz")["regions"][6:].mean((0, 1))[:, 2] for g in grp], 0), color=rs.GREY, lw=1, ls="--", label="Mean of 48 Skywork cases")
for a, yl in zip(ax, ('Count of "Wait"', "SegProbe ($z$)", "Log step ($z$)", "Problem share")):
    a.set_ylabel(yl, labelpad=2, fontsize=9); a.set_xticks([1, 5, 9])
ax[2].set_xlabel("Trace decile"); ax[3].set_xlabel("Trace decile")
ax[1].axhline(0, color="k", lw=.6, ls=":"); ax[2].axhline(0, color="k", lw=.6, ls=":")
h = ax[0].get_legend_handles_labels(); h2 = ax[3].get_legend_handles_labels()
fig.legend(h[0] + h2[0], h[1] + h2[1], loc="upper center", ncol=1, bbox_to_anchor=(.5, 1.13))
rs.save(fig, OUT, "fig_case")
