"""Where Skywork traces attend: share of attention per region, robust vs fragile. Layers 6-13, all heads, deciles 1-9.

    python3 analysis/report/fig_regions.py > analysis/report/fig_regions.txt
"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score as auc
HERE = Path(__file__).parent; ROOT = HERE.parents[1]; DATA = ROOT / "data/cvbench"
safe = lambda m: m.replace("/", "_").replace(":", "_")
d = np.load(HERE / "figure_data.npz", allow_pickle=True)
cases = [json.loads(l) for f in ("cases.jsonl", "cases_oss.jsonl") for l in open(DATA / f)]
sky, ys, caps = d["sky"], d["ys"], d["caps"]
R = np.stack([np.load(DATA / "circuit_attn" / safe(cases[i]["model_id"]) / f"{cases[i]['group']}.npz")["regions"][6:].mean((0, 1)) for i in sky])
R = R[:, :9].mean(1)    # cases x [sink, template, problem, far, recent]
A = np.c_[R[:, 0] + R[:, 1], R[:, 2], R[:, 3], R[:, 4]]
NAMES = ["BOS + template", "Problem text", "Own reasoning,\n>64 tokens back", "Own reasoning,\nlast 64 tokens"]
print("share sums", A.sum(1).min().round(3), A.sum(1).max().round(3))
rng = np.random.default_rng(0)


def ci(a, n=4000):
    return np.quantile(a[rng.integers(0, len(a), (n, len(a)))].mean(1), [.025, .975], axis=0)


def p_perm(v, l, n=20000):
    o = auc(l, v); a = np.array([auc(rng.permutation(l), v) for _ in range(n)])
    return o, ((np.abs(a - .5) >= abs(o - .5)).sum() + 1) / (n + 1)


for tag, m in (("all", np.ones(len(ys), bool)), ("capped", caps)):
    print(tag, "n", m.sum(), "robust", (ys & m).sum(), "fragile", (~ys & m).sum())
    for k, n in enumerate(NAMES):
        o, p = p_perm(A[m, k], ys[m]); r, f = A[m & ys, k].mean(), A[m & ~ys, k].mean()
        print(f"  {n!r:36s} R {r:.3f} F {f:.3f} F-R {f - r:+.3f} AUC robust-higher {o:.3f} p={p:.3f}")

INK, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e6e5e0", "#a3a29b"
ROB, FRAG = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "pdf.fonttype": 42, "text.color": INK, "axes.edgecolor": NEUTRAL,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK, "axes.linewidth": .6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 7,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.axisbelow": True})
fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.3), sharex=True, sharey=True, gridspec_kw={"wspace": .1})
H = .36
for ax, m, title in zip(axs, (np.ones(len(ys), bool), caps), ("All traces", "Capped traces")):
    for cls, col, name, off in ((True, ROB, "Robust", -H / 2 - .01), (False, FRAG, "Fragile", H / 2 + .01)):
        a = A[m & (ys == cls)]; v = a.mean(0); lo, hi = ci(a); pos = np.arange(4) + off
        ax.barh(pos, v, H, color=col, label=f"{name} ($n$={len(a)})", zorder=2)
        ax.errorbar(v, pos, xerr=[v - lo, hi - v], fmt="none", ecolor=INK, elinewidth=.6, capsize=0, zorder=3)
        for y_, h_, v_ in zip(pos, hi, v):
            ax.text(h_ + .008, y_, f"{v_:.3f}", va="center", fontsize=6.5, color=INK)
    ax.set_title(title); ax.set_xlabel("Share of attention, deciles 1–9"); ax.grid(axis="x", color=GRID, lw=.5); ax.set_xlim(0, .52)
    ax.set_xticks([0, .1, .2, .3, .4, .5])
axs[0].set_yticks(range(4)); axs[0].set_yticklabels(NAMES); axs[0].invert_yaxis()
for ax in axs: ax.legend(loc="upper right", borderaxespad=.1, handletextpad=.4, handlelength=1)
out = HERE / "figures"
fig.savefig(out / "fig_regions.pdf", bbox_inches="tight"); fig.savefig(out / "fig_regions.png", dpi=240, bbox_inches="tight", facecolor="white")
