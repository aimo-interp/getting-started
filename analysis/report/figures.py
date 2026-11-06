"""Report figures: seg probe, trajectory, attention sink. Data from figure_data.py; nothing is refitted here.

    python3 analysis/report/figures.py <out_dir>
"""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import report_figstyle as rs
HERE = Path(__file__).parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
OUT.mkdir(parents=True, exist_ok=True)
d = np.load(HERE / "figure_data.npz", allow_pickle=True)
INK, MUTED, GRID, NEUTRAL = "black", "black", "0.85", rs.GREY
ROB, FRAG = rs.ROB, rs.FRAG
rs.apply()
RNG = np.random.default_rng(0)


def ci(a, n=4000):
    b = a[RNG.integers(0, len(a), (n, len(a)))].mean(1)
    return np.quantile(b, [.025, .975], axis=0)


def two(ax, x, A, y, label=True, band=False):
    for cls, col, mk, name in ((True, ROB, "o", "Robust"), (False, FRAG, "^", "Fragile")):
        a = A[y == cls]; lo, hi = ci(a)
        if band: ax.fill_between(x, lo, hi, color=col, alpha=.2, lw=0)
        ax.plot(x, a.mean(0), color=col, marker=mk, label=f"{name} ($n$={len(a)})")


def save(fig, name):
    rs.save(fig, OUT, name)


# ---------------------------------------------------------------- seg probe
y, cfg = d["y"], d["cfg"].astype(str)
fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 2.35), gridspec_kw={"width_ratios": [1, 1.2], "wspace": .46})
x = np.arange(1, 11); lo, hi = d["band"]
a.fill_between(x, lo, hi, color=NEUTRAL, alpha=.28, lw=0, label="Shuffled labels, 90% band")
a.axhline(.5, color=NEUTRAL, lw=.7)
a.plot(x, d["dec_auc"], color=ROB, marker="o", ms=3.2, mec="white", mew=.4, label="Out-of-fold $\\mathtt{SegProbe}$")
a.set(xlabel="Decile of the chain of thought", ylabel="AUC, robust vs fragile", xticks=x, ylim=(.3, .85))
a.legend(loc="lower left", borderaxespad=.1)
ORDER = [("gpt-oss-120b:medium", "gpt-oss medium"), ("Qwen3.5-4B", "Qwen3.5"), ("Skywork-OR1-Math-7B", "Skywork"), ("Olmo-3-7B-Think", "Olmo"),
         ("gpt-oss-120b:low", "gpt-oss low"), ("DeepSeek-R1-0528-Qwen3-8B", "DeepSeek"), ("gpt-oss-120b:high", "gpt-oss high")]
from sklearn.metrics import roc_auc_score as auc
s = d["score"].copy()
for k, (m, name) in enumerate(ORDER):
    i = np.where(cfg == m)[0]; v = (s[i] - s[i].mean()) / s[i].std()
    r = len(ORDER) - 1 - k
    for cls, col, mk, off in ((True, ROB, "o", .15), (False, FRAG, "^", -.15)):
        j = y[i] == cls
        b.scatter(v[j], r + off + RNG.uniform(-.07, .07, j.sum()), s=9, color=col, marker=mk, lw=.25, ec="white", zorder=3)
    b.text(3.5, r, f"{auc(y[i], s[i]):.2f}", ha="right", va="center", color=INK, fontsize=7)
b.text(3.5, len(ORDER) - .42, "AUC", ha="right", va="bottom", color=MUTED, fontsize=7)
b.set_yticks(range(len(ORDER))); b.set_yticklabels([n for _, n in ORDER][::-1]); b.tick_params(axis="y", labelcolor=INK)
b.set(xlim=(-3.4, 3.6), ylim=(-1.45, len(ORDER) - .35), xlabel="Out-of-fold $\\mathtt{SegProbe}$ score, deciles 1\u20139 (within-model $z$)"); b.grid(axis="y", visible=False)
b.scatter([], [], s=9, color=ROB, marker="o", label="Robust"); b.scatter([], [], s=9, color=FRAG, marker="^", label="Fragile")
b.legend(loc="lower left", ncol=2, handletextpad=.1, columnspacing=.8, borderaxespad=.1)
save(fig, "fig_seg")

# ---------------------------------------------------------------- trajectory
fig, (a, b) = plt.subplots(1, 2, figsize=(rs.WIDTH, 1.9), gridspec_kw={"wspace": .62})
two(a, np.arange(1, 9), d["Zs"], y); a.axhline(0, color="k", lw=.6, ls=":")
a.set(xlabel="Step $s$", ylabel=r"$\log\Vert\bar h_{s+1}-\bar h_s\Vert$ ($z$)", xticks=[1, 4, 8]); a.yaxis.label.set_size(9)
two(b, np.arange(1, 10), d["Zn"], y); b.axhline(0, color="k", lw=.6, ls=":")
b.set(xlabel="Segment $s$", ylabel=r"$\Vert\bar h_s\Vert\,/\,\mathrm{mean}_s\Vert\bar h_s\Vert$ ($z$)", xticks=[1, 5, 9]); b.yaxis.label.set_size(9)
for ax in (a, b): ax.set_ylim(-.65, .55)
fig.legend(*a.get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.06))
save(fig, "fig_traj")

# ---------------------------------------------------------------- attention sink, Skywork
ys, sink, caps, ntok = d["ys"], d["sink"], d["caps"], d["ntok"]
plen = d["plen"][d["sky"]]; late = sink[:, 5:9].mean(1)
fig, (a, b, c) = plt.subplots(1, 3, figsize=(5.5, 2.25), gridspec_kw={"wspace": .5})
a.axvspan(5.5, 9.5, color=NEUTRAL, alpha=.16, lw=0); two(a, np.arange(1, 11), sink, ys)
a.text(7.5, .075, "late window", ha="center", color=MUTED, fontsize=6.5)
a.set(xlabel="Decile of the chain of thought", ylabel="Attention share on BOS + template", xticks=[1, 3, 5, 7, 9], ylim=(.06, None))
a.legend(loc="upper right", borderaxespad=.1, handletextpad=.3)
for cls, col, mk in ((True, ROB, "o"), (False, FRAG, "^")):
    j = ys == cls
    b.scatter(ntok[j], late[j], s=10, color=col, marker=mk, lw=.25, ec="white", zorder=3)
    k = (ys == cls) & caps
    c.scatter(plen[k], late[k], s=10, color=col, marker=mk, lw=.25, ec="white", zorder=3)
b.set(xlabel="Trace length (tokens)", ylabel="BOS + template share, deciles 6\u20139", xticks=[1000, 2000, 3000, 4000], xticklabels=["1k", "2k", "3k", "4k"])
b.text(.05, .05, "$r$ = %.2f" % np.corrcoef(late, ntok)[0, 1], transform=b.transAxes, color=INK, fontsize=7)
c.set_xscale("log"); c.set(xlabel="Problem length (characters), capped traces", ylabel="BOS + template share, deciles 6\u20139")
c.set_xticks([100, 300, 1000]); c.set_xticklabels(["100", "300", "1k"]); c.minorticks_off()
c.text(.95, .9, "$r$ = %.2f" % np.corrcoef(np.log(plen[caps]), late[caps])[0, 1], transform=c.transAxes, ha="right", color=INK, fontsize=7)
save(fig, "fig_attn")

# ---------------------------------------------------------------- attention by decile, single panel for the report
fig, a = plt.subplots(figsize=(3.1, 2.15))
a.axvspan(5.5, 9.5, color=NEUTRAL, alpha=.16, lw=0); two(a, np.arange(1, 11), sink, ys)
a.text(7.5, .072, "late window", ha="center", color=MUTED, fontsize=6.5)
a.set(xlabel="Decile of the chain of thought", ylabel="Attention share on BOS + template", xticks=range(1, 11), ylim=(.06, None))
a.legend(loc="upper right", borderaxespad=.1, handletextpad=.3)
save(fig, "fig_attn_decile")
