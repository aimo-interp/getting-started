"""Thought-anchor curves, gpt-oss-120b: P(correct | first k sentences of a solved trace on the original problem), 10 resamples per point.

    python3 analysis/report/fig_anchors.py
"""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import report_figstyle as rs
HERE = Path(__file__).parent; ROOT = HERE.parents[1]; OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "figures"
T = pd.read_csv(ROOT / "autoresearch/loop-261102-0533/anchors_fickle.csv")
ROB, FRAG, NEUTRAL = rs.ROB, rs.FRAG, rs.GREY
rs.apply()
rng = np.random.default_rng(0)
C = {}
for r in T.itertuples():
    d = pd.read_json(ROOT / "data/thought_anchors_crusoe" / r.case / "continuations.jsonl", lines=True)
    d = d[d.curve == "X"].groupby("k").correct.agg(p_correct="mean", n="size").reset_index()
    d["fraction"] = d.k / d.k.max(); C[r.case] = d
    assert len(d) == r.X_npts and abs(d.p_correct.iloc[0] - r.X_p0) < 1e-9, r.case
print("resamples per point", sorted({int(x) for c in C.values() for x in c.n}))
print("cases", len(T), "robust", T.robust.sum(), "fragile", (~T.robust).sum(), "points", T.X_npts.min(), T.X_npts.max())
print("efforts", T.groupby(["effort", "robust"]).size().to_dict())
EFFORTS = ("low", "medium", "high")
fig, axs = plt.subplots(1, 3, figsize=(5.5, 2.2), sharey=True, gridspec_kw={"wspace": .08})
low = (~T.robust) & (T.X_p0 <= .8)
print("fragile p0<=.8", low.sum(), "p0>.8", ((~T.robust) & ~low).sum(), "robust p0<=.8", (T.robust & (T.X_p0 <= .8)).sum())
for ax, e in zip(axs, EFFORTS):
    E = T[T.effort == e]; f = E[~E.robust]
    for mask, col, lw, z in ((E.robust, ROB, .9, 2), ((~E.robust) & (E.X_p0 > .8), NEUTRAL, .9, 3), ((~E.robust) & (E.X_p0 <= .8), FRAG, 1, 4)):
        for r in E[mask].itertuples():
            c = C[r.case]; ax.plot(c.fraction, c.p_correct + rng.uniform(-.008, .008), color=col, lw=lw, zorder=z)
    ax.set_title(f"{e.capitalize()} effort"); ax.text(.97, .04, f"{E.robust.sum()} robust, {len(f)} fragile", transform=ax.transAxes, ha="right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_xticks([0, .5, 1]); ax.set_xticklabels(["0", "0.5", "1"]); ax.set_xlabel("Fraction of trace kept")
axs[0].set_ylabel("Fraction of continuations correct", fontsize=9); axs[0].set_ylim(-.04, 1.06); axs[0].set_yticks([0, .25, .5, .75, 1]); axs[0].set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
H = [plt.Line2D([], [], color=ROB, lw=1, label="Robust"), plt.Line2D([], [], color=FRAG, lw=1, label="Fragile, solve rate $\\leq$ 0.8"),
     plt.Line2D([], [], color=NEUTRAL, lw=1, label="Fragile, solve rate $\\geq$ 0.9")]
fig.legend(handles=H, loc="upper center", bbox_to_anchor=(.5, 1.13), ncol=3)
rs.save(fig, OUT, "fig_anchors")
