"""Full stack on all 118 cases: confusion matrix and outcomes by trace group. Reads errors_stack_cases.csv from errors_stack.py.

    python3 analysis/report/errors_figure.py <fig_dir>
"""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as LSC
import numpy as np, pandas as pd
import report_figstyle as rs
HERE = Path(__file__).parent; OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "figures"
d = pd.read_csv(HERE / "errors_stack_cases.csv", index_col=0)
rs.apply()
fig, (a, b) = plt.subplots(2, 1, figsize=(rs.WIDTH, 2.6), gridspec_kw={"height_ratios": [1, 1.55], "hspace": .95})
n = d.out.value_counts(); M = np.array([[n["TP"], n["FN"]], [n["FP"], n["TN"]]]); sh = M / M.sum(1, keepdims=True)
a.imshow(sh, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(2):
    for j in range(2):
        a.text(j, i, M[i, j], ha="center", va="center", color="white" if sh[i, j] > .6 else "black")
a.set_xticks([0, 1]); a.set_xticklabels(["Fragile", "Robust"]); a.set_yticks([0, 1]); a.set_yticklabels(["Fragile", "Robust"])
a.set(xlabel="Predicted label", ylabel="True label")
G = ["solved sometimes", "never finished at 16k", "solved every time", "finished within 4k"]
L = ["Capped, solve rate 0.125\u20130.875", "Capped, solve rate 0", "Capped, solve rate 1", "Finished before cap"]
y = np.arange(4)[::-1]
for cls, col, off, lab in ((True, rs.FRAG, .19, "True fragile"), (False, rs.ROB, -.19, "True robust")):
    for g, yy in zip(G, y):
        s = d[(d["mode"] == g) & (d.frag == cls)]; c = int(s.out.isin(["TP", "FP"]).sum()); v = c / len(s)
        b.barh(yy + off, v, height=.36, color=col, label=lab if g == G[0] else None); b.text(v + .02, yy + off, f"{c}/{len(s)}", va="center", fontsize=8)
b.set_yticks(y); b.set_yticklabels(L); b.set_xlim(0, 1.18); b.set_xticks([0, .5, 1]); b.set_xlabel("Fraction predicted fragile")
b.legend(loc="lower right", fontsize=8)
rs.save(fig, OUT, "fig_errors")
