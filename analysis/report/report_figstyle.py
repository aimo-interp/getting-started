"""Shared style for report figures: stock matplotlib look in Times New Roman.

Figures are drawn WIDTH inches wide and included at .5\\textwidth (2.75 in), so 10 pt text prints at about 8.3 pt.
Set REPORT_FONT_DIR to a folder holding Times New Roman if it is not installed; Liberation Serif has the same metrics.
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROB, FRAG, GREY = "tab:blue", "tab:orange", "0.55"
WIDTH = 3.3


def apply():
    for f in Path(os.environ.get("REPORT_FONT_DIR", "/nonexistent")).rglob("*.[tT][tT][fF]"):
        font_manager.fontManager.addfont(str(f))
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"], "mathtext.fontset": "stix",
                         "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 9, "pdf.fonttype": 42, "lines.markersize": 4, "savefig.dpi": 400})


def save(fig, out, name):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=.02)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=.02, facecolor="white")
