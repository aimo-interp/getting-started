"""Confusion matrices and failure / success modes per method. Positive class = fragile (what we try to flag).
Majority over 20 repeats at .5; 'matched' flags, per configuration, as many cases as are truly fragile, by mean OOF p.

    python3 analysis/report/errors_analysis.py > analysis/report/errors_analysis.txt
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
HERE = Path(__file__).parent
d = pd.read_csv(HERE / "errors_cases.csv", index_col=0)
M = ["seg", "traj", "attn", "whitebox", "tx", "stack"]
d["frag"] = ~d.robust
for k in M:
    d[f"f5_{k}"] = d[f"flag_{k}"] >= .5
    d[f"fm_{k}"] = False
    for m, g in d.groupby("model"):
        n = int(g.frag.sum()); d.loc[g[f"p_{k}"].nsmallest(n).index, f"fm_{k}"] = True


def cm(s, f):
    return dict(TP=int((f & s.frag).sum()), FN=int((~f & s.frag).sum()), FP=int((f & ~s.frag).sum()), TN=int((~f & ~s.frag).sum()))


def show(s, tag):
    print(f"\n== confusion, {tag}: n={len(s)} fragile={s.frag.sum()}")
    print(f"{'':10s}{'at .5: TP  FN  FP   TN  recall prec acc':42s} matched: TP  FN  FP   TN  recall")
    for k in M:
        a, b = cm(s, s[f"f5_{k}"]), cm(s, s[f"fm_{k}"]); n = len(s)
        pr = a["TP"] / max(a["TP"] + a["FP"], 1)
        print(f"{k:10s}{a['TP']:9d}{a['FN']:4d}{a['FP']:4d}{a['TN']:5d}{a['TP'] / (a['TP'] + a['FN']):8.2f}{pr:5.2f}{(a['TP'] + a['TN']) / n:5.2f}"
              f"{'':12s}{b['TP']:4d}{b['FN']:4d}{b['FP']:4d}{b['TN']:5d}{b['TP'] / (b['TP'] + b['FN']):8.2f}")


show(d, "all 118"); show(d[d.attn_data], "cases with attention data")
print("\n== by configuration, TP/FN/FP/TN at .5")
print(f"{'':20s}" + "".join(f"{k:>14s}" for k in M))
for m, g in d.groupby("model"):
    print(f"{m[:18]:20s}" + "".join("%14s" % ("{TP}/{FN}/{FP}/{TN}".format(**cm(g, g[f'f5_{k}']))) for k in M))

fr = d[d.frag]
print("\n== who catches the 35 fragile cases at .5 (seg, traj, attn; attn on all cases, intercept without data)")
print(fr.groupby(["f5_seg", "f5_traj", "f5_attn"]).size().to_string())
print("at matched flag rate:"); print(fr.groupby(["fm_seg", "fm_traj", "fm_attn"]).size().to_string())
print("error overlap: pairs both wrong / either wrong, fragile at .5")
for a, b in (("seg", "traj"), ("seg", "attn"), ("traj", "attn"), ("seg", "tx"), ("traj", "tx")):
    x, y = ~fr[f"f5_{a}"], ~fr[f"f5_{b}"]; print(f"  {a}-{b}: both missed {(x & y).sum()}, one missed {(x ^ y).sum()}")
print("corr of OOF p within model:")
print(d[[f"p_{k}" for k in M]].sub(d.groupby("model")[[f"p_{k}" for k in M]].transform("mean")).corr().round(2).to_string())

print("\n== context by outcome (stack at .5)")


def table(key, label):
    s = d.copy(); out = np.where(s.frag, np.where(s[key], "TP", "FN"), np.where(s[key], "FP", "TN")); s["out"] = out
    print(f"-- {label}")
    print(s.groupby("out").agg(n=("frag", "size"), solve=("solve", "mean"), solved_all=("solve", lambda v: (v >= 1).mean()), solved_none=("solve", lambda v: (v <= 0).mean()),
                               greedy_right=("greedy_correct", "mean"), boxed=("greedy_boxed", "mean"), capped=("capped", "mean"), tokens=("tokens", "median"), chars=("chars", "median"),
                               drop=("max_drop", "median"), agree=("agree", "median"), organiser=("source", lambda v: (v == "train-main-v2").mean())).round(2).to_string())


for k in ("seg", "traj", "attn", "stack", "tx"):
    table(f"f5_{k}", f"{k} at .5"); table(f"fm_{k}", f"{k} matched")

print("\n== recall (matched) and flag rate by context")
FE = {"solve band": pd.cut(d.solve, [-.01, 0, .5, .99, 1.0], labels=["0", "<=.5", ".5-.99", "1"]), "capped": d.capped, "greedy right": d.greedy_correct,
      "topic": d.topic, "source": d.source, "drop band": pd.cut(d.max_drop, [-2, 0, .2, .3, .5, 1.01], labels=["<=0", "0-.2", ".2-.3", ".3-.5", ">.5"])}
for nm, b in FE.items():
    print(f"-- {nm}")
    rows = {}
    for v, g in d.groupby(b, observed=True):
        f = g[g.frag]; r = g[g.robust]
        rows[v] = {"n": len(g), "frag": len(f), **{f"{k} recall": round(f[f"fm_{k}"].mean(), 2) if len(f) else np.nan for k in ("seg", "traj", "attn", "stack")},
                   **{f"{k} FA": round(r[f"fm_{k}"].mean(), 2) if len(r) else np.nan for k in ("seg", "traj", "attn", "stack")}}
    print(pd.DataFrame(rows).T.to_string())

print("\n== fragile: caught vs missed tests (matched), Fisher / MWU")
for k in ("seg", "traj", "attn", "stack"):
    a = fr[fr[f"fm_{k}"]]; b = fr[~fr[f"fm_{k}"]]
    t = [[(a.solve >= 1).sum(), (a.solve < 1).sum()], [(b.solve >= 1).sum(), (b.solve < 1).sum()]]
    cp = [[a.capped.sum(), (~a.capped).sum()], [b.capped.sum(), (~b.capped).sum()]]
    print(f"{k:6s} caught {len(a)} missed {len(b)} | solved-all caught {t[0][0]}/{len(a)} missed {t[1][0]}/{len(b)} Fisher p {fisher_exact(t)[1]:.3f} | "
          f"solve med {a.solve.median():.2f}/{b.solve.median():.2f} MWU p {mannwhitneyu(a.solve.dropna(), b.solve.dropna()).pvalue:.3f} | "
          f"drop med {a.max_drop.median():.2f}/{b.max_drop.median():.2f} p {mannwhitneyu(a.max_drop, b.max_drop).pvalue:.3f} | chars p {mannwhitneyu(a.chars, b.chars).pvalue:.3f}")

cols = ["model", "source", "topic", "solve", "greedy_correct", "capped", "max_drop", "p_seg", "p_traj", "p_attn", "p_tx", "p_stack"]
pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 200)
print("\n== fragile cases missed by seg, traj, attn all at matched rate"); s = fr[~fr.fm_seg & ~fr.fm_traj & ~fr.fm_attn]; print(s[cols].sort_values("solve").round(2).to_string())
print("\n== fragile cases caught by at least two of seg, traj, attn"); s = fr[(fr.fm_seg.astype(int) + fr.fm_traj + fr.fm_attn) >= 2]; print(s[cols].sort_values("solve").round(2).to_string())
print("\n== robust flagged by at least two (false alarms)"); r = d[d.robust]; s = r[(r.fm_seg.astype(int) + r.fm_traj + r.fm_attn) >= 2]; print(s[cols].sort_values("solve").round(2).to_string())
