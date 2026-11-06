"""Failure / success modes. Conventions match results_tables.py: a case is wrong if wrong in >= half of the repeats.
Positive = fragile. attn is scored only where attention data exist; seg only where its windows are not gated off
(elsewhere the OOF score is fold noise). 'matched' = per configuration, flag as many of the scored cases as are fragile.

    python3 analysis/report/errors_modes.py > analysis/report/errors_modes.txt
"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, wilcoxon
HERE = Path(__file__).parent; ROOT = HERE.parents[1]; DATA = ROOT / "data/cvbench"
pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 300)
d = pd.read_csv(HERE / "errors_cases.csv", index_col=0); d["frag"] = ~d.robust
M = ["seg", "traj", "attn", "whitebox", "tx", "stack"]
scope = {k: pd.Series(True, index=d.index) for k in M}; scope["attn"] = d.attn_data; scope["seg"] = ~d.seg_gated_off.astype(bool)
print("scored cases:", {k: int(v.sum()) for k, v in scope.items()}, "| seg gated off:", d[d.seg_gated_off.astype(bool)].groupby("model").size().to_dict())
for k in M:
    d[f"c5_{k}"] = np.where(d.frag, d[f"flag_{k}"] > .5, d[f"flag_{k}"] >= .5)     # called fragile, ties count as wrong
    d[f"cm_{k}"] = False
    for m, g in d[scope[k]].groupby("model"):
        d.loc[g[f"p_{k}"].nsmallest(int(g.frag.sum())).index, f"cm_{k}"] = True


def row(s, col):
    c = s[col]; tp = (c & s.frag).sum(); fn = (~c & s.frag).sum(); fp = (c & ~s.frag).sum(); tn = (~c & ~s.frag).sum()
    return tp, fn, fp, tn


def table(mask, tag):
    print(f"\n== {tag} (fragile positive)\n{'':10s}{'n':>4s}{'frag':>5s} | at .5 {'TP':>4s}{'FN':>4s}{'FP':>4s}{'TN':>4s}{'recall':>7s}{'prec':>6s} | matched {'TP':>3s}{'FN':>4s}{'recall':>7s}")
    for k in M:
        s = d[mask & scope[k]]; a = row(s, f"c5_{k}"); b = row(s, f"cm_{k}")
        print(f"{k:10s}{len(s):4d}{s.frag.sum():5d} |       {a[0]:4d}{a[1]:4d}{a[2]:4d}{a[3]:4d}{a[0] / max(a[0] + a[1], 1):7.2f}{a[0] / max(a[0] + a[2], 1):6.2f} |        {b[0]:3d}{b[1]:4d}{b[0] / max(b[0] + b[1], 1):7.2f}")


ALL = pd.Series(True, index=d.index)
table(ALL, "all cases"); table(d.capped, "capped traces"); table(~d.capped, "finished traces")
print("\n== by configuration at .5, TP/FN/FP/TN"); print(f"{'':18s}" + "".join(f"{k:>12s}" for k in M))
for m, g in d.groupby("model"):
    print(f"{m[:16]:18s}" + "".join(("%12s" % "/".join(map(str, row(g[scope[k][g.index]], f"c5_{k}")))) if scope[k][g.index].any() else f"{'-':>12s}" for k in M))

print("\n== problem or model? problems labelled in >=2 configurations")
d["mz"] = {}.get(0)
for k in ("seg", "traj", "attn", "stack", "tx"):
    d[f"z_{k}"] = np.nan
    for m, g in d[scope[k]].groupby("model"):
        d.loc[g.index, f"z_{k}"] = (g[f"p_{k}"] - g[f"p_{k}"].mean()) / (g[f"p_{k}"].std() + 1e-9)
multi = d.groupby("group").filter(lambda g: len(g) >= 2)
print("cases", len(multi), "problems", multi.group.nunique(), "| labels agree in", (multi.groupby("group").frag.nunique() == 1).sum(), "| mixed labels", (multi.groupby("group").frag.nunique() == 2).sum())
for k in ("seg", "traj", "attn", "stack", "tx"):
    s = d[scope[k]].dropna(subset=[f"z_{k}"]); s = s.groupby("group").filter(lambda g: len(g) >= 2); z = s[f"z_{k}"]
    between = s.groupby("group")[f"z_{k}"].transform("mean"); icc = between.var() / z.var()
    wins = ties = tot = 0; diffs = []
    for g_, g in s.groupby("group"):
        if g.frag.nunique() == 2:
            r = g[g.robust][f"z_{k}"].values; f = g[g.frag][f"z_{k}"].values; D = r[:, None] - f[None]
            wins += (D > 0).sum(); ties += (D == 0).sum(); tot += D.size; diffs.append(D.mean())
    pb = (d[scope[k]].groupby("group")[f"z_{k}"].mean()); yb = d[scope[k]].groupby("group").frag.mean()
    w = wilcoxon(diffs).pvalue if len(diffs) > 5 else np.nan
    print(f"{k:6s} share of score variance between problems {icc:.2f} | same problem, mixed labels: robust scores above fragile in {wins}/{tot} pairs = {wins / tot:.2f}, over {len(diffs)} problems, Wilcoxon p {w:.3f}")
print("\nmost repeated problems: label and call per configuration (stack at .5, R/F label; lowercase = called fragile)")
for g_, g in sorted(d.groupby("group"), key=lambda x: -len(x[1]))[:10]:
    s = " ".join(f"{r.model[:6]}:{'R' if r.robust else 'F'}{'*' if r.c5_stack else ''}" for r in g.itertuples())
    print(f"  {g_} n={len(g)} {g.problem.iloc[0][:60]!r} | {s}")

print("\n== four modes on capped traces, seg+traj; fragile caught by both / missed by both; robust clear / flagged by both (matched)")
s = d[d.capped & scope["seg"]].copy(); both = s.cm_seg & s.cm_traj; neither = ~s.cm_seg & ~s.cm_traj
s["mode"] = np.where(s.frag & both, "A caught", np.where(s.frag & neither, "B missed", np.where(~s.frag & both, "D false alarm", np.where(~s.frag & neither, "C clear", "mixed"))))
print(s["mode"].value_counts().sort_index().to_string())


G = {}
for f in (DATA / "greedy").glob("*.jsonl"):
    for l in open(f):
        r = json.loads(l); G[(r["model_id"], r["group"])] = r.get("generated_text", "")
c_ = {json.loads(l)["case_id"]: json.loads(l) for f in ("cases.jsonl", "cases_oss.jsonl") for l in open(DATA / f)}


def desc(t):
    w = t.lower().split(); n = max(len(w), 1); third = len(t) // 3; tail = t[2 * third:]
    gr = lambda x: [tuple(x[i:i + 6]) for i in range(max(len(x) - 5, 1))]
    tw = tail.lower().split(); gs = gr(tw); rep = 1 - len(set(gs)) / max(len(gs), 1)
    sent = [x.strip() for x in re.split(r"[\n.]", tail) if len(x.strip()) > 15]; dup = 1 - len(set(sent)) / max(len(sent), 1)
    return dict(wait=1000 * len(re.findall(r"\bwait\b", t.lower())) / n, alt=1000 * len(re.findall(r"\balternatively\b|\banother (?:way|approach)\b", t.lower())) / n,
                unsure=1000 * len(re.findall(r"\bhmm\b|not sure|maybe|mistake|wrong\b|actually", t.lower())) / n,
                verify=1000 * len(re.findall(r"\bcheck\b|verify|confirm", t.lower())) / n, rep_tail=rep, dup_tail=dup, math=len(re.findall(r"[=\\^]", t)) / max(len(t), 1))


D = pd.DataFrame({i: desc(G.get((c_[i]["model_id"], c_[i]["group"]), "")) for i in d.index}).T
s = s.join(D); cols = list(D.columns)
print(s.groupby("mode")[cols + ["solve", "max_drop", "chars"]].median().round(3).to_string())
print("within model z, mean:"); Z = s[cols].groupby(s.model).transform(lambda v: (v - v.mean()) / (v.std() + 1e-9)); print(Z.groupby(s["mode"]).mean().round(2).to_string())
print("robust vs fragile among capped, z, MWU p:", {c: round(mannwhitneyu(Z[c][s.robust].dropna(), Z[c][s.frag].dropna()).pvalue, 3) for c in cols})
print("A vs B, z, MWU p:", {c: round(mannwhitneyu(Z[c][s["mode"] == "A caught"].dropna(), Z[c][s["mode"] == "B missed"].dropna()).pvalue, 3) for c in cols})
keep = ["model", "source", "topic", "solve", "max_drop", "tokens", "p_seg", "p_traj", "p_attn", "p_tx", "p_stack", "rep_tail"]
for m in ("A caught", "B missed", "D false alarm"):
    print(f"\n-- {m}"); print(s[s["mode"] == m][keep].sort_values("p_stack").round(2).to_string())
s.to_csv(HERE / "errors_modes_capped.csv")
d.to_csv(HERE / "errors_cases_scored.csv")
print("\n== solve 0 cases: finished any of 8 at 16k?")
oc = pd.read_csv(HERE / "organiser_competence_cases.csv", index_col=0)
z = d[d.solve == 0].join(oc[["finished"]]); print(z.groupby(["model", "source", "robust"]).agg(n=("frag", "size"), finished_med=("finished", "median"), capped=("capped", "mean")).to_string())
