"""Error analysis of the seg probe / stack from oof_cases.csv (+ labelgen per-perturbation accuracies)."""
import json, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact
D = "analysis/seg_errors/"
o = pd.read_csv(D + "oof_cases.csv", index_col=0)
o["m"] = o.model_id.str.split("/").str[-1].str.replace("-120b", "").str[:16]
o["frag"] = ~o.is_robust
o["wrong_seg"], o["wrong_stack"], o["wrong_tt"] = o.err_seg >= .5, o.err_stack >= .5, o.err_tt >= .5
o["gated_off"] = (o.gate_w05 < .5) & (o.gate_w59 < .5)
o["hit_cap"] = o.greedy_tokens >= 4096
L = pd.DataFrame([json.loads(l) for l in open("data/labelgen/labels.jsonl")])
key = o.model_id + "||" + o.problem
L = L.set_index(L.model_id + "||" + L.problem)
L = L[~L.index.duplicated()].reindex(key.values); L.index = o.index
TYPES = ["distract", "domain", "expert", "rename", "rephrase", "typos"]
for t in TYPES:
    o[f"drop_{t}"] = (L.base_accuracy - L.type_accuracy.map(lambda d: d.get(t, np.nan) if isinstance(d, dict) else np.nan)).reindex(o.index)
o["base_acc"] = L.base_accuracy.reindex(o.index)
pd.set_option("display.width", 220, "display.max_columns", 40)

print("=== 1. Per model: cases, fragile, how often each method is wrong (majority over 20 repeats)")
print(o.groupby("m").agg(n=("frag", "size"), frag=("frag", "sum"), seg_gated_off=("gated_off", "sum"),
                         seg_wrong=("wrong_seg", "sum"), stack_wrong=("wrong_stack", "sum"),
                         tt_wrong=("wrong_tt", "sum")).to_string())

print("\n=== 2. Error type: seg misses fragile (FN) vs flags robust (FP)")
for k in ["seg", "stack"]:
    w = o[f"wrong_{k}"]
    print(f"{k}: wrong={w.sum()}  missed fragile={(w & o.frag).sum()}/{o.frag.sum()}  false alarms={(w & ~o.frag).sum()}/{(~o.frag).sum()}")
print("seg wrong on fragile cases, split by gate:",
      o[o.frag].groupby("gated_off").wrong_seg.agg(["sum", "size"]).to_dict())

print("\n=== 3. Missed-fragile cases (stack), sorted by model")
cols = ["m", "source", "max_drop", "base_acc", "p_seg", "p_tt", "p_stack", "gated_off", "dim_w05", "dim_w59", "agree", "greedy_tokens"]
print(o[o.frag & o.wrong_stack].sort_values(["m", "p_stack"])[cols].round(2).to_string())
print("\n=== 3b. False alarms (stack)")
print(o[~o.frag & o.wrong_stack].sort_values(["m", "p_stack"])[cols].round(2).to_string())

print("\n=== 4. What distinguishes caught vs missed fragile cases (stack)? MWU p; Fisher for flags")
f = o[o.frag]
for c in ["max_drop", "dim_w05", "dim_w59", "agree", "anchor", "tx_length", "greedy_logprob", "diff_last", "problem_chars"]:
    a, b = f[f.wrong_stack][c].dropna(), f[~f.wrong_stack][c].dropna()
    if len(a) > 1 and len(b) > 1:
        print(f"  {c:15s} missed med={a.median():7.2f} (n={len(a)})  caught med={b.median():7.2f} (n={len(b)})  p={mannwhitneyu(a, b).pvalue:.3f}")
for flag in ["hit_cap", "gated_off"]:
    t = pd.crosstab(f[flag], f.wrong_stack)
    if t.shape == (2, 2):
        print(f"  {flag}: {t.to_dict()} Fisher p={fisher_exact(t)[1]:.3f}")
print("  by source:", f.groupby("source").wrong_stack.agg(["sum", "size"]).to_dict())
print("  max_drop bucket:", f.groupby(pd.cut(f.max_drop, [0, .3, .5, 1.01])).wrong_stack.agg(["sum", "size"]).to_dict())

print("\n=== 5. Labelgen fragile cases: which perturbation types broke it, and did we catch it?")
lg = o[o.frag & o.base_acc.notna()].copy()
lg["culprits"] = lg.apply(lambda r: "+".join(t for t in TYPES if r[f"drop_{t}"] >= .25), axis=1)
print(lg[["m", "base_acc", "max_drop", "culprits", "p_seg", "p_tt", "p_stack", "wrong_seg", "wrong_stack"]].sort_values("p_stack").round(2).to_string())
print("\n  per type: fragile cases where that type dropped accuracy >= .25 -> caught by seg / stack")
for t in TYPES:
    s = lg[lg[f"drop_{t}"] >= .25]
    print(f"  {t:9s} n={len(s):2d}  seg caught {(~s.wrong_seg).sum():2d}  stack caught {(~s.wrong_stack).sum():2d}  mean p_stack={s.p_stack.mean():.2f}")
lg["n_culprits"] = lg.culprits.str.count(r"\+") + (lg.culprits != "")
print("\n  breadth: n culprit types vs mean p_stack (lower = more fragile-looking):")
print(lg.groupby("n_culprits").agg(n=("p_stack", "size"), p_stack=("p_stack", "mean"), caught=("wrong_stack", lambda w: (~w).sum())).round(2).to_string())
r = o[o.base_acc.notna()]
print("\n  labelgen robust vs fragile: n", r.frag.value_counts().to_dict(),
      " base_acc<1 among robust:", ((r.base_acc < 1) & ~r.frag).sum(), " among fragile:", ((r.base_acc < 1) & r.frag).sum())
o.to_csv(D + "cases_annotated.csv")
