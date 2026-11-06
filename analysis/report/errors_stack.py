"""Every error of the full stack (118 cases, 20 x 5 grouped CV, wrong in >= half of repeats, matches results_tables.py).

    python3 analysis/report/errors_stack.py > analysis/report/errors_stack.txt
"""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; HERE = Path(__file__).parent
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
d = pd.read_csv(HERE / "errors_cases_scored.csv", index_col=0)
Z = np.load(HERE / "errors_oof.npz", allow_pickle=True); P = pd.DataFrame(Z["stack"].T, index=Z["case_id"].astype(str))
d["err_rate"] = (((P >= .5).values == d.loc[P.index, "robust"].values[:, None])).__invert__().mean(1)
d["wrong"] = d.err_rate >= .5
d["out"] = np.where(d.frag, np.where(d.wrong, "FN", "TP"), np.where(d.wrong, "FP", "TN"))
d["finished"] = ~d.capped
d["mode"] = np.select([d.finished, d.solve >= 1, d.solve <= 0], ["finished within 4k", "solved every time", "never finished at 16k"], "solved sometimes")
print("== outcomes", d.out.value_counts().to_dict(), "mean err rate on errors %.2f" % d[d.wrong].err_rate.mean())
print("always wrong", int((d.err_rate == 1).sum()), "| wrong in 50-80%", int((d.wrong & (d.err_rate <= .8)).sum()), "| right cases wrong in 20-50%", int((~d.wrong & (d.err_rate >= .2)).sum()))
print("\n== by configuration"); print(pd.crosstab(d.model, d.out).reindex(columns=["TP", "FN", "FP", "TN"], fill_value=0).to_string())
print("\n== by mode"); print(pd.crosstab(d["mode"], d.out).reindex(columns=["TP", "FN", "FP", "TN"], fill_value=0).to_string())
print("\n== by source"); print(pd.crosstab(d.source, d.out).reindex(columns=["TP", "FN", "FP", "TN"], fill_value=0).to_string())
print("\n== by topic"); print(pd.crosstab(d.topic, d.out).reindex(columns=["TP", "FN", "FP", "TN"], fill_value=0).to_string())
print("\n== by drop"); print(pd.crosstab(pd.cut(d.max_drop[d.frag], [0, .3, .5, 1.01]), d.out[d.frag]).to_string())
L = {}
for l in open(ROOT / "data/labelgen/labels.jsonl"):
    r = json.loads(l); L[(r["model"] if "model" in r else r["model_id"], r["problem"])] = r
c_ = {json.loads(l)["case_id"]: json.loads(l) for f in ("cases.jsonl", "cases_oss.jsonl") for l in open(ROOT / "data/cvbench" / f)}
G = {}
for f in glob.glob(str(ROOT / "data/cvbench/greedy/*.jsonl")):
    for l in open(f):
        r = json.loads(l); G[(r["model_id"], r["group"])] = r.get("generated_text", "")
e = d[d.wrong].copy()
e["blocks_right"] = ["".join(x for k, x in (("seg", "S"), ("traj", "T"), ("attn", "A"), ("tx", "X")) if (r[f"c5_{k}"] == r.frag)) for _, r in e.iterrows()]
e["variants"] = [(lambda b: (f"{sum(b['base_accuracy'] - v >= .25 for v in b['type_accuracy'].values())}/{len(b['type_accuracy'])}" if b else ""))(L.get((c_[i]["model_id"], c_[i]["problem"]))) for i in e.index]
cols = ["out", "model", "mode", "source", "topic", "solve", "tokens", "max_drop", "agree", "err_rate", "p_stack", "blocks_right", "variants"]
print("\n== all 19 errors"); print(e.sort_values(["out", "mode", "model"])[cols].round(2).to_string())
print("\n== errors with no block individually right:", int((e.blocks_right == "").sum()), "of", len(e), "| FN with tx right", int(e[e.out == "FN"].blocks_right.str.contains("X").sum()))
print("repeat problems among errors:", e.group.value_counts().head(5).to_dict())
for i, r in e.sort_values(["out", "mode"]).iterrows():
    c = c_[i]; t = G[(c["model_id"], c["group"])]
    print(f"\n[{r.out}] {i} | {c['problem'][:150]!r}\n   END: {t[-260:]!r}")
d.to_csv(HERE / "errors_stack_cases.csv")
