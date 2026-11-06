"""Do generated labels survive? (a) finished samples only (truncated = missing, not wrong); (b) pooled sep2 labels (10 more samples per
prompt, same 8k cap, truncated = wrong); (c) forced answers at the cap, Skywork. Plus trace length on the original as a predictor,
and logit contributions of each block on the 19 errors.

    python3 analysis/report/errors_why_relabel.py > analysis/report/errors_why_relabel.txt
"""
import json, glob
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as auc
ROOT = Path(__file__).resolve().parents[2]; HERE = Path(__file__).parent; LG = ROOT / "data/labelgen"; SEP = ROOT / "data/cvbench/sep2/samples"
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 300)
d = pd.read_csv(HERE / "errors_stack_cases.csv", index_col=0); c_ = {json.loads(l)["case_id"]: json.loads(l) for f in ("cases.jsonl", "cases_oss.jsonl") for l in open(ROOT / "data/cvbench" / f)}
lab = {}; 
for l in open(LG / "labels.jsonl"):
    r = json.loads(l); lab[(r["model_id"], r["problem"])] = r
pool = {}
for l in open(LG / "labels_pooled_sep2.jsonl"):
    r = json.loads(l); pool[(r["model_id"], r["problem"])] = r
S = defaultdict(list)
for f in glob.glob(str(LG / "samples/*__default.jsonl")):
    for l in open(f):
        r = json.loads(l); S[(r["model_id"], r["problem_id"])].append(r)
PR = {json.loads(l)["problem_id"]: json.loads(l) for l in open(LG / "problems.jsonl")}
F = defaultdict(list)
for l in open(SEP / "Skywork_Skywork-OR1-Math-7B__default.forced.jsonl"):
    r = json.loads(l); F[r["problem_id"]].append(r)


def rule(x):
    return "robust" if x <= .10 else ("fragile" if x >= .25 else "unlabelled")


rows = []
for i, r in d[d.source == "labelgen"].iterrows():
    c = c_[i]; lb = lab[(c["model_id"], c["problem"])]; p = lb["problem_id"]; acc = defaultdict(list); ntok = []
    for x in S[(c["model_id"], p)]:
        for a, ok, t in zip(x["answers"], x["correct"], x["n_tokens"]):
            if t < 8192 and a is not None: acc[x["type"]].append(ok)
            if x["type"] == "original": ntok.append(t)
    base = np.mean(acc["original"]) if acc["original"] else np.nan
    dr = [base - np.mean(v) for t, v in acc.items() if t != "original"]
    fin_drop = max(dr) if dr else np.nan
    pl = pool.get((c["model_id"], c["problem"])); pr = None if pl is None else {True: "robust", False: "fragile", None: "unlabelled"}[pl["is_robust"]]
    forced = np.nan
    if c["model_id"].startswith("Skywork") and p in F:
        g = PR[p]["answer"]; A = defaultdict(list)
        for x in F[p]:
            for a, fa, t in zip(x["answers"], x["forced"], x["n_tokens"]):
                v = a if a is not None else fa; A[x["type"]].append(v == g)
        if A["original"]:
            forced = max([np.mean(A["original"]) - np.mean(v) for t, v in A.items() if t != "original"] or [np.nan])
    rows.append(dict(case=i, out=r["out"], label="robust" if r.robust else "fragile", drop=r.max_drop, finished_drop=fin_drop, finished_only=rule(fin_drop) if fin_drop == fin_drop else "no finished", pooled=pr or "absent",
                     forced_drop=forced, forced=rule(forced) if forced == forced else "absent", orig_tok=int(np.median(ntok)), orig_at_cap=float(np.mean(np.array(ntok) >= 8192))))
T = pd.DataFrame(rows).set_index("case")
for k in ("finished_only", "pooled", "forced"):
    print(f"\n== current label vs {k}"); print(pd.crosstab(T.label, T[k]).to_string())
print("\n== by stack outcome, generated cases"); print(pd.crosstab([T.out], T.finished_only).to_string())
print(T[T.label == "fragile"].sort_values(["out", "drop"]).drop(columns=["label"]).round(2).to_string())
print("\nrobust cases changing under finished-only or pooled:"); print(T[(T.label == "robust") & ((T.finished_only == "fragile") | (T.pooled == "fragile") | (T.forced == "fragile"))].round(2).to_string())
g = T[T.label.isin(["robust", "fragile"])]
print("\noriginal median tokens AUC for fragile, generated, %.3f; share of originals at cap AUC %.3f" % (auc(g.label == "fragile", g.orig_tok), auc(g.label == "fragile", g.orig_at_cap)))
print(g.groupby("label")[["orig_tok", "orig_at_cap"]].median().to_string())
ok = T.finished_only.isin(["robust", "fragile"]); s = T[ok]; called = s.out.isin(["TP", "FP"])
print(f"\nstack calls against finished-only labels (no retraining): n={len(s)}, correct {int(((s.finished_only == 'fragile') == called).sum())}, fragile {(s.finished_only == 'fragile').sum()}, called fragile {int(called.sum())}; against current labels correct {int((s.label.eq('fragile') == called).sum())}")
Cn = pd.read_csv(HERE / "errors_why_contrib.csv", index_col=0)
e = d.join(Cn[["seg", "traj", "tx", "attn", "intercept", "p", "err"]], rsuffix="_c")
print("\n== fixed-option stack vs CV stack: same outcome on", int(((e.err >= .5) == e.wrong).sum()), "of", len(e), "| corr p", round(np.corrcoef(e.p, e.p_stack)[0, 1], 3))
print("mean logit contribution by outcome (positive pushes robust)"); print(e.groupby("out")[["seg", "traj", "tx", "attn", "intercept"]].mean().round(2).reindex(["TP", "FN", "FP", "TN"]).to_string())
print(e[e.wrong].sort_values(["out", "mode"])[["out", "model", "mode", "seg", "traj", "tx", "attn", "intercept", "p"]].round(2).to_string())
f = e[e.out == "FN"]; print("\nFN largest push to robust:", f[["seg", "traj", "tx"]].idxmax(axis=1).value_counts().to_dict(), "| FP largest push to fragile:", e[e.out == "FP"][["seg", "traj", "tx"]].idxmin(axis=1).value_counts().to_dict())
T.to_csv(HERE / "errors_why_relabel.csv")
