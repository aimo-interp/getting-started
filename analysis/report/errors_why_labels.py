"""How each generated label arose: per variant sample, right / wrong answer / unfinished, and whether numbers changed in the variant.

    python3 analysis/report/errors_why_labels.py > analysis/report/errors_why_labels.txt
"""
import json, re, glob
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; HERE = Path(__file__).parent; LG = ROOT / "data/labelgen"
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 300, "display.max_colwidth", 70)
d = pd.read_csv(HERE / "errors_stack_cases.csv", index_col=0)
c_ = {json.loads(l)["case_id"]: json.loads(l) for f in ("cases.jsonl", "cases_oss.jsonl") for l in open(ROOT / "data/cvbench" / f)}
pid = {}
for l in open(LG / "labels.jsonl"):
    r = json.loads(l); pid[(r["model_id"], r["problem"])] = r
PR = {json.loads(l)["problem_id"]: json.loads(l) for l in open(LG / "problems.jsonl")}
V = {}
for l in open(LG / "variants.jsonl"):
    r = json.loads(l); V[(r["problem_id"], r["type"], r["variant"])] = r["text"]
S = {}
for f in glob.glob(str(LG / "samples/*__default.jsonl")):
    for l in open(f):
        r = json.loads(l); S.setdefault((r["model_id"], r["problem_id"]), []).append(r)
nums = lambda t: sorted(re.findall(r"\d+(?:\.\d+)?", t))
gold_key = next((k for k in ("answer", "gold", "reference") if k in next(iter(PR.values()))), None); print("gold key", gold_key)
rows, fail = [], []
for i, r in d[d.source == "labelgen"].iterrows():
    c = c_[i]; lb = pid[(c["model_id"], c["problem"])]; p = lb["problem_id"]; rs = S.get((c["model_id"], p), [])
    o = [x for x in rs if x["type"] == "original"]; v = [x for x in rs if x["type"] != "original"]
    tk = [t for x in o for t in x["n_tokens"]]; cap = max([t for x in rs for t in x["n_tokens"]] or [0])
    A = dict(right=0, wrong=0, unfinished=0, numchg=0)
    for x in v:
        ch = nums(V.get((p, x["type"], x["variant"]), "")) != nums(c["problem"])
        for a, ok, t in zip(x["answers"], x["correct"], x["n_tokens"]):
            kind = "right" if ok else ("unfinished" if a is None else "wrong"); A[kind] += 1
            if not ok:
                fail.append(dict(case=i, out=r.out, type=x["type"], variant=x["variant"], kind=kind, answer=a, gold=PR[p].get(gold_key), tokens=t, numbers_changed=ch)); A["numchg"] += ch
    of = sum(o[0]["correct"]) if o else 0
    rows.append(dict(case=i, out=r["out"], model=r.model, mode=r["mode"], drop=r.max_drop, types=len(lb["type_accuracy"]), orig_right=of, orig_n=len(o[0]["correct"]) if o else 0, orig_med_tok=int(np.median(tk)) if tk else 0, cap=cap, **A))
T = pd.DataFrame(rows).set_index("case"); F = pd.DataFrame(fail)
T["var_n"] = T.right + T.wrong + T.unfinished; T["fail_rate"] = 1 - T.right / T.var_n
print(T.groupby("out")[["right", "wrong", "unfinished", "numchg"]].sum().to_string()); print(T.groupby("out")[["orig_med_tok", "fail_rate", "types"]].median().round(2).to_string())
print("\ncap per model:", T.groupby("model").cap.max().to_dict())
print("\n== stack errors, generated cases"); print(T[T.out.isin(["FN", "FP"])].sort_values(["out", "fail_rate"]).drop(columns=["model", "cap"]).round(2).to_string())
print("\n== caught fragile"); print(T[T.out == "TP"].sort_values("fail_rate").drop(columns=["model", "cap"]).round(2).to_string())
print("\n== failed samples on missed fragile cases"); print(F[F.out == "FN"].drop(columns=["out"]).assign(case=lambda x: x.case.str[-12:]).to_string(index=False))
print("\n== share of failed samples by kind, fragile cases"); print(F[F.out.isin(["TP", "FN"])].groupby("out").kind.value_counts().to_string())
print("\n== text of failed variants whose numbers changed, missed fragile")
seen = set()
for r in F[(F.out == "FN") & F.numbers_changed].itertuples():
    p = pid[(c_[r.case]["model_id"], c_[r.case]["problem"])]["problem_id"]; k = (p, r.type, r.variant)
    if k in seen: continue
    seen.add(k); a = nums(c_[r.case]["problem"]); b = nums(V[k]); print(f"{r.case[-12:]} {r.type}: only original {sorted(set(a) - set(b))[:8]} only variant {sorted(set(b) - set(a))[:8]} | {V[k][:170]!r}")
T.to_csv(HERE / "errors_why_labels.csv"); F.to_csv(HERE / "errors_why_failed_samples.csv", index=False)
