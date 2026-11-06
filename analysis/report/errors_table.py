"""Join OOF predictions with per-case context: solve rate, did the greedy 4k trace finish / get it right, length, topic, drop.

    python3 analysis/report/errors_table.py   # -> analysis/report/errors_cases.csv
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; HERE = Path(__file__).parent; DATA = ROOT / "data/cvbench"
safe = lambda m: m.replace("/", "_").replace(":", "_")
cases = [json.loads(l) for f in ("cases.jsonl", "cases_oss.jsonl") for l in open(DATA / f)]
o = pd.read_csv(ROOT / "analysis/seg_errors/cases_annotated.csv", index_col=0)
Z = np.load(HERE / "errors_oof.npz", allow_pickle=True); ids = Z["case_id"].astype(str); y = Z["y"]
d = pd.DataFrame({"case_id": ids, "robust": y}).set_index("case_id")
c_ = {c["case_id"]: c for c in cases}
d["model"] = [o.m[i] for i in d.index]; d["group"] = [c_[i]["group"] for i in d.index]; d["source"] = [c_[i]["source"] for i in d.index]
d["max_drop"] = [c_[i]["max_drop"] for i in d.index]; d["chars"] = [len(c_[i]["problem"]) for i in d.index]
d["problem"] = [c_[i]["problem"] for i in d.index]
d["tokens"] = [int(np.load(DATA / "segs" / safe(c_[i]["model_id"]) / f"{c_[i]['group']}.npz")["n_resp"]) for i in d.index]
d["capped"] = d.tokens >= 4090
d["agree"] = o.agree.reindex(d.index)
d["attn_data"] = [(DATA / "circuit_attn" / safe(c_[i]["model_id"]) / f"{c_[i]['group']}.npz").is_file() for i in d.index]
d["seg_gated_off"] = o.gated_off.reindex(d.index)
for k in ("seg", "traj", "attn", "stack", "whitebox", "tx"):
    P = Z[k]; d[f"p_{k}"] = P.mean(0); d[f"flag_{k}"] = (P < .5).mean(0)
# solve rate on the original: 8 samples at 16k (organiser cases), 4 samples (generated cases)
oc = pd.read_csv(HERE / "organiser_competence_cases.csv", index_col=0)
d["solve"] = oc.solve_rate.reindex(d.index); d["solve_n"] = np.where(d.solve.notna(), 8, 4)
d["solve"] = d.solve.fillna(o.base_acc.reindex(d.index))
g = dict(zip(oc.group, oc.gold))
for l in open(ROOT / "data/labelgen/problems.jsonl"):
    p = json.loads(l); a = p.get("answer", p.get("gold"))
    k = p.get("problem_id") or p.get("id")
    if a is not None and k is not None: g.setdefault(k, a)
pid = {}
for l in open(ROOT / "data/labelgen/labels.jsonl"):
    r = json.loads(l); pid[r["problem"]] = r.get("problem_id")


def last_boxed(t):
    m = re.findall(r"\\boxed\{\s*([^{}]*?)\s*\}", t or ""); return m[-1].replace(",", "").strip() if m else None


G = {}
for f in (DATA / "greedy").glob("*.jsonl"):
    for l in open(f):
        r = json.loads(l); G[(r["model_id"], r["group"])] = r.get("generated_text", "")
ans, ok = [], []
for i in d.index:
    c = c_[i]; a = last_boxed(G.get((c["model_id"], c["group"]))); ans.append(a)
    gold = g.get(c["group"], g.get(pid.get(c["problem"])))
    ok.append(np.nan if gold is None else float(a is not None and str(a) == str(gold).replace(",", "").strip()))
d["greedy_answer"] = ans; d["greedy_correct"] = ok; d["greedy_boxed"] = [a is not None for a in ans]
hits = {k: None for k in ("geometry", "number", "combinatorics")}
KW = {"geometry": r"triangle|circle|square|angle|polygon|radius|coordinate|axis|segment|perpendicular|tangent|area|rectangle|hexagon|plane|sphere|parallel|chord|diameter|\\cos|\\sin",
      "number": r"prime|divisor|divisible|remainder|gcd|integer|digit|mod|factor|multiple of",
      "combinatorics": r"probability|ways|choose|arrange|random|count|subsets?|sequences?|permutation|game"}
H = np.array([[len(re.findall(KW[k], p.lower())) for k in KW] for p in d.problem])
d["topic"] = np.where(H.max(1) > 0, np.array(list(KW))[H.argmax(1)], "other")
d.to_csv(HERE / "errors_cases.csv")
print(d.shape, "greedy_correct known", d.greedy_correct.notna().sum(), "solve known", d.solve.notna().sum())
print("thr: max robust drop", d.max_drop[d.robust].max(), "min fragile drop", d.max_drop[~d.robust].min())
print(d.groupby("source").agg(n=("robust", "size"), robust=("robust", "mean"), greedy=("greedy_correct", "mean"), solve=("solve", "mean")))
