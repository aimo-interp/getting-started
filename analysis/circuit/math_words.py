"""Attention to math vs non-math problem tokens (Skywork, 11 focus heads, from circuit_attn ptok arrays).
Token classes: sink (first 2 problem tokens, which act as attention sinks), symbol (digits/LaTeX/operators),
math word (mathematical vocabulary), function word, other (narrative content: names, story words)."""
import re, sys, numpy as np, pandas as pd
sys.path.insert(0, "cvbench")
import cv, circuit_attr as ca
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(ca.MODEL, local_files_only=True)
MATH = r"integer|prime|divis|remainder|gcd|lcm|digit|sum|product|number|real|positive|negative|triangle|circle|square|" \
       r"angle|polygon|radius|area|perimeter|point|line|segment|plane|sphere|tangent|perpendicular|parallel|probab|" \
       r"ways|choose|sequence|function|polynomial|root|equation|value|maximum|minimum|ratio|fraction|even|odd|" \
       r"factor|multiple|mod|coordinate|axis|length|distance|volume|expected|set|subset|order|consecutive|divide"
FUNC = {"the", "a", "an", "of", "is", "are", "and", "to", "in", "that", "be", "for", "with", "as", "by", "on", "it",
        "its", "this", "each", "has", "have", "can", "what", "which", "such", "find", "let", "if", "then", "there",
        "at", "from", "or", "how", "many", "all", "any", "so", "we", "his", "her", "their", "they", "he", "she"}
def cls(t, pos):
    if pos < 2: return "sink"
    s = t.strip().lower()
    if not s or re.fullmatch(r"[\d\$\\\^_{}()\[\]+\-*/=<>.,:;!?'|&~%]+", s) or re.search(r"\d|\\|\$", s): return "symbol"
    if s in FUNC: return "function"
    if re.match(MATH, s): return "math word"
    return "other"
c = cv.load_cases(list(cv.CASE_FILES)); c = c[c.model_id == ca.MODEL]
y = c.is_robust.to_numpy(bool)
rows = []
for g, yy in zip(c.group, y):
    z = np.load(ca.ATT / f"{g}.npz")
    ids = z["problem_ids"]; labs = np.array([cls(tok.decode([int(i)]), k) for k, i in enumerate(ids)])
    for w in ca.WINDOWS:
        P = z[f"ptok_{w}"]                                  # [heads, P]
        for j, h in enumerate(ca.FOCUS):
            v = P[j] / (P[j].sum() + 1e-9)
            for k in ["symbol", "math word", "function", "other", "sink"]:
                m = labs == k
                rows.append(dict(group=g, rob=yy, w=w, head=h, cls=k, share=v[m].sum(), frac_tok=m.mean(),
                                 per_tok=(v[m].sum() / m.mean()) if m.any() else np.nan))
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("Token class sizes (mean fraction of problem tokens):", df[(df.w == "w05") & (df.head == ca.FOCUS[0])].groupby("cls").frac_tok.mean().round(3).to_dict())
print("\nAttention share per class, normalised by its token fraction (1 = uniform), excluding nothing; mean over cases")
print(df.pivot_table(index=["w", "head"], columns="cls", values="per_tok", aggfunc="mean").round(2).to_string())
print("\nNon-sink only: share of non-sink attention going to math (symbol+math word) vs 'other' narrative words")
ns = df[df.cls != "sink"].copy()
tot = ns.groupby(["group", "w", "head"]).share.transform("sum"); ns["ns_share"] = ns.share / tot
mm = ns[ns.cls.isin(["symbol", "math word"])].groupby(["group", "rob", "w", "head"]).ns_share.sum().rename("math").reset_index()
ot = ns[ns.cls == "other"].groupby(["group", "w", "head"]).ns_share.sum().rename("other").reset_index()
mm = mm.merge(ot)
for (w, h), g in mm.groupby(["w", "head"]):
    a = roc_auc_score(g.rob, g.math) if g.rob.nunique() == 2 else np.nan
    print(f"  {w} {h:8s} math share R {g[g.rob].math.mean():.3f} F {g[~g.rob].math.mean():.3f} | other R {g[g.rob].other.mean():.3f} F {g[~g.rob].other.mean():.3f} | AUC(math share -> robust) {a:.2f}")
agg = mm.groupby(["group", "rob"]).math.mean().reset_index()
print(f"\nPooled over 11 heads x 2 windows: math share R {agg[agg.rob].math.mean():.3f} F {agg[~agg.rob].math.mean():.3f}, AUC {roc_auc_score(agg.rob, agg.math):.2f}")
