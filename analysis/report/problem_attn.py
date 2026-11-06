"""Does attention to the problem text differ between robust and fragile traces? Skywork, capped traces control trace length."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score as auc
ROOT = Path(__file__).resolve().parents[2]; DATA = ROOT / "data/cvbench"
safe = lambda m: m.replace("/", "_").replace(":", "_")
d = np.load(ROOT / "analysis/report/figure_data.npz", allow_pickle=True)
cases = [json.loads(l) for f in ("cases.jsonl", "cases_oss.jsonl") for l in open(DATA / f)]
sky = d["sky"]; ys = d["ys"]; caps = d["caps"]; plen = d["plen"][sky]
R = np.stack([np.load(DATA / "circuit_attn" / safe(cases[i]["model_id"]) / f"{cases[i]['group']}.npz")["regions"][6:].mean((0, 1)) for i in sky])
P = R[:, :, 2]; rng = np.random.default_rng(0)


def perm(v, l, n=20000):
    o = auc(l, v); a = np.array([auc(rng.permutation(l), v) for _ in range(n)])
    return o, ((np.abs(a - .5) >= abs(o - .5)).sum() + 1) / (n + 1)


print("problem mass mean robust %.4f fragile %.4f (all) | capped robust %.4f fragile %.4f" % (P[ys].mean(), P[~ys].mean(), P[ys & caps].mean(), P[~ys & caps].mean()))
print("by decile robust", P[ys].mean(0).round(3)); print("by decile fragile", P[~ys].mean(0).round(3))
for name, v, l in (("all, whole", P.mean(1), ys), ("all, early 1-5", P[:, :5].mean(1), ys), ("all, late 6-9", P[:, 5:9].mean(1), ys),
                   ("capped, whole", P[caps].mean(1), ys[caps]), ("capped, late", P[caps, 5:9].mean(1), ys[caps])):
    a, p = perm(v, l); print(f"{name}: AUC for robust {a:.3f}  fragile-higher AUC {1 - a:.3f}  p={p:.3f}")
x = np.log(plen); print("corr problem mass ~ log problem chars: all %.2f capped %.2f" % (np.corrcoef(P.mean(1), x)[0, 1], np.corrcoef(P[caps].mean(1), x[caps])[0, 1]))
v = P[caps].mean(1); xc = x[caps]; res = v - np.polyval(np.polyfit(xc, v, 1), xc)
a, p = perm(res, ys[caps]); print("capped, problem length removed: AUC for robust %.3f p=%.3f" % (a, p))
print("per-char mass capped robust %.3e fragile %.3e" % ((v / plen[caps])[ys[caps]].mean(), (v / plen[caps])[~ys[caps]].mean()))
