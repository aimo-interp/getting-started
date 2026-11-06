"""Is any attention pattern cleanly separable? For each model search region x window and layer x head x region x window
features; compare the best AUC with the best under shuffled labels (exact when small), check held-out selection, and
leave-one-model-out transfer. Variant 'adj': capped traces, log problem chars regressed out.

    python3 analysis/report/attn_separable.py > analysis/report/attn_separable.txt
"""
import json, itertools
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
ROOT = Path(__file__).resolve().parents[2]; DATA = ROOT / "data/cvbench"
safe = lambda m: m.replace("/", "_").replace(":", "_")
cases = [json.loads(l) for f in ("cases.jsonl", "cases_oss.jsonl") for l in open(DATA / f)]
MODELS = [("Skywork/Skywork-OR1-Math-7B", "Skywork"), ("allenai/Olmo-3-7B-Think", "Olmo"),
          ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "DeepSeek"), ("Qwen/Qwen3.5-4B", "Qwen3.5")]
REG = ["first", "template", "problem", "far", "recent", "first+template"]
WIN = {f"d{i+1}": (i, i + 1) for i in range(10)} | {"d1-5": (0, 5), "d6-9": (5, 9), "all": (0, 10)}
rng = np.random.default_rng(0)


def load(mid):
    Q, y, pl, nt = [], [], [], []
    for c in cases:
        f = DATA / "circuit_attn" / safe(mid) / f"{c['group']}.npz"
        if c["model_id"] == mid and f.is_file():
            z = np.load(f); r = z["regions"] if "layers" in z else z["regions"][6:]
            Q.append(r); y.append(c["is_robust"]); pl.append(len(c["problem"]))
            nt.append(int(np.load(DATA / "segs" / safe(mid) / f"{c['group']}.npz")["n_resp"]))
    return np.stack(Q), np.array(y, bool), np.array(pl, float), np.array(nt)


def extend(R):                                   # [..., deciles, 5] -> [..., windows, 6]
    R = np.concatenate([R, R[..., :2].sum(-1, keepdims=True)], -1)
    return np.stack([R[..., a:b, :].mean(-2) for a, b in WIN.values()], -2)


def feats(Q):
    A = extend(np.nanmean(Q, (1, 2)))             # region level: [n, W, 6]
    names_a = [f"{r} {w}" for w in WIN for r in REG]
    ok = ~np.isnan(Q).all((0, 2, 3, 4)); Qk = Q[:, ok]; ls = np.where(ok)[0]
    H = np.stack([Qk[:, :, :, a:b].mean(3) for a, b in ((0, 5), (5, 9))], 3)        # [n, L, H, 2, 5]
    H = np.concatenate([H, H[..., :2].sum(-1, keepdims=True)], -1)
    names_h = [f"layer#{l} head{h} {w} {r}" for l in ls for h in range(Q.shape[2]) for w in ("d1-5", "d6-9") for r in REG]
    return A.reshape(len(A), -1), names_a, H.reshape(len(H), -1), names_h


def aucs(X, l):
    n1 = l.sum(); n0 = len(l) - n1
    return (rankdata(X, axis=0)[l].sum(0) - n1 * (n1 + 1) / 2) / (n0 * n1)


def nulls(X, l, n=2000):
    k = int(l.sum()); N = len(l); R = rankdata(X, axis=0); n0 = N - k
    from math import comb
    if comb(N, k) <= 5000:
        ps = [np.isin(np.arange(N), c) for c in itertools.combinations(range(N), k)]
    else:
        ps = [rng.permutation(l) for _ in range(n)]
    return np.array([np.abs((R[p].sum(0) - k * (k + 1) / 2) / (n0 * k) - .5).max() + .5 for p in ps])


def report(tag, X, names, y):
    X = X[:, np.ptp(X, 0) > 0]if False else X
    a = aucs(X, y); o = np.abs(a - .5) + .5; i = int(o.argmax()); nl = nulls(X, y)
    print(f"  {tag:18s} {X.shape[1]:6d} feats | best {o[i]:.3f} ({names[i]}, {'robust' if a[i] > .5 else 'fragile'} higher) | "
          f"perfect splits {(o > .9999).sum()} | null best median {np.median(nl):.3f} | p(max) {((nl >= o[i] - 1e-9).sum() + 1) / (len(nl) + 1):.3f}")


def heldout(X, y, repeats=20):
    k = min(5, int((~y).sum()), int(y.sum())); out = []
    for r in range(repeats):
        s = np.zeros(len(y))
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=r).split(y, y):
            a = aucs(X[tr], y[tr]); j = int(np.abs(a - .5).argmax()); sg = np.sign(a[j] - .5) or 1
            mu, sd = X[tr, j].mean(), X[tr, j].std() + 1e-12
            s[te] = sg * (X[te, j] - mu) / sd
        out.append(aucs(s[:, None], y)[0])
    return np.mean(out)


prof = {}
for mid, name in MODELS:
    Q, y, pl, nt = load(mid); A, na, H, nh = feats(Q)
    print(f"== {name}: n={len(y)} robust {y.sum()} fragile {(~y).sum()} capped {(nt >= 4090).sum()}")
    report("region", A, na, y); report("head", H, nh, y)
    print(f"  held-out selected feature AUC: region {heldout(A, y):.3f}  head {heldout(H, y):.3f}")
    c = nt >= 4090; L = np.log(pl[c]); D = np.c_[L, np.ones(c.sum())]
    resid = lambda X: X[c] - D @ np.linalg.lstsq(D, X[c], rcond=None)[0]
    Ar, Hr = resid(A), resid(H)
    if (~y[c]).sum() and y[c].sum():
        report("region adj", Ar, na, y[c]); report("head adj", Hr, nh, y[c])
        print(f"  held-out adj: region {heldout(Ar, y[c]):.3f}  head {heldout(Hr, y[c]):.3f}")
    prof[name] = aucs(Ar, y[c])
ns = [m[1] for m in MODELS]; P = np.stack([prof[m] for m in ns])
print("== shared pattern: correlation of region-level AUC profiles (length-adjusted)")
print(np.round(np.corrcoef(P), 2)); print(ns)
print("leave-one-model-out: feature chosen on the other three, AUC on the held-out model (length-adjusted)")
for k, m in enumerate(ns):
    o = np.delete(P, k, 0).mean(0) - .5; j = int(np.abs(o).argmax()); v = P[k, j] if o[j] > 0 else 1 - P[k, j]
    print(f"  {m:9s} {na[j]:24s} {'robust' if o[j] > 0 else 'fragile'} higher, others mean {abs(o[j]) + .5:.3f}, held-out {v:.3f}")
