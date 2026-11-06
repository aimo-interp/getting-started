"""Is the seg probe cleanly separable? Per configuration: out-of-fold diff-in-means score (windows d1-5, d6-9, averaged as
in figure_data.py), against a null that shuffles labels and refits inside the folds. DIM is computed from the case-by-case
kernel, so refits are cheap. Folds are within configuration, k = min(5, fragile cases). The null matters: with few
fragile cases, held-out DIM scores are biased below .5, so .5 is not the right reference.

    python3 analysis/report/seg_separable.py > analysis/report/seg_separable.txt
"""
import itertools, json, math
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score as auc
ROOT = Path(__file__).resolve().parents[2]; DATA = ROOT / "data/cvbench"
safe = lambda m: m.replace("/", "_").replace(":", "_")
cases = [json.loads(l) for f in ("cases.jsonl", "cases_oss.jsonl") for l in open(DATA / f)]
cfg = np.array([c["model_id"].split("/")[-1] for c in cases]); y = np.array([c["is_robust"] for c in cases], bool)
plen = np.array([len(c["problem"]) for c in cases], float); prob = np.array([c["problem"] for c in cases])
FRAC, REPEATS, PERMS = .5, 5, 1000
SITES = {f"d{s + 1}": (s, s + 1) for s in range(10)} | {"early": (0, 5), "late": (5, 9)}
ORDER = ["Skywork-OR1-Math-7B", "Qwen3.5-4B", "DeepSeek-R1-0528-Qwen3-8B", "Olmo-3-7B-Think", "gpt-oss-120b:high", "gpt-oss-120b:medium", "gpt-oss-120b:low"]
segs, nresp = [], []
for c in cases:
    z = np.load(DATA / "segs" / safe(c["model_id"]) / f"{c['group']}.npz"); s = z["seg_mean"]
    segs.append(s[:, int(round(FRAC * (s.shape[1] - 1)))].astype(np.float64)); nresp.append(int(z["n_resp"]))
nresp = np.array(nresp); capped = nresp >= 4090
rng = np.random.default_rng(0)


def folds(l, k, r):
    g = np.random.default_rng(r); f = np.zeros(len(l), int)
    for cls in (True, False):
        i = np.where(l == cls)[0]; f[g.permutation(i)] = np.arange(len(i)) % k
    return f


def oof(Ks, l, k):
    """{site: mean held-out z-score over repeats}"""
    out = {s: np.zeros(len(l)) for s in Ks}
    for r in range(REPEATS):
        f = folds(l, k, r)
        for b in range(k):
            tr = np.where(f != b)[0]; te = np.where(f == b)[0]; lt = l[tr]
            w = lt / lt.sum() - ~lt / (~lt).sum()
            for s, K in Ks.items():
                pt = K[np.ix_(tr, tr)] @ w
                out[s][te] += (K[np.ix_(te, tr)] @ w - pt.mean()) / (pt.std() + 1e-12) / REPEATS
    out["E+L"] = (out["early"] + out["late"]) / 2
    return out


def adj(s, l, c, x):
    D = np.c_[x[c], np.ones(c.sum())]; return s[c] - D @ np.linalg.lstsq(D, s[c], rcond=None)[0]


print(f"null: {PERMS} shuffles or exact; {REPEATS} repeats; p_hi = P(null >= observed), p_lo = P(null <= observed)")
raw = []
for m in ORDER:
    i = np.where(cfg == m)[0]; l = y[i]; nf = int((~l).sum()); k = min(5, nf); c = capped[i]; x = np.log(plen[i])
    Ks = {}
    for s, (a, b) in SITES.items():
        A = np.stack([segs[j][a:b].mean(0) for j in i]); Ks[s] = A @ A.T
    O = oof(Ks, l, k); s = O["E+L"]; o = auc(l, s); best = max(O, key=lambda q: auc(l, O[q]))
    both = c.any() and l[c].all() != True and (~l[c]).sum() > 0
    a_cap = auc(l[c], s[c]) if both else np.nan; a_adj = auc(l[c], adj(s, l, c, x)) if both else np.nan
    N = len(l)
    perms = ([np.isin(np.arange(N), q) for q in itertools.combinations(range(N), int(l.sum()))] if math.comb(N, nf) <= 5000
             else [rng.permutation(l) for _ in range(PERMS)])
    nl = np.zeros((len(perms), 4))
    for p, q in enumerate(perms):
        Q = oof(Ks, q, k); v = Q["E+L"]; nl[p, 0] = auc(q, v); nl[p, 1] = max(auc(q, Q[w]) for w in Q)
        if both and 0 < q[c].sum() < c.sum():
            nl[p, 2] = auc(q[c], v[c]); nl[p, 3] = auc(q[c], adj(v, q, c, x))
        else:
            nl[p, 2:] = np.nan
    P = lambda v, n: (np.nansum(n >= v - 1e-9) + 1) / (np.sum(~np.isnan(n)) + 1)
    PL = lambda v, n: (np.nansum(n <= v + 1e-9) + 1) / (np.sum(~np.isnan(n)) + 1)
    p_hi = P(o, nl[:, 0]); raw.append(p_hi)
    th = np.sort(s); acc = max(((s >= t) == l).mean() for t in th); clean = s[l].min() > s[~l].max()
    print(f"== {m}: n={N} robust {l.sum()} fragile {nf} capped {c.sum()} (fragile {(~l[c]).sum()}), k={k}, {len(perms)} labelings")
    print(f"  E+L AUC {o:.3f}  null median {np.median(nl[:, 0]):.3f}  90% [{np.quantile(nl[:, 0], .05):.3f},{np.quantile(nl[:, 0], .95):.3f}]  p_hi {p_hi:.4f}  p_lo {PL(o, nl[:, 0]):.4f}")
    print(f"  best site {best} AUC {auc(l, O[best]):.3f}  null best median {np.median(nl[:, 1]):.3f}  p {P(auc(l, O[best]), nl[:, 1]):.4f}")
    print(f"  perfect split on mean score: {clean}; fragile above lowest robust {(s[~l] > s[l].min()).sum()}/{nf}; best-threshold acc (in-sample) {acc:.3f}, majority {max(l.mean(), 1 - l.mean()):.3f}")
    print(f"  hit-the-cap alone AUC {auc(l, -c.astype(float)):.3f} (fragile capped {(~l & c).sum()}/{nf}, robust capped {(l & c).sum()}/{l.sum()}); corr score ~ tokens {np.corrcoef(s, nresp[i])[0, 1] if np.ptp(nresp[i]) else float('nan'):.2f}")
    print(f"  capped AUC {a_cap:.3f} p {P(a_cap, nl[:, 2]) if both else float('nan'):.4f} | capped, log chars removed AUC {a_adj:.3f} p {P(a_adj, nl[:, 3]) if both else float('nan'):.4f}", flush=True)
print("Holm-adjusted p_hi, E+L:", dict(zip(ORDER, np.round(np.maximum.accumulate((len(raw) - np.arange(len(raw))) * np.sort(raw))[np.argsort(np.argsort(raw))].clip(max=1), 3))))

print("== transfer across gpt-oss efforts (same model, same space; train problems exclude test problems; E+L)")
E_ = ["high", "medium", "low"]
for a, b in itertools.permutations(E_, 2):
    ia = np.where(cfg == f"gpt-oss-120b:{a}")[0]; ib = np.where(cfg == f"gpt-oss-120b:{b}")[0]
    ia = ia[~np.isin(prob[ia], prob[ib])]; S = np.zeros(len(ib))
    if y[ia].all() or (~y[ia]).all():
        print(f"  {a:6s} -> {b:6s} train n={len(ia)} has one class, skipped"); continue
    for s0 in ("early", "late"):
        q, r = SITES[s0]; Xa = np.stack([segs[j][q:r].mean(0) for j in ia]); Xb = np.stack([segs[j][q:r].mean(0) for j in ib]); la = y[ia]
        d = Xa[la].mean(0) - Xa[~la].mean(0); pt = Xa @ d; S += (Xb @ d - pt.mean()) / pt.std()
    print(f"  {a:6s} -> {b:6s} train n={len(ia)} fragile {(~y[ia]).sum()} | test n={len(ib)} fragile {(~y[ib]).sum()} | AUC {auc(y[ib], S):.3f}")
