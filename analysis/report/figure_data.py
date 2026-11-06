"""Data for the report's seg / trajectory / attention figures, recomputed from raw per-decile files.
Prints the quoted numbers as a check and saves analysis/report/figure_data.npz; figures.py draws them.

    python3 analysis/report/figure_data.py > analysis/report/figure_data.txt
"""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import StratifiedGroupKFold
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/cvbench"
safe = lambda m: m.replace("/", "_").replace(":", "_")
cases = [json.loads(l) for f in ("cases.jsonl", "cases_oss.jsonl") for l in open(DATA / f)]
y = np.array([c["is_robust"] for c in cases], bool)
cfg = np.array([c["model_id"].split("/")[-1] for c in cases])
groups = np.array([c["group"] for c in cases])
FRAC, REPEATS, PERMS = .5, 20, 200
segs, nresp = [], []
for c in cases:
    z = np.load(DATA / "segs" / safe(c["model_id"]) / f"{c['group']}.npz")
    s = z["seg_mean"]
    segs.append(s[:, int(round(FRAC * (s.shape[1] - 1)))].astype(np.float32))
    nresp.append(int(z["n_resp"]))
nresp = np.array(nresp); capped = nresp >= 4090
models = sorted(set(cfg))
idx = {m: np.where(cfg == m)[0] for m in models}
print("capped", capped.sum(), "of", len(y))

# ------------------------------------------------------------- seg probe, out of fold
SITES = {f"d{s}": (s, s + 1) for s in range(10)} | {"early": (0, 5), "late": (5, 9)}
X = {k: {m: np.stack([segs[i][a:b].mean(0) for i in idx[m]]) for m in models} for k, (a, b) in SITES.items()}


def oof(labels, repeats, sites):
    """score[site][repeat, case]: held-out diff-in-means projection, z-scored on training projections."""
    out = {k: np.zeros((repeats, len(labels))) for k in sites}
    for r in range(repeats):
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(np.zeros(len(labels)), labels, groups))
        for tr, te in folds:
            mask = np.zeros(len(labels), bool); mask[tr] = True
            for m in models:
                ii = idx[m]; t = mask[ii]; yt = labels[ii][t]
                if yt.all() or (~yt).all():
                    continue
                for k in sites:
                    A = X[k][m]; At = A[t]
                    d = At[yt].mean(0) - At[~yt].mean(0); d /= np.linalg.norm(d) + 1e-12
                    mu = At.mean(0); sd = ((At - mu) @ d).std() + 1e-12
                    out[k][r, ii[~t]] = (A[~t] - mu) @ d / sd
    return out


def rep_auc(S, labels):
    return np.mean([auc(labels, s) for s in S])


S = oof(y, REPEATS, list(SITES))
dec_auc = np.array([rep_auc(S[f"d{s}"], y) for s in range(10)])
win_auc = {k: rep_auc(S[k], y) for k in ("early", "late")}
both = (S["early"] + S["late"]) / 2
print("decile OOF AUC", dec_auc.round(3), "windows", {k: round(v, 3) for k, v in win_auc.items()}, "E+L %.3f" % rep_auc(both, y))
rng = np.random.default_rng(0)
null = np.zeros((PERMS, 10))
for p in range(PERMS):
    lab = y.copy()
    for m in models:
        lab[idx[m]] = rng.permutation(lab[idx[m]])
    N = oof(lab, 2, [f"d{s}" for s in range(10)])
    null[p] = [rep_auc(N[f"d{s}"], lab) for s in range(10)]
band = np.quantile(null, [.05, .95], axis=0)
print("null 90% band mean", band.mean(1).round(3), "p per decile", ((null >= dec_auc).mean(0)).round(3))
score = both.mean(0)
print("per config OOF AUC", {m: round(auc(y[idx[m]], score[idx[m]]), 2) for m in models})

# ------------------------------------------------------------- trajectory
steps = np.stack([np.log(np.linalg.norm(np.diff(s[:9], axis=0), axis=1) + 1e-12) for s in segs])
norms = np.stack([np.linalg.norm(s[:9], axis=1) for s in segs]); norms = norms / norms.mean(1, keepdims=True)
Zs, Zn = steps.copy(), norms.copy()
for m in models:
    i = idx[m]
    Zs[i] = (steps[i] - steps[i].mean(0)) / (steps[i].std(0) + 1e-9)
    Zn[i] = (norms[i] - norms[i].mean(0)) / (norms[i].std(0) + 1e-9)
step_mean = steps.mean(1); norm_slope = np.array([np.polyfit(np.arange(9), n, 1)[0] for n in norms])


def zw(v):
    o = v.copy()
    for m in models:
        o[idx[m]] = (v[idx[m]] - v[idx[m]].mean()) / (v[idx[m]].std(ddof=1) + 1e-9)
    return o


zstep, zslope = zw(step_mean), zw(norm_slope)
print("step_mean AUC %.3f (capped %.3f) | falling norm_slope AUC %.3f (capped %.3f)" % (
    auc(y, zstep), auc(y[capped], zstep[capped]), auc(y, -zslope), auc(y[capped], -zslope[capped])))

# ------------------------------------------------------------- attention, Skywork
sky = idx["Skywork-OR1-Math-7B"]
R = np.stack([np.load(DATA / "circuit_attn" / safe(cases[i]["model_id"]) / f"{cases[i]['group']}.npz")["regions"][6:].mean((0, 1))
              for i in sky])   # [cases, deciles, regions: sink, template, problem, far, recent]
ys = y[sky]; sink = R[:, :, 0] + R[:, :, 1]; far = R[:, :, 3]; prob = R[:, :, 2]; recent = R[:, :, 4]
print("mean region mass", R.mean((0, 1)).round(3))


def perm(v, n=20000, seed=0):
    r = np.random.default_rng(seed); o = auc(ys, v)
    ranks = np.argsort(np.argsort(v)) + 1; n1 = ys.sum(); n0 = len(ys) - n1
    a = (np.array([ranks[r.permutation(ys)].sum() for _ in range(n)]) - n1 * (n1 + 1) / 2) / (n0 * n1)
    return o, ((np.abs(a - .5) >= abs(o - .5)).sum() + 1) / (n + 1)


caps = capped[sky]
for name, v in (("sink whole", sink.mean(1)), ("sink late 6-9", sink[:, 5:9].mean(1)), ("far 1-9 (fragile higher)", -far[:, :9].mean(1)),
                ("problem whole", prob.mean(1))):
    a, p = perm(v)
    print(f"{name}: AUC {a:.3f} p={p:.4f} | capped n={caps.sum()} frag={(~ys[caps]).sum()} AUC {auc(ys[caps], v[caps]):.3f}")
print("sink by decile AUC", np.array([auc(ys, sink[:, s]) for s in range(10)]).round(2))
print("far by decile AUC (fragile higher)", np.array([auc(ys, -far[:, s]) for s in range(10)]).round(2))
print("corr sink-late with tokens", np.corrcoef(sink[:, 5:9].mean(1), nresp[sky])[0, 1].round(2),
      "| with seg score", np.corrcoef(sink[:, 5:9].mean(1), score[sky])[0, 1].round(2))
np.savez(ROOT / "analysis/report/figure_data.npz", y=y, cfg=cfg, dec_auc=dec_auc, band=band, win=np.array([win_auc["early"], win_auc["late"]]),
         score=score, Zs=Zs, Zn=Zn, ys=ys, sink=sink, far=far, zstep=zstep, zslope=zslope)

# ------------------------------------------------------------- confound checks
print("\n== confound checks")
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import StratifiedGroupKFold as SGK
plen = np.array([len(c["problem"]) for c in cases], float)
print("Skywork tokens AUC (shorter=robust) %.3f | uncapped n=%d fragile=%d | capped n=%d fragile=%d" % (
    auc(ys, -nresp[sky]), (~caps).sum(), (~ys[~caps]).sum(), caps.sum(), (~ys[caps]).sum()))
late = sink[:, 5:9].mean(1)
print("Skywork late sink mean: uncapped %.3f capped %.3f | capped range %.3f-%.3f, uncapped %.3f-%.3f" % (
    late[~caps].mean(), late[caps].mean(), late[caps].min(), late[caps].max(), late[~caps].min(), late[~caps].max()))


def perm_sub(v, lab, strata, n=20000, seed=1):
    r = np.random.default_rng(seed); o = auc(lab, v); a = np.zeros(n)
    for k in range(n):
        l = lab.copy()
        for s in set(strata):
            j = np.where(strata == s)[0]; l[j] = r.permutation(l[j])
        a[k] = auc(l, v)
    return o, ((np.abs(a - .5) >= abs(o - .5)).sum() + 1) / (n + 1)


for name, v in (("late sink", late), ("whole sink", sink.mean(1)), ("far 1-9 (fragile higher)", -far[:, :9].mean(1))):
    print("Skywork capped only: %s AUC %.3f p=%.4f" % ((name,) + perm_sub(v[caps], ys[caps], np.zeros(caps.sum()))))
print("Skywork capped: corr late sink ~ problem chars %.2f; problem chars AUC %.3f" % (
    np.corrcoef(late[caps], plen[sky][caps])[0, 1], auc(ys[caps], plen[sky][caps])))
print("Skywork all: problem chars AUC %.3f, corr late sink %.2f" % (auc(ys, plen[sky]), np.corrcoef(late, plen[sky])[0, 1]))
# other models with attention weights; all their traces are capped
rows = []
for m in ("Skywork-OR1-Math-7B", "Olmo-3-7B-Think", "DeepSeek-R1-0528-Qwen3-8B"):
    ii = idx[m]; ii = ii[capped[ii]]
    v = []
    for i in ii:
        z = np.load(DATA / "circuit_attn" / safe(cases[i]["model_id"]) / f"{cases[i]['group']}.npz")
        Q = z["regions"] if "layers" in z else z["regions"][6:]
        v.append(np.nanmean(Q[:, :, 5:9, 0] + Q[:, :, 5:9, 1]))
    v = np.array(v); print(m, "n", len(ii), "fragile", (~y[ii]).sum(), "late sink AUC %.3f" % auc(y[ii], v))
    rows.append((ii, (v - v.mean()) / v.std(ddof=1)))
I = np.concatenate([r[0] for r in rows]); V = np.concatenate([r[1] for r in rows])
print("capped pooled, within-model z: n %d AUC %.3f p=%.4f" % ((len(I),) + perm_sub(V, y[I], cfg[I])))
print("capped pooled excl. Skywork: AUC %.3f p=%.4f" % perm_sub(V[34:], y[I][34:], cfg[I][34:]))

# seg/last decile: does a score turn into accuracy? logistic on OOF score, grouped 5-fold, vs always robust .703
def acc(sc, r=0):
    hit = np.zeros(len(y))
    for tr, te in SGK(5, shuffle=True, random_state=r).split(sc[:, None], y, groups):
        hit[te] = LR(C=1).fit(sc[tr, None], y[tr]).predict(sc[te, None]) == y[te]
    return hit.mean()
print("accuracy from single OOF score:", {k: round(np.mean([acc(S[k][r], r) for r in range(REPEATS)]), 3) for k in ("d9", "d3", "d7", "early", "late")},
      "E+L", round(np.mean([acc(both[r], r) for r in range(REPEATS)]), 3))
print("seg E+L AUC capped %.3f | d9 capped %.3f" % (rep_auc(both[:, capped], y[capped]), rep_auc(S["d9"][:, capped], y[capped])))
gt = capped.astype(float); print("is_capped alone within-model: AUC %.3f" % auc(y, -gt))
print("seg E+L by config, capped only:", {m: round(auc(y[idx[m]][capped[idx[m]]], score[idx[m]][capped[idx[m]]]), 2) for m in models
      if len(set(y[idx[m]][capped[idx[m]]])) == 2})

# late sink after removing problem length (capped: trace length is fixed), within model
print("\n== sink residual after log problem chars, capped traces")
sky_v = late[caps]; L = np.log(plen[sky][caps]); b = np.polyfit(L, sky_v, 1); res = sky_v - np.polyval(b, L)
print("Skywork capped R2 on log problem chars %.2f | residual AUC %.3f p=%.4f" % ((np.corrcoef(L, sky_v)[0, 1] ** 2,) + perm_sub(res, ys[caps], np.zeros(caps.sum()))))
R_, I_ = [], []
for ii, zv in rows:
    l = np.log(plen[ii]); r_ = zv - np.polyval(np.polyfit(l, zv, 1), l); R_.append(r_ / r_.std(ddof=1)); I_.append(ii)
R_, I_ = np.concatenate(R_), np.concatenate(I_)
print("pooled three models residual AUC %.3f p=%.4f; problem chars alone within-model AUC %.3f" % (perm_sub(R_, y[I_], cfg[I_]) + (auc(y[I_], -np.concatenate([(np.log(plen[ii]) - np.log(plen[ii]).mean()) / np.log(plen[ii]).std(ddof=1) for ii, _ in rows])),)))
print("seg E+L vs problem chars: within-model corr %.2f; trace tokens corr on Skywork %.2f" % (np.corrcoef(zw(np.log(plen)), zw(score))[0, 1], np.corrcoef(score[sky], nresp[sky])[0, 1]))
print("step_mean vs problem chars within-model corr %.2f; norm_slope %.2f" % (np.corrcoef(zw(np.log(plen)), zstep)[0, 1], np.corrcoef(zw(np.log(plen)), zslope)[0, 1]))
print("AUC problem chars within-model, all 118 (shorter=robust) %.3f" % auc(y, -zw(np.log(plen))))
np.savez(ROOT / "analysis/report/figure_data.npz", y=y, cfg=cfg, dec_auc=dec_auc, band=band, win=np.array([win_auc["early"], win_auc["late"]]),
         score=score, Zs=Zs, Zn=Zn, ys=ys, sink=sink, far=far, caps=caps, ntok=nresp[sky], capped=capped, plen=plen, sky=sky, res=res)
