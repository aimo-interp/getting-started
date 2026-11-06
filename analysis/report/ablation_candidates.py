"""Matched ablation of the candidate submissions (stack-attn, stack-ds5, pca3): every row uses the same outer folds,
gated rows nest the per-model option choice as stack.run_gated does. CI = bootstrap over problems of the Δaccuracy
(mean over repeats per case), because the repeat sd only measures fold noise, not which 118 cases we happened to get.

    python analysis/report/ablation_candidates.py --repeats 20 --save analysis/report/ablation_candidates.npz
"""
import argparse, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold as SGK
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cvbench"))
import cv, stack, whitebox as wb

ap = argparse.ArgumentParser()
ap.add_argument("--repeats", type=int, default=20)
ap.add_argument("--save", default=None)
args = ap.parse_args()
args.crossfit = False; args.sample = 0; args.frac = 0.5

def attn_cfg(tx):
    s, t, a = "seg_safe", "traj", "attn"
    return {
        f"attn FULL ({tx})":    [f"{s}+{t}+{tx}+{a}", f"{t}+{tx}+{a}", f"{s}+{t}+{tx}", f"{t}+{tx}"],
        f"attn -attn ({tx})":   [f"{s}+{t}+{tx}", f"{t}+{tx}"],
        f"attn -seg ({tx})":    [f"{t}+{tx}+{a}", f"{t}+{tx}"],
        f"attn -traj ({tx})":   [f"{s}+{tx}+{a}", f"{tx}+{a}", f"{s}+{tx}", tx],
        f"attn -tx ({tx})":     [f"{s}+{t}+{a}", f"{t}+{a}", f"{s}+{t}", t],
        f"tx only ({tx})":      [tx],
    }
CFG = {**attn_cfg("tx"), **attn_cfg("tx4"),
       "ds5 FULL (tx)": ["seg_ds5+traj+tx"], "ds5 -seg (tx)": ["traj+tx"], "ds5 -traj (tx)": ["seg_ds5+tx"],
       "ds5 -tx (tx)": ["seg_ds5+traj"], "ds5 FULL (tx4)": ["seg_ds5+traj+tx4"], "ds5 -seg (tx4)": ["traj+tx4"],
       "ds5 -traj (tx4)": ["seg_ds5+tx4"],
       "safe FULL (tx)": ["seg_safe+traj+tx"],
       "pca3 @.5 (tx)": ["segp_pca3_0.1+traj+tx"], "wdim3 @.5 (tx)": ["segp_wdim3_0+traj+tx"]}

cases = cv.load_cases(list(cv.CASE_FILES))
y = cases.is_robust.astype(bool).to_numpy()
groups, models = cases.group.to_numpy(), cases.model_id.to_numpy()
short = np.array([m.split("/")[-1] for m in models])
names = sorted({b for opts in CFG.values() for o in opts for b in o.split("+")})
blocks = stack.make_blocks(cases, names, args)
fit = lambda F, tr, te: wb._predict(wb._fit_logistic(F[tr], y[tr], stack.C), F[te], True)[1]
P = {k: np.zeros((args.repeats, len(y))) for k in CFG}
picks = {k: {} for k in CFG}

for r in range(args.repeats):
    for tr, te in SGK(5, shuffle=True, random_state=r).split(y, y, groups):
        memo = {}
        def X(opt, idx):
            key = (idx.tobytes())
            return np.concatenate([memo.setdefault((b, key), blocks[b](idx)) if (b, key) not in memo else memo[(b, key)]
                                   for b in opt.split("+")], 1)
        inner = list(SGK(5, shuffle=True, random_state=1).split(tr, y[tr], groups[tr]))
        oof = {}
        def get_oof(o):
            if o not in oof:
                z = np.zeros(len(y))
                for itr, ite in inner:
                    z[tr[ite]] = fit(X(o, tr[itr]), tr[itr], tr[ite])
                oof[o] = z
            return oof[o]
        for k, opts in CFG.items():
            if len(opts) == 1:
                P[k][r, te] = fit(X(opts[0], tr), tr, te); continue
            for m in np.unique(models[te]):
                mtr, mte = tr[models[tr] == m], models[te] == m
                best = max(opts, key=lambda o: (((get_oof(o)[mtr] >= .5) == y[mtr]).mean(), o == opts[0]))
                picks[k].setdefault((m.split("/")[-1], best), 0); picks[k][(m.split("/")[-1], best)] += 1
                P[k][r, te[mte]] = fit(X(best, tr), tr, te[mte])
    print(f"repeat {r+1}/{args.repeats}", file=sys.stderr, flush=True)

def boot(a, b, n=4000, seed=0):
    """95% CI of acc(a)-acc(b), resampling problems."""
    rng = np.random.default_rng(seed)
    d = ((P[a] >= .5) == y).mean(0).astype(float) - ((P[b] >= .5) == y).mean(0)
    u, inv = np.unique(groups, return_inverse=True)
    sum_g, cnt_g = np.bincount(inv, d), np.bincount(inv)
    w = rng.multinomial(len(u), np.ones(len(u)) / len(u), n)
    s = (w * sum_g).sum(1) / (w * cnt_g).sum(1)
    return d.mean(), *np.percentile(s, [2.5, 97.5]), (s <= 0).mean()

acc = lambda k: ((P[k] >= .5) == y).mean(1)
auc = lambda k: np.mean([roc_auc_score(y, P[k][r]) for r in range(args.repeats)])
print(f"{'config':24s} {'acc':>13s} {'auc':>6s}")
for k in CFG: print(f"{k:24s} {acc(k).mean():.4f}±{acc(k).std():.3f} {auc(k):.3f}")
print("\nΔacc, paired, 95% CI bootstrap over problems, P(Δ<=0)")
pairs = []
for tx in ("tx", "tx4"):
    F = f"attn FULL ({tx})"
    pairs += [(F, f"attn {x} ({tx})") for x in ("-attn", "-seg", "-traj", "-tx")] + [(F, f"tx only ({tx})"), (f"attn -attn ({tx})", f"tx only ({tx})")]
    F = f"ds5 FULL ({tx})"
    pairs += [(F, f"ds5 -seg ({tx})"), (F, f"ds5 -traj ({tx})"), (F, f"tx only ({tx})")]
pairs += [("attn FULL (tx)", "ds5 FULL (tx)"), ("ds5 FULL (tx)", "ds5 -tx (tx)"), ("attn FULL (tx)", "pca3 @.5 (tx)")]
for a, b in pairs:
    d, lo, hi, p = boot(a, b)
    print(f"{a:20s} vs {b:20s} {d:+.4f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}")
print("\nper-model accuracy")
ms = list(np.unique(short))
print(f"{'':24s} " + " ".join(f"{m[:11]:>11s}" for m in ms))
print(f"{'n / fragile':24s} " + " ".join(f"{(short==m).sum():>6d}/{(~y[short==m]).sum():<4d}" for m in ms))
for k in CFG:
    print(f"{k:24s} " + " ".join(f"{((P[k][:, short==m] >= .5) == y[short==m]).mean():>11.3f}" for m in ms))
print("\npicks (attn FULL tx):", dict(sorted(picks["attn FULL (tx)"].items())))
if args.save:
    np.savez(args.save, y=y, short=short, groups=groups, **{k: P[k] for k in CFG})
