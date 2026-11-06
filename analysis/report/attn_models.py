"""Attention features by decile for every model with attention data (report's Skywork figure, extended).
Absolute levels are not comparable across models (template length differs), so read each panel within model.

    python3 analysis/report/attn_models.py <fig_dir> > analysis/report/attn_models.txt
"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import report_figstyle as rs
from sklearn.metrics import roc_auc_score as auc
ROOT = Path(__file__).resolve().parents[2]; DATA = ROOT / "data/cvbench"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "figures"
safe = lambda m: m.replace("/", "_").replace(":", "_")
cases = [json.loads(l) for f in ("cases.jsonl", "cases_oss.jsonl") for l in open(DATA / f)]
MODELS = [("Skywork/Skywork-OR1-Math-7B", "Skywork"), ("allenai/Olmo-3-7B-Think", "Olmo"),
          ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "DeepSeek"), ("Qwen/Qwen3.5-4B", "Qwen3.5")]
LATE, FAR = (5, 9), (0, 9)        # same windows as stack.attn_feats
RNG = np.random.default_rng(0)
NEUTRAL, ROB, FRAG = rs.GREY, rs.ROB, rs.FRAG
rs.apply()


def load(mid):
    rows = []
    for c in cases:
        f = DATA / "circuit_attn" / safe(mid) / f"{c['group']}.npz"
        if c["model_id"] != mid or not f.is_file():
            continue
        z = np.load(f)
        Q = z["regions"] if "layers" in z else z["regions"][6:]       # [layers, heads, deciles, regions]
        n = int(np.load(DATA / "segs" / safe(mid) / f"{c['group']}.npz")["n_resp"])
        rows.append((c["is_robust"], len(c["problem"]), n, np.nanmean(Q[..., 0] + Q[..., 1], (0, 1)), np.nanmean(Q[..., 3], (0, 1))))
    y, plen, ntok, sink, far = map(np.array, zip(*rows))
    return y.astype(bool), plen.astype(float), ntok, sink, far, sum(c["model_id"] == mid for c in cases)


def perm(v, l, n=20000):
    o = auc(l, v); r = np.argsort(np.argsort(v)) + 1; n1 = l.sum(); n0 = len(l) - n1
    a = (np.array([r[RNG.permutation(l)].sum() for _ in range(n)]) - n1 * (n1 + 1) / 2) / (n0 * n1)
    return o, ((np.abs(a - .5) >= abs(o - .5) - 1e-12).sum() + 1) / (n + 1)


def draw(ax, A, y, shade=False):
    x = np.arange(1, 11)
    if shade:
        ax.axvspan(5.5, 9.5, color=NEUTRAL, alpha=.16, lw=0)
    for cls, col, mk, name in ((True, ROB, "o", "Robust"), (False, FRAG, "^", "Fragile")):
        a = A[y == cls]
        lo, hi = np.quantile(a[RNG.integers(0, len(a), (4000, len(a)))].mean(1), [.025, .975], axis=0)
        ax.fill_between(x, lo, hi, color=col, alpha=.2, lw=0)
        ax.plot(x, a.mean(0), color=col, marker=mk)
    ax.set_xticks([1, 5, 10])


fig, axes = plt.subplots(2, 4, figsize=(5.5, 3.0), sharex=True, gridspec_kw={"wspace": .48, "hspace": .14})
for k, (mid, name) in enumerate(MODELS):
    y, plen, ntok, sink, far, total = load(mid)
    late = sink[:, LATE[0]:LATE[1]].mean(1); fr = far[:, FAR[0]:FAR[1]].mean(1); cap = ntok >= 4090
    print(f"== {name}: attention on {len(y)} of {total} cases, robust {y.sum()}, fragile {(~y).sum()}, capped {cap.sum()} (fragile {(~y[cap]).sum()})")
    a1, p1 = perm(late, y); a2, p2 = perm(-fr, y)
    print(f"late sink+template mean R {late[y].mean():.3f} F {late[~y].mean():.3f}; AUC robust {a1:.3f} p={p1:.4f}")
    print(f"far 1-9 mean R {fr[y].mean():.3f} F {fr[~y].mean():.3f}; AUC fragile-higher {a2:.3f} p={p2:.4f}")
    print("corr late sink ~ tokens %s | ~ log problem chars %.2f" % ("%.2f" % np.corrcoef(late, ntok)[0, 1] if np.ptp(ntok) else "n/a", np.corrcoef(late, np.log(plen))[0, 1]))
    if (~y[cap]).sum() and y[cap].sum():
        v = late[cap]; L = np.log(plen[cap]); res = v - np.polyval(np.polyfit(L, v, 1), L)
        print("capped: late sink AUC %.3f p=%.4f; problem chars removed AUC %.3f p=%.4f" % (perm(v, y[cap]) + perm(res, y[cap])))
    draw(axes[0, k], sink, y, shade=True); draw(axes[1, k], far, y)
    axes[0, k].set_title(f"{name} ({y.sum()}/{(~y).sum()})", fontsize=9); axes[1, k].set_xlabel("Trace decile")
axes[0, 0].set_ylabel("BOS + template share", fontsize=9); axes[1, 0].set_ylabel("Share >64 tokens back", fontsize=9)
axes[1, 0].plot([], [], color=ROB, marker="o", label="Robust"); axes[1, 0].plot([], [], color=FRAG, marker="^", label="Fragile")
fig.legend(*axes[1, 0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(.5, -.17))
rs.save(fig, OUT, "fig_attn_models")
