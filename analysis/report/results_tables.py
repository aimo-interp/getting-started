"""Numbers quoted in the report's Results sections that are not printed by ablation_candidates.py.
Uses the saved out-of-fold predictions (ablation_candidates.npz); nothing is refitted.

    python analysis/report/results_tables.py > analysis/report/results_tables.txt
"""
import ast, re, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binom, fisher_exact, mannwhitneyu
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cvbench"))
import cv, traj

d = np.load(ROOT / "analysis/report/ablation_candidates.npz", allow_pickle=True)
y, short, groups = d["y"], d["short"].astype(str), d["groups"]
cases = cv.load_cases(list(cv.CASE_FILES))
assert (cases.is_robust.astype(bool).to_numpy() == y).all()
ALL, SMALL = np.ones(len(y), bool), ~np.char.startswith(short, "gpt-oss")
FULL, TX, WB = "attn FULL (tx)", "tx only (tx)", "attn -tx (tx)"
FULL4, TX4 = "attn FULL (tx4)", "tx only (tx4)"

eng = zipfile.ZipFile(ROOT / "dist/solution-stack-attn-pmt.zip").read("stack_probe/engine.py").decode()
thr_map = ast.literal_eval(re.search(r"^THRESHOLDS = (\{.*?\})", eng, re.S | re.M).group(1))
DEFAULT = float(re.search(r"^DEFAULT_THRESHOLD = ([\d.]+)", eng, re.M).group(1))
print("thresholds shipped in stack-attn-pmt:", thr_map, "default", DEFAULT)
thr_map = {k.split("/")[-1]: v for k, v in thr_map.items()}
THR = np.array([thr_map[s] for s in short])
print("threshold per configuration:", {s: float(THR[short == s][0]) for s in sorted(set(short))})
assert len(set(THR)) > 1

def correct(k, t=.5):
    return ((d[k] >= t) == y).mean(0).astype(float)
def boot(a, b, mask=ALL, n=4000, seed=0, ta=.5, tb=.5):
    rng = np.random.default_rng(seed)
    x = (correct(a, ta) - correct(b, tb))[mask]
    u, inv = np.unique(groups[mask], return_inverse=True)
    sg, cg = np.bincount(inv, x), np.bincount(inv)
    w = rng.multinomial(len(u), np.ones(len(u)) / len(u), n)
    s = (w * sg).sum(1) / (w * cg).sum(1)
    return x.mean(), *np.percentile(s, [2.5, 97.5])
fmt = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"

print("\n== accuracy at .5 and paired gains, bootstrap over problems")
print("always-robust main %.3f small %.3f (n small %d)" % (y.mean(), y[SMALL].mean(), SMALL.sum()))
for name, m in (("main", ALL), ("small", SMALL)):
    for f, t in ((FULL, TX), (FULL4, TX4)):
        print(f"{name:5s} {f:16s} {correct(f)[m].mean():.3f}  {t:14s} {correct(t)[m].mean():.3f}  white-box {correct(WB)[m].mean():.3f}"
              f"  full-tx {fmt(boot(f, t, m))}  wb-tx {fmt(boot(WB, t, m))}  full-wb {fmt(boot(f, WB, m))}")
print("\n== with the per-model thresholds shipped in the pmt zip (chosen on these labels: optimistic)")
for name, m in (("main", ALL), ("small", SMALL)):
    for f, t in ((FULL, TX), (FULL4, TX4)):
        print(f"{name:5s} {f:16s} {correct(f, THR)[m].mean():.3f} vs tx@.5 {correct(t)[m].mean():.3f}"
              f"  thr-vs-thr tx {correct(t, THR)[m].mean():.3f}  gain {fmt(boot(f, t, m, ta=THR, tb=THR))}")

print("\n== per configuration")
print(f"{'model':30s} n frag always  full  tx   wb | full4 tx4")
for s in sorted(set(short)):
    k = short == s
    print(f"{s:30s} {k.sum():3d} {(~y[k]).sum():3d} {y[k].mean():.3f} {correct(FULL)[k].mean():.3f} {correct(TX)[k].mean():.3f}"
          f" {correct(WB)[k].mean():.3f} | {correct(FULL4)[k].mean():.3f} {correct(TX4)[k].mean():.3f}")

print("\n== errors, wrong in >= half of repeats")
ann = pd.read_csv(ROOT / "analysis/seg_errors/cases_annotated.csv").set_index("case_id").reindex(cases.index if "case_id" not in cases else cases.case_id)
assert ann.is_robust.notna().all()
wrong = {k: correct(k) <= .5 for k in (FULL, TX, WB, FULL4, TX4)}
for k, w in wrong.items():
    print(f"{k:18s} missed fragile {(w & ~y).sum()}/{(~y).sum()}  false alarms {(w & y).sum()}/{y.sum()}")

print("\n== competence: labelgen cases with per-type data, base_acc = solved share of 4 samples on the original")
lg = ann.base_acc.notna().to_numpy()
print("labelgen cases", lg.sum(), "fragile", (~y[lg]).sum())
for lo, hi, name in ((1, 1.01, "4/4"), (0, 1, "<4/4")):
    k = lg & (ann.base_acc.to_numpy() >= lo) & (ann.base_acc.to_numpy() < hi)
    fr = k & ~y
    print(f"solved {name}: n={k.sum()} fragile={fr.sum()} share fragile {fr.sum()/k.sum():.2f} | full caught {(~wrong[FULL][fr]).sum()}/{fr.sum()}"
          f" mean p {d[FULL].mean(0)[fr].mean():.2f} | tx caught {(~wrong[TX][fr]).sum()}/{fr.sum()} | white-box caught {(~wrong[WB][fr]).sum()}/{fr.sum()}")
a, b = lg & (ann.base_acc.to_numpy() >= 1) & ~y, lg & (ann.base_acc.to_numpy() < 1) & ~y
tab = [[(~wrong[FULL][a]).sum(), wrong[FULL][a].sum()], [(~wrong[FULL][b]).sum(), wrong[FULL][b].sum()]]
print("Fisher fragile share, 4/4 vs <4/4: 5/25 vs 9/13, p=%.3f" % fisher_exact([[5, 20], [9, 4]])[1])
print("Fisher caught vs solved-4/4, fragile only:", tab, "p=%.3f" % fisher_exact(tab)[1])
k44 = lg & (ann.base_acc.to_numpy() >= 1)
kless = lg & (ann.base_acc.to_numpy() < 1)
print("\nAUC for robust (OOF prob averaged over repeats); labelgen cases by base accuracy")
cols = {"stack": FULL, "tx": TX, "white-box": WB}
extra = {"p_seg": ann.p_seg.to_numpy()}
for name, k in (("solved 4/4", k44), ("solved <4/4", kless), ("all labelgen", lg)):
    r = {c: roc_auc_score(y[k], d[v].mean(0)[k]) for c, v in cols.items()}
    r["seg"] = roc_auc_score(y[k], extra["p_seg"][k])
    print(name, f"n={k.sum()} fragile={(~y[k]).sum()}", {c: round(v, 3) for c, v in r.items()})
# permutation null for the solved-4/4 AUC
rng = np.random.default_rng(0); yy = y[k44]; pp = d[FULL].mean(0)[k44]
obs = roc_auc_score(yy, pp); nul = [roc_auc_score(rng.permutation(yy), pp) for _ in range(5000)]
print("solved-4/4 stack AUC %.3f, null 90%% band [%.2f, %.2f], P(null >= obs) %.3f" % (obs, *np.percentile(nul, [5, 95]), np.mean(np.array(nul) >= obs)))
fr = ~y
print("missed vs caught fragile (full stack), MWU p:")
for c in ("max_drop", "agree", "anchor", "tx_length", "diff_last"):
    x = ann[c].to_numpy(); print(f"  {c:10s} missed med {np.nanmedian(x[fr & wrong[FULL]]):.2f} caught med {np.nanmedian(x[fr & ~wrong[FULL]]):.2f} p={mannwhitneyu(x[fr & wrong[FULL]], x[fr & ~wrong[FULL]]).pvalue:.3f}")

print("\n== trajectory direction (AUC for robust, higher feature = robust), and control for greedy length")
F = pd.DataFrame([traj.traj_features(s) for s in traj.load_segs(cases)], index=cases.index)[traj.FEATS]
Z = F.groupby(short).transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
gt = ann.greedy_tokens.to_numpy(); capped = gt >= 4090
print("capped traces:", capped.sum(), "of", len(y), "| robust share capped %.2f uncapped %.2f" % (y[capped].mean(), y[~capped].mean()))
print("capped share by model:", {s: round(capped[short == s].mean(), 2) for s in sorted(set(short))})
for c in F:
    res = {s: round(roc_auc_score(y[short == s], F[c].to_numpy()[short == s]), 2) for s in sorted(set(short))}
    print(c, "pooled %.3f" % roc_auc_score(y, Z[c]), "| capped only %.3f" % roc_auc_score(y[capped], Z[c].to_numpy()[capped]), res)
sky = short == "Skywork-OR1-Math-7B"
print("Skywork: tokens AUC %.3f; capped n=%d frag=%d" % (roc_auc_score(y[sky], gt[sky]), capped[sky].sum(), (~y[sky & capped]).sum()))
if capped[sky].sum() and (~y[sky & capped]).sum():
    for c in F: print("  Skywork capped", c, "%.3f" % roc_auc_score(y[sky & capped], F[c].to_numpy()[sky & capped]))

print("\n== leaderboard: exact binomial, P(score <= observed | true accuracy = CV at 4k)")
for name, k, n, p in (("stack-attn main 1027309", 14, 21, .804), ("stack-attn main 1027141", 12, 21, .804), ("stack-attn small", 8, 14, .777),
                      ("transcript main", 15, 21, .741), ("transcript small", 11, 14, .689)):
    print(f"{name:26s} {k}/{n}={k/n:.2f} CV {p}  P={binom.cdf(k, n, p):.3f}")
def wilson(k, n, z=1.96):
    p, c = k / n, z * z / n
    mid, half = (p + c / 2) / (1 + c), z * np.sqrt(p * (1 - p) / n + c / (4 * n)) / (1 + c)
    return mid - half, mid + half
for k, n in ((14, 21), (12, 21), (15, 21), (8, 14), (11, 14)):
    print(f"{k}/{n} Wilson 95%% [%.2f, %.2f]" % wilson(k, n))
