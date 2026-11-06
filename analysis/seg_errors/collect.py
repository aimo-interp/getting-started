"""Per-case out-of-fold predictions for the seg probe (and the stack) over 20x5 grouped CV, joined with case
metadata, for error analysis. Run from cvbench/: python ../analysis/seg_errors/collect.py"""
import sys, argparse, numpy as np, pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold
import cv, stack, whitebox as wb

REPEATS = 20
cases = cv.load_cases(list(cv.CASE_FILES))
y = cases.is_robust.astype(bool).to_numpy()
models, groups = cases.model_id.to_numpy(), cases.group.to_numpy()
args = argparse.Namespace(frac=0.5, crossfit=False, sample=0)
acts = wb.load_acts(cases, cv)

def oof(name, feat_fn, record_gate=False):
    P = np.zeros((REPEATS, len(y))); G = np.zeros((REPEATS, len(y), 2)); R = np.zeros((REPEATS, len(y), 2))
    for r in range(REPEATS):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(y, y, groups):
            F = feat_fn(tr)
            P[r, te] = wb._predict(wb._fit_logistic(F[tr], y[tr], stack.C), F[te], True)[1]
            if record_gate:
                G[r, te] = F[te] != 0      # window kept by the gate
                R[r, te] = np.concatenate([wb.diffmean_scores(acts, models, y, tr, None, s, 0.5)
                                           for s in ("seg_0_5", "seg_5_9")], 1)[te]   # ungated DIM score
    print(name, "acc", ((P >= .5) == y).mean().round(4), file=sys.stderr)
    return P, G, R

B = stack.make_blocks(cases, ["seg_safe", "seg", "traj", "tx"], args)
P_seg, G, R = oof("seg_safe", B["seg_safe"], record_gate=True)
P_raw, *_ = oof("seg", B["seg"])
P_tt, *_ = oof("traj+tx", lambda tr: np.c_[B["traj"](tr), B["tx"](tr)])
P_st, *_ = oof("stack", lambda tr: np.c_[B["seg_safe"](tr), B["traj"](tr), B["tx"](tr)])

out = cases[["model_id", "group", "source", "is_robust", "max_drop"]].copy()
for k, P in [("seg", P_seg), ("segraw", P_raw), ("tt", P_tt), ("stack", P_st)]:
    out[f"p_{k}"] = P.mean(0)
    out[f"err_{k}"] = ((P >= .5) != y).mean(0)
out["gate_w05"], out["gate_w59"] = G[..., 0].mean(0), G[..., 1].mean(0)
out["dim_w05"], out["dim_w59"] = R[..., 0].mean(0), R[..., 1].mean(0)   # OOF ungated DIM score (z within model)
g = cv.greedy_table(cases)
out["greedy_tokens"] = g["generation_num_tokens"].values
out["greedy_logprob"] = g["generation_mean_logprob"].values
tx = stack.tx_feats(cases)
out[["agree", "anchor", "tx_length"]] = tx[["agree", "anchor", "length"]].values
try:
    import difficulty
    out["diff_last"] = difficulty.diff_feats(cases, ("diff_last",))["diff_last"].values
except Exception as e:
    print("no difficulty:", e, file=sys.stderr)
out["problem_chars"] = cases.problem.str.len()
out["problem"] = cases.problem
out.to_csv("../analysis/seg_errors/oof_cases.csv")
