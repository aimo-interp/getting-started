"""Single-feature AUCs (robust = positive) for the sampling-stage features."""
import sys
from pathlib import Path
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv

cases = cv.load_cases()
args = argparse.Namespace(n=4, budget=2048)
s = cv.sample_table(cases, args)
have = s["group"].notna().to_numpy()
cases, s = cases[have], s[have]
y = cases.is_robust.astype(bool).to_numpy()
print(f"{len(cases)} cases with samples; finish_x mean={s.finish_x.mean():.2f}; no forced answer={s.reference.isna().mean():.2f}")
for col in ["x_consistency", "agree_mean", "agree_min", "lp_x", "dlogp", "dlogp_max", "finish_x", "tokens_x"]:
    v = s[col].astype(float).fillna(s[col].astype(float).median()).to_numpy()
    print(f"  {col:14s} AUC={roc_auc_score(y, v):.3f}  mean robust={v[y].mean():.3f} not={v[~y].mean():.3f}")
