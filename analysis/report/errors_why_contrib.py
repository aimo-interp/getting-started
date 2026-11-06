"""Why the stack decided: logit contribution of each block per case. Uses each model's final option (stack-attn artifact),
same 20 x 5 folds. contribution = coef * feature, grouped by block; intercept reported separately.

    python analysis/report/errors_why_contrib.py   # -> errors_why_contrib.csv
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold as SGK
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "cvbench"))
import cv, stack, whitebox as wb
args = argparse.Namespace(crossfit=False, sample=0, frac=.5)
OPT = {"Qwen/Qwen3.5-4B": "seg_safe+traj+tx", "Skywork/Skywork-OR1-Math-7B": "seg_safe+traj+tx", "allenai/Olmo-3-7B-Think": "seg_safe+traj+tx+attn",
       "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "traj+tx+attn", "openai/gpt-oss-120b:high": "seg_safe+traj+tx", "openai/gpt-oss-120b:low": "traj+tx", "openai/gpt-oss-120b:medium": "seg_safe+traj+tx"}
cases = cv.load_cases(list(cv.CASE_FILES)); y = cases.is_robust.astype(bool).to_numpy(); groups = cases.group.to_numpy(); models = cases.model_id.to_numpy()
blocks = stack.make_blocks(cases, ["seg_safe", "traj", "tx", "attn"], args)
BL = ["seg_safe", "traj", "tx", "attn"]; R = 20
C = np.zeros((R, len(y), len(BL) + 1)); P = np.zeros((R, len(y)))
for r in range(R):
    for tr, te in SGK(5, shuffle=True, random_state=r).split(y, y, groups):
        F = {b: blocks[b](tr) for b in BL}
        for o in set(OPT.values()):
            bs = o.split("+"); X = np.concatenate([F[b] for b in bs], 1); clf = wb._fit_logistic(X[tr], y[tr], stack.C)
            idx = te[[OPT[models[i]] == o for i in te]]
            if not len(idx): continue
            P[r, idx] = clf.predict_proba(X[idx])[:, 1]; C[r, idx, -1] = clf.intercept_[0]; j = 0
            for b in bs:
                k = F[b].shape[1]; C[r, idx, BL.index(b)] = X[idx][:, j:j + k] @ clf.coef_[0][j:j + k]; j += k
    print("repeat", r + 1, file=sys.stderr, flush=True)
d = pd.DataFrame(C.mean(0), index=cases.index, columns=["seg", "traj", "tx", "attn", "intercept"]); d["p"] = P.mean(0); d["err"] = ((P >= .5) != y).mean(0)
d.to_csv(HERE / "errors_why_contrib.csv")
