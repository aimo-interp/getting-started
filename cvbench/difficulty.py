#!/usr/bin/env python3
"""Difficulty direction (loop 2026-10-31): a prompt-site probe for human-rated problem difficulty.

``--build`` writes ``data/cvbench/difficulty/prompts.jsonl``: E2H-AMC problems (Easy2Hard-Bench, continuous
difficulty ratings from contestant performance) minus anything whose text matches our labelgen pool or a
cvbench case, 2,000 of them spread over the rating range. ``generate.py --stages difficulty`` then saves the
last-prompt-token, question-mean and template-mean states at every layer (chunks under ``difficulty/<model>/``).

    pyine/bin/python cvbench/difficulty.py --build
    bash cvbench/submit_tsegs.sh  (NAME=diff_small STAGES=difficulty; NAME=diff_oss CASES=cases_oss.jsonl)
    pyine/bin/python cvbench/difficulty.py            # fit + comparisons (below)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv  # noqa: E402
import whitebox as wb  # noqa: E402

DIR = cv.DATA / "difficulty"
N_PROMPTS = 2000


def _key(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", t.lower())


def build(n: int = N_PROMPTS, seed: int = 0) -> None:
    from datasets import load_dataset
    rows = []
    for split in ("train", "eval"):
        ds = load_dataset("furonghuang-lab/Easy2Hard-Bench", "E2H-AMC", split=split)
        for r in ds:
            rows.append({"id": f"{split}:{r['index']}", "text": r["problem"], "rating": float(r["rating"]),
                         "rating_std": float(r["rating_std"]), "contest": r["contest"], "year": r["year"],
                         "answer": r["answer"]})
    print(f"E2H-AMC: {len(rows)} problems")
    taken = {_key(r["problem"]) for r in cv.read_jsonl(cv.ROOT / "data" / "labelgen" / "problems.jsonl")}
    taken |= {_key(p) for p in cv.load_cases().problem}
    prefixes = {k[:60] for k in taken}
    keep = [r for r in rows if _key(r["text"]) not in taken and _key(r["text"])[:60] not in prefixes]
    seen, uniq = set(), []
    for r in keep:
        if _key(r["text"]) not in seen and r["text"].strip():
            seen.add(_key(r["text"])), uniq.append(r)
    print(f"after removing pool/case overlaps and duplicates: {len(uniq)}")
    uniq.sort(key=lambda r: r["rating"])
    idx = np.linspace(0, len(uniq) - 1, min(n, len(uniq))).round().astype(int)
    out = [uniq[i] for i in sorted(set(idx))]
    DIR.mkdir(parents=True, exist_ok=True)
    with open(DIR / "prompts.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    rat = np.array([r["rating"] for r in out])
    print(f"wrote {len(out)} prompts; rating mean {rat.mean():.0f} sd {rat.std():.0f} min {rat.min():.0f} max {rat.max():.0f}")


# ------------------------------------------------------------------------------------ fit and compare
SITES = {"h_last": "last_prompt", "h_prob": "prob_mean"}   # E2H state -> matching cvbench acts site
FRACS = (0.25, 0.35, 0.5, 0.65, 0.8, 0.9)
ALPHA = 1e3
_E2H: dict = {}


def load_e2h(model_id: str) -> dict | None:
    """Concatenated chunks of one model: {site: [n, L+1, d] float32, 'rating': [n], 'pid': [n]}; None if absent."""
    if model_id not in _E2H:
        d = DIR / cv.safe(model_id)
        files = sorted(d.glob("chunk_*.npz")) if d.is_dir() else []
        if not files:
            _E2H[model_id] = None
        else:
            zs = []
            for f in files:
                try:
                    z = np.load(f)
                    zs.append({k: z[k] for k in ("h_last", "h_prob", "rating", "pid")})
                except Exception:  # noqa: BLE001  (a chunk still being written by the GPU job)
                    print(f"[difficulty] skipping unreadable {f.name}", flush=True)
            _E2H[model_id] = ({k: np.concatenate([z[k] for z in zs]) for k in ("h_last", "h_prob", "rating", "pid")}
                              if zs else None)
    return _E2H[model_id]


def ridge_fit(H: np.ndarray, t: np.ndarray, alpha: float = ALPHA):
    """Standardised ridge H -> t; returns a predictor on raw states."""
    from sklearn.linear_model import Ridge
    H = H.astype(np.float64)
    mu, sd = H.mean(0), H.std(0) + 1e-6
    r = Ridge(alpha=alpha).fit((H - mu) / sd, t)
    return lambda X: r.predict((np.asarray(X, np.float64) - mu) / sd)


def dim_direction(H: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Mean state of the hardest third minus the easiest third, unit norm."""
    lo, hi = np.quantile(t, [1 / 3, 2 / 3])
    r = H[t >= hi].astype(np.float64).mean(0) - H[t <= lo].astype(np.float64).mean(0)
    return r / (np.linalg.norm(r) + 1e-8)


def heldout(H, t, fit, k=5, seed=0):
    """Out-of-fold Spearman of ``fit(H_tr, t_tr)(H_te)`` against t."""
    from scipy.stats import spearmanr
    from sklearn.model_selection import KFold
    oof = np.zeros(len(t))
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(t):
        oof[te] = fit(H[tr], t[tr])(H[te])
    return spearmanr(oof, t).correlation


def choose_layer(e2h: dict, site: str, fracs=FRACS) -> tuple[int, dict]:
    """Layer with the best out-of-fold ridge Spearman among ``fracs``; also the DIM Spearman per layer."""
    H, t = e2h[site], e2h["rating"]
    L = H.shape[1]
    table = {}
    for f in fracs:
        layer = wb.layer_of(L, f)
        rho_ridge = heldout(H[:, layer], t, ridge_fit)
        rho_dim = heldout(H[:, layer], t, lambda Ht, tt: (lambda X: X.astype(np.float64) @ dim_direction(Ht, tt)))
        table[layer] = (rho_ridge, rho_dim)
    best = max(table, key=lambda l: table[l][0])
    return best, table


_FITS: dict = {}


def diff_feats(cases, kinds=("diff_last", "diff_prob", "dim_last")) -> "pd.DataFrame":
    """Per case: the model's E2H difficulty probe applied to the case's own prompt state (no case labels used).
    ``diff_<site>``: ridge prediction; ``dim_<site>``: projection on the hardest-minus-easiest direction. Layer
    chosen per model by held-out Spearman on E2H. NaN for models without E2H states."""
    import pandas as pd
    acts = wb.load_acts(cases, cv)
    rows = []
    for c, a in zip(cases.itertuples(), acts):
        e2h = load_e2h(c.model_id)
        if e2h is None:
            rows.append({})
            continue
        row = {}
        for site, asite in SITES.items():
            if asite not in a:
                continue
            key = (c.model_id, site)
            if key not in _FITS:
                layer, _ = choose_layer(e2h, site)
                H, t = e2h[site][:, layer], e2h["rating"]
                _FITS[key] = (layer, ridge_fit(H, t), dim_direction(H, t))
            layer, pred, r = _FITS[key]
            tag = site.split("_")[1]
            row[f"diff_{tag}"] = float(pred(a[asite][layer][None])[0])
            row[f"dim_{tag}"] = float(a[asite][layer].astype(np.float64) @ r)
        rows.append(row)
    return pd.DataFrame(rows, index=cases.index, columns=list(kinds))


def analyse() -> None:
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    cases = cv.load_cases()
    y = cases.is_robust.astype(bool).to_numpy()
    models, groups = cases.model_id.to_numpy(), cases.group.to_numpy()
    acts = wb.load_acts(cases, cv)
    print("== held-out Spearman on E2H-AMC (ridge / DIM) per layer")
    for m in cases.model_id.unique():
        e2h = load_e2h(m)
        if e2h is None:
            print(f"  {m}: no states yet")
            continue
        for site in SITES:
            best, table = choose_layer(e2h, site)
            cells = "  ".join(f"L{l}:{r:.2f}/{d:.2f}" for l, (r, d) in table.items())
            print(f"  {m:40s} {site:7s} n={len(e2h['rating'])} best L{best}  {cells}")
    F = diff_feats(cases)
    Z = pd.DataFrame(np.asarray(cv.zscore_within_model(F, cases)), index=cases.index, columns=F.columns)
    ok = np.isfinite(F["diff_last"].to_numpy())
    print(f"\n== on the {ok.sum()} cases with E2H states ({len(set(models[ok]))} models)")
    for col in F.columns:
        z = Z[col].to_numpy()
        print(f"  {col:10s} AUC(harder -> fragile) {roc_auc_score(y[ok], -z[ok]):.3f}")
    # robustness probe score (out-of-fold diff-in-means on seg 0-5 and 5-9, as the seg block) for comparisons
    seg = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(y, groups=groups):
        s = np.mean([wb.diffmean_scores(acts, models, y, tr, None, w, 0.5)[:, 0] for w in ("seg_0_5", "seg_5_9")], 0)
        seg[te] = s[te]
    seg_z = np.asarray(cv.zscore_within_model(pd.DataFrame({"seg": seg}, index=cases.index), cases))[:, 0]
    print(f"  seg probe  AUC {roc_auc_score(y[ok], seg_z[ok]):.3f} (out of fold)")
    for col in ("diff_last", "dim_last", "diff_prob"):
        z = -Z[col].to_numpy()  # oriented like seg: higher = more robust
        r, p = pearsonr(z[ok], seg_z[ok])
        beta = np.polyfit(z[ok], seg_z[ok], 1)[0]
        resid = seg_z - beta * z
        print(f"  {col:10s} corr with seg score {r:+.3f} (p={p:.3f}); leftover AUC of seg after regressing it out "
              f"{roc_auc_score(y[ok], resid[ok]):.3f}")
    # direction cosines at the prompt site: difficulty DIM vs robustness DIM (last_prompt, same layer), random baseline
    print("\n== cosine(difficulty DIM, robustness DIM) at last_prompt, same layer; random baseline = E|cos| of random pairs")
    rng = np.random.default_rng(0)
    for m in cases.model_id.unique():
        e2h = load_e2h(m)
        if e2h is None:
            continue
        layer, _, r_diff = _FITS[(m, "h_last")]
        idx = np.where(models == m)[0]
        H = np.stack([acts[i]["last_prompt"][layer] for i in idx]).astype(np.float64)
        r_rob = H[y[idx]].mean(0) - H[~y[idx]].mean(0)
        r_rob /= np.linalg.norm(r_rob) + 1e-8
        d = len(r_rob)
        base = np.mean([abs(rng.standard_normal(d) @ rng.standard_normal(d)) / d for _ in range(50)])
        print(f"  {m:40s} L{layer} cos {r_diff @ r_rob:+.3f}  (random |cos| ~ {base:.3f}; harder-minus-easier vs robust-minus-fragile)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    if args.build:
        build()
    else:
        analyse()


if __name__ == "__main__":
    main()
