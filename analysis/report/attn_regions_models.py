"""Where attention goes, per model: mean mass over layers and heads by region, whole trace and early/late."""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; DATA = ROOT / "data/cvbench"
safe = lambda m: m.replace("/", "_").replace(":", "_")
cases = [json.loads(l) for f in ("cases.jsonl", "cases_oss.jsonl") for l in open(DATA / f)]
NAMES = ["first token", "template", "problem", "reasoning >64", "last 64"]
MODELS = ["Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "Qwen/Qwen3.5-4B"]
print(f"{'':32s}" + "".join(f"{n:>15s}" for n in NAMES) + f"{'sum':>7s}")
for mid in MODELS:
    Q = []
    for c in cases:
        f = DATA / "circuit_attn" / safe(mid) / f"{c['group']}.npz"
        if c["model_id"] == mid and f.is_file():
            z = np.load(f); r = z["regions"] if "layers" in z else z["regions"][6:]
            Q.append(np.nanmean(r, (0, 1)))     # [deciles, regions]
    Q = np.stack(Q)
    for w, (a, b) in {"all": (0, 10), "d1": (0, 1), "d2-5": (1, 5), "d6-9": (5, 9), "d10": (9, 10)}.items():
        v = Q[:, a:b].mean((0, 1)); print(f"{mid.split('/')[-1][:24]:24s} {w:6s} " + "".join(f"{x:15.3f}" for x in v) + f"{v.sum():7.3f}")
    # per-layer split, whole trace
    L = []
    for c in cases:
        f = DATA / "circuit_attn" / safe(mid) / f"{c['group']}.npz"
        if c["model_id"] == mid and f.is_file():
            z = np.load(f); r = z["regions"] if "layers" in z else z["regions"][6:]
            L.append(np.nanmean(r, (1, 2)))
    L = np.nanmean(np.stack(L), 0)
    ls = z["layers"] if "layers" in z else np.arange(6, 14)
    print("   by layer (first token, template, problem, far, recent):")
    for l, v in zip(ls, L):
        if not np.isnan(v).any(): print(f"     L{l}: " + " ".join(f"{x:.3f}" for x in v))
