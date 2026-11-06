#!/bin/bash
# Smoke test of stack-attn: (1) attention_signals on stored greedy traces vs the training features,
# (2) one full are_robust call on a Skywork case.
#SBATCH --job-name=smoke_attn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_out/smoke_attn_%j.txt
scancel --name="$SLURM_JOB_NAME" --user="$USER" --state=PENDING || true
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/data/gpfs/projects/punim0478/bansaab
source /data/gpfs/projects/punim0478/bansaab/venvs/pyine/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
cd solutions/stack-attn && python - <<'PY'
import json, sys, time
sys.path.insert(0, "../../cvbench")
import cv, stack
from stack_probe import engine
from uncertainty_profile.extraction import load_model_and_tokenizer
art = json.load(open("artifacts/stack.json"))
M = "Skywork/Skywork-OR1-Math-7B"
c = cv.load_cases(list(cv.CASE_FILES)); s = c[c.model_id == M]
A = stack.attn_feats(s)
greedy = {r["group"]: r for r in cv.read_jsonl(cv.DATA / "greedy" / f"{cv.safe(M)}.jsonl")}
model, tok = load_model_and_tokenizer(M, local_files_only=True)
print("attn impl:", getattr(model.config, "_attn_implementation", None), flush=True)
for g in ["1c2bec482bb1", "43a47a368257", "3f2e57075a10"]:
    i = s.index[s.group == g][0]
    t0 = time.time()
    raw = engine.attention_signals(model, tok, s.loc[i, "problem"], greedy[g]["generated_text"], art["attention"])
    print("CHECK", g, "engine", {k: round(v, 5) for k, v in raw.items()}, "train", A.loc[i].round(5).to_dict(),
          f"{time.time() - t0:.1f}s", flush=True)
print("attn impl after:", getattr(model.config, "_attn_implementation", None), flush=True)
del model; engine.release()
case = s[s.group == "43a47a368257"].iloc[0]
t0 = time.time()
print("FULL label", case.is_robust, "->", engine.are_robust(M, "default", [case.problem]), f"{time.time() - t0:.0f}s", flush=True)
PY
