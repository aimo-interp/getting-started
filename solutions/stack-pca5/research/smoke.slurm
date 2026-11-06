#!/bin/bash
# Smoke test of the stack-probe engine on a GPU: two Qwen cases from cvbench, prints predictions and the log.
#   sbatch --partition=gpu-a100-short --qos=normal solutions/stack-probe/research/smoke.slurm
#SBATCH --job-name=smoke_stack
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_out/smoke_stack_%j.txt
scancel --name="$SLURM_JOB_NAME" --user="$USER" --state=PENDING || true
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/data/gpfs/projects/punim0478/bansaab
source /data/gpfs/projects/punim0478/bansaab/venvs/pyine/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
cd solutions/stack-probe && python - <<'PY'
import json, time, sys
cases = [json.loads(l) for l in open("../../data/cvbench/cases.jsonl")]
qwen = [c for c in cases if c["model_id"] == "Qwen/Qwen3.5-4B"][:3]
from solution import are_robust
t0 = time.time()
for c in qwen[:2]:
    print("CASE label", c["is_robust"], "->", are_robust(c["model_id"], "default", [c["problem"]]), f"{time.time()-t0:.0f}s", flush=True)
print("two problems in one call:", are_robust("Qwen/Qwen3.5-4B", "default", [c["problem"] for c in qwen[1:3]]), flush=True)
PY
