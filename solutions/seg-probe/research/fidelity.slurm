#!/bin/bash
# GPU fidelity check of the shipped seg-probe inference against the cached cvbench features.
#   sbatch --partition=gpu-a100-short solutions/seg-probe/research/fidelity.slurm
#   MODELS=openai/gpt-oss-120b:medium sbatch --export=ALL --partition=feit-gpu-a100 --qos=feit solutions/seg-probe/research/fidelity.slurm
# One copy per partition is fine: the first to start cancels its pending siblings.
#SBATCH --job-name=segfid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_out/fidelity_%j.txt
set -eo pipefail
scancel --name="$SLURM_JOB_NAME" --user="$USER" --state=PENDING || true
cd "$SLURM_SUBMIT_DIR"
source /data/gpfs/projects/punim0478/bansaab/venvs/pyine/bin/activate
export HF_HOME=/data/gpfs/projects/punim0478/bansaab  # online: gpt-oss's MXFP4 kernel resolves its version tag
nvidia-smi --query-gpu=name,memory.total --format=csv
for MODEL in ${MODELS:-Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B}; do
  echo "=== $MODEL $(date)"
  python solutions/seg-probe/research/check_fidelity.py --model "$MODEL"
done
