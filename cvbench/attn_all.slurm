#!/bin/bash
# Submit copies to several partitions; the first to start cancels its pending siblings.
#SBATCH --job-name=attnds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:08:00
#SBATCH --output=slurm_out/attnds_%j.txt
set -eo pipefail
scancel --name="$SLURM_JOB_NAME" --user="$USER" --state=PENDING || true
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
export HF_HOME=/data/gpfs/projects/punim0478/bansaab
source /data/gpfs/projects/punim0478/bansaab/venvs/pyine/bin/activate
python cvbench/attn_collect.py --models ${MODELS:-deepseek-ai/DeepSeek-R1-0528-Qwen3-8B}
