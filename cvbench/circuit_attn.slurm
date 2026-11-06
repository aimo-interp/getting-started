#!/bin/bash
# Component attribution of the Skywork seg-probe direction (cvbench/circuit_attr.py collect, then analyze).
# Submit copies to several partitions; the first to start cancels its pending siblings.
#SBATCH --job-name=circattn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_out/circattn_%j.txt
set -eo pipefail
scancel --name="$SLURM_JOB_NAME" --user="$USER" --state=PENDING || true
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
export HF_HOME=/data/gpfs/projects/punim0478/bansaab
source /data/gpfs/projects/punim0478/bansaab/venvs/pyine/bin/activate
python cvbench/circuit_attr.py attn
python cvbench/circuit_attr.py attn_analyze | tee analysis/circuit/attn_skywork.txt
