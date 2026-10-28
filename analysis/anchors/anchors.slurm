#!/bin/bash
# Batch version of run.sh. Submit one copy per partition with submit.sh; the first
# copy to start cancels its still-pending siblings (same job name).
#SBATCH --job-name=anchors
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=analysis/anchors/work/logs/slurm_%j.txt
set -eo pipefail
scancel --name="$SLURM_JOB_NAME" --user="$USER" --state=PENDING || true
cd "$SLURM_SUBMIT_DIR"
bash analysis/anchors/run.sh
