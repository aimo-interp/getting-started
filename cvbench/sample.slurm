#!/bin/bash
# Resampling + test-time perturbation stage for the CV set (resumable).
#   sbatch cvbench/sample.slurm            # defaults: n=4, budget=2048
#SBATCH --job-name=cvsample
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=data/cvbench/logs/sample_%j.txt
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source /data/gpfs/projects/punim0478/bansaab/venvs/pyine/bin/activate
python cvbench/generate.py --stages sample --n "${N:-4}" --budget "${BUDGET:-2048}"
