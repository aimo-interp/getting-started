#!/bin/bash
# Resampling + test-time perturbation stage for the CV set (resumable).
# Submit one copy per partition with cvbench/submit_sample.sh; the first copy
# to start cancels its still-pending siblings (same job name).
#SBATCH --job-name=cvsample
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=data/cvbench/logs/sample_%j.txt
set -eo pipefail
scancel --name="$SLURM_JOB_NAME" --user="$USER" --state=PENDING || true
cd "$SLURM_SUBMIT_DIR"
source /data/gpfs/projects/punim0478/bansaab/venvs/pyine/bin/activate
python cvbench/generate.py --stages sample --n "${N:-4}" --budget "${BUDGET:-2048}" ${MODELS:+--models $MODELS}
