#!/bin/bash
# logit lens + perturbation sensitivity (cvbench/sens_collect.py); first copy to start cancels pending siblings
#SBATCH --job-name=sens
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_out/sens_%j.txt
set -eo pipefail
scancel --name="$SLURM_JOB_NAME" --user="$USER" --state=PENDING || true
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
export HF_HOME=/data/gpfs/projects/punim0478/bansaab
source /data/gpfs/projects/punim0478/bansaab/venvs/pyine/bin/activate
python cvbench/sens_collect.py --models ${MODELS:?}
