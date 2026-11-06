#!/bin/bash
# seg-probe steering on Skywork (cvbench/steer.py), 4 shards. Pass the partition: sbatch -p gpu-l40s cvbench/steer.slurm
#SBATCH --job-name=steer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm_out/steer_%A_%a.txt
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
export HF_HOME=/data/gpfs/projects/punim0478/bansaab
source /data/gpfs/projects/punim0478/bansaab/venvs/pyine/bin/activate
python cvbench/steer.py --alphas ${ALPHA:-1} --shard $SLURM_ARRAY_TASK_ID/4
