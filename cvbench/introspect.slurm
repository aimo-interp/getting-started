#!/bin/bash
# Zero-shot self-prediction baseline (Looking Inward) on the cvbench cases: one forward pass per case and
# question, no generation (cvbench/generate.py --stages introspect). Resumable.
#   sbatch --export=ALL --partition=feit-gpu-a100 --qos=feit cvbench/introspect.slurm
#SBATCH --job-name=introspect
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_out/introspect_%j.txt
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source /data/gpfs/projects/punim0478/bansaab/venvs/pyine/bin/activate
python cvbench/generate.py --stages introspect
