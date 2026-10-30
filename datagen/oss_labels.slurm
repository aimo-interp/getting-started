#!/bin/bash
# gpt-oss-120b robustness labels through OpenRouter (datagen/balanced_labels.py), one effort per job, CPU only.
#   EFFORT=low PER_CLASS=50 sbatch --export=ALL --job-name=oss_labels_low datagen/oss_labels.slurm
# Resumable; labels -> data/labelgen/balanced_<effort>/labels.jsonl. Spend is capped by datagen/oss_budget.slurm.
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_out/oss_labels_%x_%j.txt
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
source /data/gpfs/projects/punim0478/bansaab/venvs/pyine/bin/activate
python datagen/balanced_labels.py --models openai/gpt-oss-120b --effort "${EFFORT:?}" --per-class "${PER_CLASS:-50}" \
  --out-dir "data/labelgen/balanced_${EFFORT}" --max-tokens 32768 --workers "${WORKERS:-16}"
