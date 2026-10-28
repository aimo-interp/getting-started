#!/bin/bash
# Train the shared probe (hidden states + short-resample behaviour score) on
# aimo-interp/train-main-v2 plus any generated labels, then build the ZIPs.
# Submit from the repo root:
#
#   sbatch solutions/diffmean-probe/scripts/run_probe.slurm
#
# Steps:
#   1. If data/labelgen/samples/ has samples, turn them into data/labelgen/labels.jsonl.
#   2. Each small model reads every row: hidden states + behaviour features.
#   3. Grouped CV picks site, depth, beta, gamma, threshold; --nested reports
#      an out-of-sample estimate. The selected probe is exported to artifacts/.
#   4. Build dist/solution-diffmean-probe.zip and the -small variant.
#
# gpt-oss-120b is not extracted here (it needs both GPUs and CPU offload), so it
# is scored by its prior. Log: slurm_out/probe_<jobid>.txt
#SBATCH --job-name=probe
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_out/probe_%j.txt
set -eo pipefail

PYTHON="${PYTHON:-/data/gpfs/projects/punim0478/bansaab/venvs/pyine/bin/python}"
MODELS="${MODELS:-Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B allenai/Olmo-3-7B-Think deepseek-ai/DeepSeek-R1-0528-Qwen3-8B}"
FEATURES="${FEATURES:-data/diffmean-probe/features-${SLURM_JOB_ID:-local}}"   # fresh dir: rows may have changed

cd "${SLURM_SUBMIT_DIR:-.}"

EXTRA=()
if ls data/labelgen/samples/*.jsonl >/dev/null 2>&1; then
  "$PYTHON" datagen/label.py
  EXTRA=(--extra-rows data/labelgen/labels.jsonl)
fi

# shellcheck disable=SC2086  # MODELS is a space-separated list
"$PYTHON" solutions/diffmean-probe/scripts/extract_features.py "${EXTRA[@]}" \
  --output-dir "$FEATURES" --models $MODELS
"$PYTHON" solutions/diffmean-probe/scripts/train_diffmean.py --features-dir "$FEATURES" --nested
"$PYTHON" scripts/build.py solutions
"$PYTHON" scripts/build.py solutions --small
