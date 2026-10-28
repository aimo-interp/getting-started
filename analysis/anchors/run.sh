#!/bin/bash
# Full anchor-analysis pipeline on Qwen3.5-4B. Every stage resumes, so rerun the
# same command after an interruption. Run from anywhere on a GPU node:
#
#   bash analysis/anchors/run.sh                 # everything
#   STAGES="whitebox" bash analysis/anchors/run.sh
#   N=10 MAX_POINTS=40 BUDGET=16384 bash analysis/anchors/run.sh
#
# Outputs: analysis/anchors/work/ (summary.json, wb_*.json, logs/).
set -eo pipefail
cd "$(dirname "$0")/../.."

export HF_HOME=/data/gpfs/projects/punim0478/bansaab HF_HUB_OFFLINE=1
VLLM_PY=/data/gpfs/projects/punim0478/bansaab/venvs/vllm-q35/bin/python
STAGES="${STAGES:-prepare screen curves whitebox}"
N="${N:-10}"                    # resamples per prefix (analyses 1, 2) and per steering condition
SCREEN_N="${SCREEN_N:-8}"       # samples per X / dX prompt when screening
MAX_POINTS="${MAX_POINTS:-40}"  # prefix points per curve
BUDGET="${BUDGET:-16384}"       # completion tokens, counted from <think>
LOGS=analysis/anchors/work/logs
mkdir -p "$LOGS"

has() { [[ " $STAGES " == *" $1 "* ]]; }

if has prepare; then
  uv run analysis/anchors/prepare.py 2>&1 | tee "$LOGS/prepare.txt"
fi
if has screen; then
  # PICK="--spurious G --robust G --probe-fail G" overrides the automatic choice
  $VLLM_PY analysis/anchors/resample.py --stages screen select --screen-n "$SCREEN_N" --budget "$BUDGET" $PICK \
    2>&1 | tee "$LOGS/screen.txt"
fi
if has curves; then
  $VLLM_PY analysis/anchors/resample.py --stages anchors transplant summary --n "$N" --max-points "$MAX_POINTS" \
    --budget "$BUDGET" 2>&1 | tee "$LOGS/curves.txt"
fi
if has whitebox; then
  uv run analysis/anchors/whitebox.py --n "$N" --budget "$BUDGET" ${WB_ARGS} 2>&1 | tee "$LOGS/whitebox.txt"
fi
