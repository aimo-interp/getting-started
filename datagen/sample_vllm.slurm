#!/bin/bash
# Sample one model with vLLM on reduced settings (first 100 problems,
# 4 samples per original, 2 per variant, up to 8k tokens). Submit from the repo root:
#
#   sbatch datagen/sample_vllm.slurm deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
#   PROBLEMS=50 sbatch datagen/sample_vllm.slurm Qwen/Qwen3.5-4B
#
# Resubmit the same command after a timeout: finished prompts are skipped.
# Output: data/labelgen/samples/<model>__default.jsonl; log: slurm_out/sample_vllm_<jobid>.txt
#SBATCH --job-name=sample
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_out/sample_vllm_%j.txt
set -eo pipefail

SKYWORK=Skywork/Skywork-OR1-Math-7B
DEEPSEEK=deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
OLMO=allenai/Olmo-3-7B-Think

PROBLEMS="${PROBLEMS:-100}"
BASE="${BASE:-4}"              # samples per original problem
VARIANT="${VARIANT:-2}"        # samples per variant
MAX_TOKENS="${MAX_TOKENS:-8192}"

cd "$SLURM_SUBMIT_DIR"
# python datagen/sample.py --model "$SKYWORK" --limit-problems "$PROBLEMS" \
  # --base-samples "$BASE" --variant-samples "$VARIANT" --max-tokens "$MAX_TOKENS"

# python datagen/sample.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --work-dir data/diffmean-probe/direction_pilot/work --base-samples 4 --variant-samples 4 --max-tokens 32768 --chunk-size 16
source /data/gpfs/projects/punim0478/bansaab/venvs/pyine/bin/activate
python solutions/perturb-probe/research/extract_features.py --models deepseek-ai/DeepSeek-R1-0528-Qwen3-8B Skywork/Skywork-OR1-Math-7B allenai/Olmo-3-7B-Think Qwen/Qwen3.5-4B