#!/bin/bash
# Budget-forced answers for the truncated sep2 samples (cvbench/force_answers.py), resumable.
#   one copy per free partition (not gpu-h100), e.g. MODELS="Skywork/Skywork-OR1-Math-7B" sbatch --export=ALL --partition=gpu-l40s cvbench/force.slurm
# Then: uv run cvbench/cv.py --methods sep2_probe --sep-answers forced [--sep-pos slt]
#SBATCH --job-name=force
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_out/force_%j.txt
set -eo pipefail
scancel --name="$SLURM_JOB_NAME" --user="$USER" --state=PENDING || true  # first copy to start cancels its siblings
cd "$SLURM_SUBMIT_DIR"
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.8.0  # vLLM 0.32 JIT-compiles kernels
export PATH=$CUDA_HOME/bin:$PATH
source /data/gpfs/projects/punim0478/bansaab/venvs/vllm-q35/bin/activate
for MODEL in ${MODELS:?set MODELS to one or more model ids}; do
  echo "=== $MODEL $(date)"
  python cvbench/force_answers.py --model "$MODEL"
done
