#!/bin/bash
# Semantic-entropy probe data in the paper's setup (Kossen et al., 2024), resumable.
# Per model, on every labelgen prompt (286 originals + their variants, cvbench prompt format):
#   1. one greedy response (vLLM, T=0, 4,096 tokens, the cases' greedy cap) -> data/cvbench/sep2/greedy/
#   2. 10 samples (T=1, top-p 0.9, top-k 50, 8,192 tokens, as labelgen)    -> data/cvbench/sep2/samples/
#   3. TBG/SLT hidden states (generate.py --stages sep2)                   -> data/cvbench/sep2/<model>.npz
# Resubmit the same command after a timeout: finished prompts are skipped.
#
#   MODELS="Skywork/Skywork-OR1-Math-7B deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" \
#     sbatch --export=ALL --partition=deeplearn --qos=gpgpudeeplearn cvbench/sep2.slurm
#SBATCH --job-name=sep2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/sep2_%j.txt
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
VENVS=/data/gpfs/projects/punim0478/bansaab/venvs
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.8.0  # vLLM 0.32 JIT-compiles kernels
export PATH=$CUDA_HOME/bin:$PATH
SAMPLE="datagen/sample.py --no-suffix"

for MODEL in ${MODELS:?set MODELS to one or more model ids}; do
  echo "=== $MODEL $(date)"
  source $VENVS/vllm-q35/bin/activate
  python $SAMPLE --model "$MODEL" --samples-dir data/cvbench/sep2/greedy \
    --base-samples 1 --variant-samples 1 --temperature 0 --max-tokens 4096
  python $SAMPLE --model "$MODEL" --samples-dir data/cvbench/sep2/samples \
    --base-samples 10 --variant-samples 10 --temperature 1.0 --top-p 0.9 --top-k 50 --max-tokens 8192
  deactivate
  source $VENVS/pyine/bin/activate
  python cvbench/generate.py --stages sep2 --models "$MODEL"
  deactivate
done
