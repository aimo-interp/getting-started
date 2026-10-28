#!/bin/bash
# Length-probe training data for one model (cvbench len_probe), resumable:
#   1. unless SAMPLE=0: 4 samples of each labelgen original (vLLM, T=1, top-p 0.9, top-k 50, 8,192 tokens,
#      cvbench prompt format)                                    -> data/cvbench/len/samples/
#   2. the last-prompt hidden states of those originals         -> data/cvbench/len/<model>.npz
#   MODEL=Qwen/Qwen3.5-4B sbatch --export=ALL --partition=deeplearn --qos=gpgpudeeplearn cvbench/lens.slurm
#   MODEL=allenai/Olmo-3-7B-Think SAMPLE=0 sbatch ...   (its sep2 samples already cover the originals)
#SBATCH --job-name=lens
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_out/lens_%j.txt
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
VENVS=/data/gpfs/projects/punim0478/bansaab/venvs
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.8.0  # vLLM 0.32 JIT-compiles kernels
export PATH=$CUDA_HOME/bin:$PATH
echo "=== ${MODEL:?set MODEL} $(date)"
if [ "${SAMPLE:-1}" != 0 ]; then
  source $VENVS/vllm-q35/bin/activate
  python datagen/sample.py --no-suffix --model "$MODEL" --samples-dir data/cvbench/len/samples \
    --base-samples 4 --variant-samples 0 --temperature 1.0 --top-p 0.9 --top-k 50 --max-tokens 8192
  deactivate
fi
source $VENVS/pyine/bin/activate
python cvbench/generate.py --stages lens --models "$MODEL"
