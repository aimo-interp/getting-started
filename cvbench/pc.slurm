#!/bin/bash
# Competence probe data (pcorrect.py) for one model[:effort]: last-prompt states on every labelgen original
# (generate.py --stages pcprompt), then N samples per original (T=1, top-p .9, top-k 50, 8192 tokens, as sep2)
# -> data/cvbench/sep2/samples/<model>__<effort>.jsonl. Resumable; first copy to start cancels pending siblings.
#   MODEL=openai/gpt-oss-120b EFFORT=low N=6 sbatch -J pc_oss_low --export=ALL -p gpu-a100 cvbench/pc.slurm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_out/pc_%x_%j.txt
set -eo pipefail
if [ "${NEED80:-0}" = 1 ]; then
  mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  if [ "$mib" -lt 75000 ]; then echo "GPU has $mib MiB; need an 80 GB card"; exit 1; fi
fi
scancel --name="$SLURM_JOB_NAME" --user="$USER" --state=PENDING || true
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/data/gpfs/projects/punim0478/bansaab
VENVS=/data/gpfs/projects/punim0478/bansaab/venvs
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.8.0
export PATH=$CUDA_HOME/bin:$PATH
nvidia-smi --query-gpu=name,memory.total --format=csv
MID="$MODEL${EFFORT:+:$EFFORT}"
CASES=cases.jsonl; [[ "$MODEL" == *gpt-oss* ]] && CASES=cases_oss.jsonl
if [ "${SKIP_STATES:-0}" != 1 ]; then
  source $VENVS/pyine/bin/activate
  python cvbench/generate.py --cases $CASES --models "$MID" --stages pcprompt
  deactivate
fi
source $VENVS/vllm-q35/bin/activate
python datagen/sample.py --no-suffix --model "$MODEL" --effort "${EFFORT:-default}" --samples-dir data/cvbench/sep2/samples \
  --base-samples "${N:-10}" --variant-samples 0 --temperature 1.0 --top-p 0.9 --top-k 50 --max-tokens "${MAXTOK:-8192}" \
  --chunk-size "${CHUNK:-32}" --tensor-parallel-size "${TP:-1}" --gpu-memory-utilization "${GPUMEM:-0.90}"
echo "done $(date)"
