#!/bin/bash
# Self-challenge / CoVe continuations (cvbench/challenge.py) for MODELS, resumable. Submit with
# cvbench/submit_challenge.sh; the first copy to start cancels its pending siblings (same job name).
# The run stops itself 2 min before the 30 min limit and, if not finished, resubmits to the same partition.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_out/challenge_%x_%j.txt
set -eo pipefail
scancel --name="$SLURM_JOB_NAME" --user="$USER" --state=PENDING || true
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/data/gpfs/projects/punim0478/bansaab
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.8.0  # vLLM 0.32 JIT-compiles kernels
export PATH=$CUDA_HOME/bin:$PATH
nvidia-smi --query-gpu=name,memory.total --format=csv
source /data/gpfs/projects/punim0478/bansaab/venvs/vllm-q35/bin/activate
end=$(( $(date +%s) + 27 * 60 ))
for MODEL in ${MODELS:?set MODELS}; do
  left=$(( end - $(date +%s) ))
  rc=0
  [ "$left" -gt 120 ] && { timeout --signal=INT "$left" python cvbench/challenge.py --model "$MODEL" || rc=$?; } || rc=124
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 130 ]; then
    echo "time slice used up at $MODEL; resubmitting to $SLURM_JOB_PARTITION"
    sbatch --export=ALL --job-name="$SLURM_JOB_NAME" --partition="$SLURM_JOB_PARTITION" --qos="$SLURM_JOB_QOS" \
           cvbench/challenge.slurm
    exit 0
  fi
  [ "$rc" -ne 0 ] && echo "$MODEL failed rc=$rc"
done
exit 0
