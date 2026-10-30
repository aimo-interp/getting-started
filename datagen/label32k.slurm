#!/bin/bash
# 32k-token robustness labels for one local model (datagen/label_local.py), resumable. Submit with
# datagen/submit_label32k.sh: one copy per GPU partition, the first to start cancels its pending siblings
# (same job name); stops 10 min before its limit and resubmits itself to the same partition if not done.
# Env: MODEL (HF id), TIME (this partition's limit, passed on at resubmission), TARGET (labels, default 100).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --requeue
#SBATCH --output=slurm_out/label32k_%x_%j.txt
set -eo pipefail
scancel --name="$SLURM_JOB_NAME" --user="$USER" --state=PENDING || true
cd "$SLURM_SUBMIT_DIR"
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.8.0  # vLLM JIT-compiles kernels
export PATH=$CUDA_HOME/bin:$PATH
export HF_HOME=/data/gpfs/projects/punim0478/bansaab
nvidia-smi --query-gpu=name,memory.total --format=csv
source /data/gpfs/projects/punim0478/bansaab/venvs/vllm-q35/bin/activate
# minutes available: TIME is H:MM:SS or D-HH:MM:SS
mins=$(python3 -c "
import sys; t='$TIME'; d,_,hms=t.rpartition('-'); h,m,s=hms.split(':'); print(int(d or 0)*1440+int(h)*60+int(m))")
rc=0
timeout --signal=INT "$((mins - 10))m" python datagen/label_local.py --model "$MODEL" --target-labels "${TARGET:-100}" || rc=$?
if [ "$rc" -eq 124 ]; then
  echo "time slice used up; resubmitting to $SLURM_JOB_PARTITION"
  sbatch --export=ALL --job-name="$SLURM_JOB_NAME" --partition="$SLURM_JOB_PARTITION" --qos="$SLURM_JOB_QOS" \
         --time="$TIME" datagen/label32k.slurm
  exit 0
fi
exit "$rc"
