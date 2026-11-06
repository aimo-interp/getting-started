#!/bin/bash
# GPU attention pass of cvbench/distract_attn.py (gpt-oss-120b:low traces on organiser distract variants).
# Resumable; the first copy to start cancels its pending siblings (same job name). Submit with
# SCRIPT=cvbench/distract.slurm NAME=distract CASES=cases_oss.jsonl bash cvbench/submit_tsegs.sh (80 GB cards),
# or on L40S: sbatch --partition=gpu-l40s --gres=gpu:2 --job-name=distract cvbench/distract.slurm (model split over 2 cards).
# Stops itself 2 min before the 20 min limit and, if not finished, resubmits to the same partition.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_out/%x_%j.txt
set -eo pipefail
scancel --name="$SLURM_JOB_NAME" --user="$USER" --state=PENDING || true
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/data/gpfs/projects/punim0478/bansaab  # online: the MXFP4 kernel resolves its version tag
mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
if [ "$mib" -lt 75000 ]; then echo "GPUs have $mib MiB in total; gpt-oss-120b needs 80 GB (one A100/H100 or 2 L40S)"; exit 1; fi
source /data/gpfs/projects/punim0478/bansaab/venvs/pyine/bin/activate
rc=0
timeout --signal=INT 18m python cvbench/distract_attn.py attn || rc=$?
if [ "$rc" -eq 124 ]; then
  echo "time slice used up; resubmitting to $SLURM_JOB_PARTITION"
  sbatch --export=ALL --job-name="$SLURM_JOB_NAME" --partition="$SLURM_JOB_PARTITION" --qos="$SLURM_JOB_QOS" \
         --gres="gpu:$(nvidia-smi -L | wc -l)" cvbench/distract.slurm
  exit 0
fi
exit "$rc"
