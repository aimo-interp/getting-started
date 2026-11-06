#!/bin/bash
# seg_probe activations over the transcript probe's 8 sampled 16k traces per case (generate.py --stages tsegs).
# Resumable; the first copy to start cancels its pending siblings (same job name). Submit with
# cvbench/submit_tsegs.sh. Short slots backfill sooner: the run stops itself 2 min before the
# 20 min limit and, if not finished, resubmits this script to the same partition. gpt-oss-120b (CASES=cases_oss.jsonl) needs an 80 GB card for MXFP4 on one GPU.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_out/tsegs_%x_%j.txt
set -eo pipefail
scancel --name="$SLURM_JOB_NAME" --user="$USER" --state=PENDING || true
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/data/gpfs/projects/punim0478/bansaab  # online: the MXFP4 kernel resolves its version tag
nvidia-smi --query-gpu=name,memory.total --format=csv
if [ "${CASES:-cases.jsonl}" = cases_oss.jsonl ]; then
  mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  if [ "$mib" -lt 75000 ]; then echo "GPU has $mib MiB; gpt-oss-120b needs an 80 GB card"; exit 1; fi
fi
source /data/gpfs/projects/punim0478/bansaab/venvs/pyine/bin/activate
rc=0
timeout --signal=INT 18m python cvbench/generate.py --cases "${CASES:-cases.jsonl}" --stages ${STAGES:-tsegs} || rc=$?
if [ "$rc" -eq 124 ]; then  # out of time, not done: next slot picks up where this one stopped
  echo "time slice used up; resubmitting to $SLURM_JOB_PARTITION"
  sbatch --export=ALL --job-name="$SLURM_JOB_NAME" --partition="$SLURM_JOB_PARTITION" --qos="$SLURM_JOB_QOS" \
         cvbench/tsegs.slurm
  exit 0
fi
exit "$rc"
