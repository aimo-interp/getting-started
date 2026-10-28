#!/bin/bash
# gpt-oss-120b (MXFP4, 2 GPUs) on data/cvbench/cases_oss.jsonl, every effort: greedy CoT, then the
# acts/segs/pool activations act_probe and seg_probe need. Resumable; the first copy to start
# cancels its pending siblings. Submit one copy per free non-H100 partition, e.g.
#   sbatch --export=ALL --partition=feit-gpu-a100 --qos=feit cvbench/oss.slurm
# One 80 GB A100 also fits (MXFP4 weights ~65 GB): add --gres=gpu:1 and BATCH=8.
#SBATCH --job-name=oss_acts
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_out/oss_%j.txt
set -eo pipefail
# MXFP4 gpt-oss-120b needs ~65 GB per GPU when run on one: a smaller card leaves its siblings queued.
mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$mib" -lt 75000 ] && [ "${SLURM_GPUS_ON_NODE:-1}" -lt 2 ]; then echo "GPU has $mib MiB; need an 80 GB card"; exit 1; fi
scancel --name="$SLURM_JOB_NAME" --user="$USER" --state=PENDING || true
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/data/gpfs/projects/punim0478/bansaab  # online: the MXFP4 kernel resolves its version tag
nvidia-smi --query-gpu=name,memory.total --format=csv
source /data/gpfs/projects/punim0478/bansaab/venvs/pyine/bin/activate
python cvbench/generate.py --cases cases_oss.jsonl --stages greedy acts segs pool --batch-size "${BATCH:-16}"
