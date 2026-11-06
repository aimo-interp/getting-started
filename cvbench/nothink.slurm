#!/bin/bash
# No-reasoning semantic entropy (generate.py --stages nothink, direct mode: think block closed, \boxed{
# prefilled, n=50 answers of ~16 tokens) for every cvbench model in one short job: the 4 small models, then
# gpt-oss-120b (MXFP4 fits one 80 GB card) at low/medium/high. Resumable. The first copy to start cancels
# its pending siblings:
#   for p in gpu-a100-short gpu-a100 feit-gpu-a100 gpu-h100; do sbatch --partition=$p ... cvbench/nothink.slurm; done
# FREE=1 also runs the free-answer mode (n=16, 768 tokens; much slower).
#SBATCH --job-name=nothink
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:45:00
#SBATCH --output=slurm_out/nothink_%j.txt
set -eo pipefail
scancel --name="$SLURM_JOB_NAME" --user="$USER" --state=PENDING || true
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
export HF_HOME=/data/gpfs/projects/punim0478/bansaab
source /data/gpfs/projects/punim0478/bansaab/venvs/pyine/bin/activate
python cvbench/generate.py --cases cases.jsonl --stages nothink --n 50 --nt-mode direct --nt-max-new 16
if [ "$mib" -ge 75000 ]; then
  python cvbench/generate.py --cases cases_oss.jsonl --stages nothink --n 50 --nt-mode direct --nt-max-new 16
else echo "GPU has $mib MiB; skipping gpt-oss-120b"; fi
if [ -n "$FREE" ]; then
  python cvbench/generate.py --cases cases.jsonl --stages nothink --n 16 --nt-mode free --nt-max-new 768
fi
