#!/bin/bash
# CPU job: solutions/trace-heads/research/export.py for the slow heads (SAPLMA refits an MLP per model and fold).
#   PACKAGES="saplma" sbatch --partition=cascade --export=ALL solutions/trace-heads/research/export_cpu.slurm
# (the first copy to start cancels its pending siblings; bigmem only takes jobs asking for >= 745 GB)
#SBATCH --job-name=heads_export
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_out/heads_export_%j.txt
set -eo pipefail
scancel --name="$SLURM_JOB_NAME" --user="$USER" --state=PENDING || true
cd "$SLURM_SUBMIT_DIR"
uv run solutions/trace-heads/research/export.py --only ${PACKAGES:-saplma}
