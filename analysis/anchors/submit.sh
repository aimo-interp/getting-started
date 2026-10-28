#!/bin/bash
# Submit analysis/anchors/anchors.slurm to every allowed GPU partition that has none of our jobs pending.
# Run from the repo root:  bash analysis/anchors/submit.sh   (env vars such as N, STAGES pass through)
mkdir -p analysis/anchors/work/logs
declare -A QOS=([feit-gpu-a100]=feit [gpu-a100]=normal [deeplearn]=gpgpudeeplearn [gpu-l40s]=normal)
for p in feit-gpu-a100 gpu-a100 deeplearn gpu-l40s; do
  if [ -n "$(squeue -u "$USER" -p "$p" -t PENDING -h -o %i)" ]; then
    echo "skip $p: a job is already pending there"; continue
  fi
  sbatch --export=ALL --partition="$p" --qos="${QOS[$p]}" analysis/anchors/anchors.slurm
done
