#!/bin/bash
# Submit cvbench/sample.slurm to every allowed GPU partition that has none of our jobs pending.
# Usage: MODELS="a b" bash cvbench/submit_sample.sh
declare -A QOS=([feit-gpu-a100]=feit [gpu-a100]=normal [deeplearn]=gpgpudeeplearn [gpu-l40s]=normal)
for p in feit-gpu-a100 gpu-a100 deeplearn gpu-l40s; do
  if [ -n "$(squeue -u "$USER" -p "$p" -t PENDING -h -o %i)" ]; then
    echo "skip $p: a job is already pending there"; continue
  fi
  sbatch --export=ALL --partition="$p" --qos="${QOS[$p]}" cvbench/sample.slurm
done
