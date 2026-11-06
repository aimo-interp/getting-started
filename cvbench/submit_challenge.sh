#!/bin/bash
# Submit cvbench/challenge.slurm: Skywork (most cases) and the other three small models as two jobs,
# each to its own set of partitions (skipping partitions where one of our jobs is pending). Run from the repo root.
declare -A QOS=([gpu-a100-short]=normal [gpu-l40s-preempt]=normal [gpu-a100-preempt]=normal [gpu-l40s]=normal
                [gpu-a100]=normal [gpu-h100]=normal [feit-gpu-a100]=feit [deeplearn]=gpgpudeeplearn)
submit() {  # job name, models, partitions...
  local name=$1 models=$2; shift 2
  for p in "$@"; do
    if [ -n "$(squeue -u "$USER" -p "$p" -t PENDING -h -o %i)" ]; then echo "skip $p: a job is pending there"; continue; fi
    MODELS=$models sbatch --export=ALL --job-name="$name" --partition="$p" --qos="${QOS[$p]}" cvbench/challenge.slurm
    sleep 2
  done
}
submit chal_sky "Skywork/Skywork-OR1-Math-7B" gpu-l40s-preempt gpu-a100-short feit-gpu-a100 gpu-a100
submit chal_rest "Qwen/Qwen3.5-4B allenai/Olmo-3-7B-Think deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" \
  gpu-l40s-preempt gpu-l40s gpu-a100-preempt deeplearn gpu-h100
