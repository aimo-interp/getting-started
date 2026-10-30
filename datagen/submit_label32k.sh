#!/bin/bash
# Submit datagen/label32k.slurm for each small model to every GPU partition we can use (one copy each,
# distinct job names per model; the first copy to start cancels its siblings). Run from the repo root.
#   bash datagen/submit_label32k.sh                      # all four small models
#   MODELS="Qwen/Qwen3.5-4B" bash datagen/submit_label32k.sh
declare -A QOS=([gpu-a100-short]=normal [gpu-l40s-preempt]=normal [gpu-a100-preempt]=normal [gpu-l40s]=normal
                [gpu-a100]=normal [gpu-h100]=normal [feit-gpu-a100]=feit [deeplearn]=gpgpudeeplearn)
declare -A TIME=([gpu-a100-short]=4:00:00 [gpu-l40s-preempt]=12:00:00 [gpu-a100-preempt]=12:00:00 [gpu-l40s]=12:00:00
                 [gpu-a100]=12:00:00 [gpu-h100]=12:00:00 [feit-gpu-a100]=12:00:00 [deeplearn]=12:00:00)
PARTS=(gpu-a100-short gpu-l40s-preempt gpu-a100-preempt gpu-l40s gpu-a100 gpu-h100 feit-gpu-a100 deeplearn)
MODELS=${MODELS:-"Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B allenai/Olmo-3-7B-Think deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}
for m in $MODELS; do
  short=$(basename "$m" | tr 'A-Z' 'a-z' | cut -d- -f1 | tr -d '.')
  name="label32k_$short"
  for p in "${PARTS[@]}"; do
    MODEL=$m TIME=${TIME[$p]} sbatch --export=ALL --job-name="$name" --partition="$p" --qos="${QOS[$p]}" --time="${TIME[$p]}" datagen/label32k.slurm
  done
done
