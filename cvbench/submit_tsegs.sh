#!/bin/bash
# Submit cvbench/tsegs.slurm (any generate.py stages) to every GPU partition we can use, one copy each;
# the first copy to start cancels its pending siblings (same job name). Skips partitions where one of
# our jobs is already pending. Run from the repo root.
#   bash cvbench/submit_tsegs.sh                                   # tsegs, small models + gpt-oss
#   NAME=lookback_small CASES=cases.jsonl STAGES=lookback bash cvbench/submit_tsegs.sh
declare -A QOS=([gpu-a100-short]=normal [gpu-l40s-preempt]=normal [gpu-a100-preempt]=normal [gpu-l40s]=normal
                [gpu-a100]=normal [gpu-h100]=normal [feit-gpu-a100]=feit [deeplearn]=gpgpudeeplearn)
SMALL=(gpu-a100-short gpu-l40s-preempt gpu-a100-preempt gpu-l40s gpu-a100 gpu-h100 feit-gpu-a100 deeplearn)
BIG=(gpu-a100-short gpu-a100-preempt gpu-a100 gpu-h100 feit-gpu-a100 deeplearn)  # gpt-oss-120b: 80 GB cards only
submit() {  # job name, case file, partitions...
  local name=$1 cases=$2; shift 2
  for p in "$@"; do
    if [ -n "$(squeue -u "$USER" -p "$p" -t PENDING -h -o %i)" ]; then
      echo "skip $p: a job is already pending there"; continue
    fi
    CASES=$cases STAGES=${STAGES:-tsegs} sbatch --export=ALL --job-name="$name" --partition="$p" --qos="${QOS[$p]}" "${SCRIPT:-cvbench/tsegs.slurm}"
  done
}
if [ -n "$NAME" ]; then
  if [ "${CASES:-cases.jsonl}" = cases_oss.jsonl ]; then submit "$NAME" "$CASES" "${BIG[@]}"
  else submit "$NAME" "${CASES:-cases.jsonl}" "${SMALL[@]}"; fi
else
  submit tsegs_small cases.jsonl "${SMALL[@]}"
  STAGES="introspect tsegs" submit tsegs_oss cases_oss.jsonl "${BIG[@]}"
fi
