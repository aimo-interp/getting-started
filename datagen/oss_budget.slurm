#!/bin/bash
# Spend watchdog for the oss_labels_* jobs: every 5 min reads the OpenRouter key's usage and cancels them once
# usage - START exceeds LIMIT dollars. START=<usage at launch> LIMIT=15 sbatch --export=ALL datagen/oss_budget.slurm
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=oss_budget
#SBATCH --output=slurm_out/oss_budget_%j.txt
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
while true; do
  usage=$(curl -s https://openrouter.ai/api/v1/auth/key -H "Authorization: Bearer $OPENAI_API_KEY" \
          | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['usage'])" 2>/dev/null)
  running=$(squeue -u "$USER" -h -o %j | grep -c '^oss_labels_' || true)
  echo "$(date +%FT%T) usage=$usage start=$START spent=$(python3 -c "print(round(float('${usage:-0}')-float('$START'),2))") running=$running"
  if [ -n "$usage" ] && python3 -c "import sys; sys.exit(0 if float('$usage') - float('$START') > float('$LIMIT') else 1)"; then
    echo "budget of \$$LIMIT used; cancelling oss_labels jobs"
    scancel --user="$USER" --name=oss_labels_low --name=oss_labels_medium --name=oss_labels_high
    for n in oss_labels_low oss_labels_medium oss_labels_high; do scancel --user="$USER" --name="$n"; done
    exit 0
  fi
  if [ "$running" -eq 0 ]; then echo "no oss_labels jobs left"; exit 0; fi
  sleep 300
done
