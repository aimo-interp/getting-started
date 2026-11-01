#!/bin/bash
# gpt-oss-120b solve-rate samples via OpenRouter for the competence probe (pcorrect.py): 8 samples per labelgen
# original at low/medium/high -> data/cvbench/sep2/samples/openai_gpt-oss-120b__<effort>.jsonl. Resumable.
# Spend guard: kills itself if account usage exceeds STOP_USAGE dollars.
#SBATCH --job-name=pc_or
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_out/pc_or_%j.txt
cd /data/gpfs/projects/punim0478/bansaab/aimo_interp
set -a; . ./.env; set +a
source /data/gpfs/projects/punim0478/bansaab/venvs/pyine/bin/activate
STOP_USAGE=${STOP_USAGE:-52}
usage() { curl -s -H "Authorization: Bearer $OPENAI_API_KEY" https://openrouter.ai/api/v1/credits | python3 -c "import json,sys;print(json.load(sys.stdin)['data']['total_usage'])"; }
( while sleep 120; do u=$(usage); echo "[guard] usage \$$u"; if python3 -c "import sys;sys.exit(0 if float('$u')>$STOP_USAGE else 1)"; then echo "[guard] over budget, stopping"; pkill -f "sample.py --backend openrouter"; scancel $SLURM_JOB_ID 2>/dev/null; kill $$; fi; done ) &
exec 9>data/cvbench/sep2/pc_or.lock; flock 9   # one sampler at a time (local run or Slurm copy)
for spec in low:16384 medium:32768 high:32768; do
  e=${spec%%:*}; cap=${spec##*:}
  echo "=== $e cap $cap $(date) usage \$$(usage)"
  for try in 1 2; do
    python datagen/sample.py --backend openrouter --model openai/gpt-oss-120b --effort $e \
      --samples-dir data/cvbench/sep2/samples --base-samples 8 --variant-samples 0 --max-tokens $cap --workers 48
  done
done
echo "done $(date) usage \$$(usage)"
