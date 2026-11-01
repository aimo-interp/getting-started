#!/bin/bash
# CPU watcher: every 20 min while pc_* jobs exist, re-run cvbench/pc_analyze.sh when sample files grew; final run at the end.
#SBATCH --job-name=pc_watch
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_out/pc_watch_%j.txt
cd /data/gpfs/projects/punim0478/bansaab/aimo_interp
last=""
while true; do
  cur=$(cat data/cvbench/sep2/samples/Qwen_Qwen3.5-4B__default.jsonl data/cvbench/sep2/samples/openai_gpt-oss-120b__*.jsonl 2>/dev/null | wc -l)
  if [ "$cur" != "$last" ]; then bash cvbench/pc_analyze.sh; last=$cur; fi
  n=$(squeue -u "$USER" -h -o %j | grep -c '^pc_\(oss\|qwen\)')
  [ "$n" = 0 ] && { bash cvbench/pc_analyze.sh; echo finished; exit 0; }
  sleep 1200
done
