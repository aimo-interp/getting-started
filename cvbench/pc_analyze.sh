#!/bin/bash
# Re-run the competence-probe analysis on whatever pc.slurm data exists; appends to the loop log.
cd /data/gpfs/projects/punim0478/bansaab/aimo_interp
source /data/gpfs/projects/punim0478/bansaab/venvs/pyine/bin/activate
L=autoresearch/loop-261102-0533
{
echo "===== $(date)"
for f in data/cvbench/sep2/samples/Qwen_Qwen3.5-4B__default.jsonl data/cvbench/sep2/samples/openai_gpt-oss-120b__*.jsonl; do
  [ -f "$f" ] && echo "$f rows=$(wc -l < $f)"; done
python cvbench/pcorrect.py bench 2>&1 | tail -n 14
for b in "seg_safe traj tx" "seg_safe traj tx pc" "seg_safe traj tx pc1" "traj tx pc" "pc" "pc1"; do
  python cvbench/stack.py --blocks $b --repeats 20 2>&1 | grep acc=; done
} >> $L/pc_analysis.txt 2>&1
