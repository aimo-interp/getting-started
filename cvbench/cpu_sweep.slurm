#!/bin/bash
# Run a list of stack.py configurations on a CPU node, 4 at a time (the interactive node has 1 core).
# CONFIGS holds one stack.py argument string per line; results go to OUT, one line each, then "done".
#   CONFIGS=$'--blocks seg traj tx\n--blocks seg_ml traj tx' OUT=autoresearch/x.txt sbatch --export=ALL cvbench/cpu_sweep.slurm
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_out/cpu_sweep_%j.txt
cd "$SLURM_SUBMIT_DIR"
PY=/data/gpfs/projects/punim0478/bansaab/venvs/pyine/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
: > "$OUT"
i=0
while IFS= read -r cfg; do
  [ -z "$cfg" ] && continue
  ( r=$($PY cvbench/stack.py --repeats "${REPEATS:-10}" $cfg 2>&1 | grep -E " acc=|Error" | tail -1 | cut -c1-360)
    echo "[$cfg] $r" >> "$OUT" ) &
  i=$((i + 1))
  if [ $((i % 4)) -eq 0 ]; then wait; fi
done <<< "$CONFIGS"
wait
echo done >> "$OUT"
