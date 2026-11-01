#!/bin/bash
# Fickle-anchor resampling for one small model (cvbench/fickle.py sample). Resumable; first copy cancels siblings.
#   MODEL=Skywork/Skywork-OR1-Math-7B sbatch -J fk_sky --export=ALL -p gpu-a100 cvbench/fickle.slurm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=slurm_out/fk_%x_%j.txt
set -eo pipefail
scancel --name="$SLURM_JOB_NAME" --user="$USER" --state=PENDING || true
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/data/gpfs/projects/punim0478/bansaab
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.8.0
export PATH=$CUDA_HOME/bin:$PATH
source /data/gpfs/projects/punim0478/bansaab/venvs/vllm-q35/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
python cvbench/fickle.py sample --model "$MODEL"
echo "done $(date)"
