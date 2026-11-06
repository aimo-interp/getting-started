# Reproducing the CV results

Run from the repo root. CV is 5-fold grouped by problem, 20 repeats, on 118 cases.

```bash
export HF_HOME=/data/gpfs/projects/punim0478/bansaab
source $HF_HOME/venvs/pyine/bin/activate
SMALL="Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B allenai/Olmo-3-7B-Think deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
```

GPUs: gpt-oss-120b needs one 80 GB card (or 2 L40S) and 120 GB RAM. The small models ran on 40-80 GB cards.

## Inference (GPU)

```bash
# Seg probe: greedy 4k traces and activations (traj reads these too)
python cvbench/generate.py --stages greedy acts segs
sbatch --export=ALL --partition=feit-gpu-a100 --qos=feit --gres=gpu:1 --mem=120G cvbench/oss.slurm    # gpt-oss

# Transcript probe: 8 sampled traces at 16k tokens (uses vllm-q35; repeat per model)
STAGES="prepare screen" MODEL=Skywork/Skywork-OR1-Math-7B JOB_NAME=anchors_sky TIME=24:00:00 \
    bash solutions/anchors/submit.sh
OPENROUTER_API_KEY=... python solutions/transcript-probe/research/sample_oss.py                       # gpt-oss, no GPU

# Attention (gpt-oss was not run and is zero-filled)
python cvbench/attn_collect.py --models $SMALL
```

## Training and CV (CPU)

| Method | Command | Accuracy | AUC |
|---|---|---|---|
| Seg probe | `python cvbench/stack.py --blocks seg_safe` | .774 | .727 |
| Transcript probe, 16k | `python cvbench/stack.py --blocks tx` | .772 | .851 |
| Attention | `python cvbench/stack.py --blocks attn` | .711 | .644 |

Stack, with options chosen per model (.831, AUC .850):

```bash
python cvbench/stack.py --blocks seg_safe traj tx attn \
    --gate seg_safe+traj+tx+attn traj+tx+attn seg_safe+traj+tx traj+tx
```

Full ablation with bootstrap intervals:

```bash
python analysis/report/ablation_candidates.py --repeats 20 --save analysis/report/ablation_candidates.npz
```
