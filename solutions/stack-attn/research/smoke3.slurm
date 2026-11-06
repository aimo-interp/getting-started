#!/bin/bash
# stack-attn final smoke: fresh run clock; DeepSeek attention check (runtime tokenizer) + full calls on DeepSeek and Olmo.
#SBATCH --job-name=smoke_attn3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_out/smoke_attn3_%j.txt
scancel --name="$SLURM_JOB_NAME" --user="$USER" --state=PENDING || true
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/data/gpfs/projects/punim0478/bansaab
export TMPDIR=$(mktemp -d)   # fresh engine clock (state file lives in tempdir)
source /data/gpfs/projects/punim0478/bansaab/venvs/pyine/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
cd solutions/stack-attn && python - <<'PY'
import json, sys, time
sys.path.insert(0, "../../cvbench")
import cv, stack
from stack_probe import engine
from uncertainty_profile.extraction import load_model_and_tokenizer
art = json.load(open("artifacts/stack.json"))
c = cv.load_cases(list(cv.CASE_FILES)); A = stack.attn_feats(c)
M = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"; s = c[c.model_id == M]; e = art["models"][M]
greedy = {r["group"]: r for r in cv.read_jsonl(cv.DATA / "greedy" / f"{cv.safe(M)}.jsonl")}
model, tok = load_model_and_tokenizer(M, local_files_only=True)
for g in list(s.group)[:2]:
    i = s.index[s.group == g][0]
    raw = engine.attention_signals(model, tok, s.loc[i, "problem"], greedy[g]["generated_text"], {**art["attention"], **e["attn_layers"]})
    print("CHECK", g, "engine", {k: round(v, 5) for k, v in raw.items()}, "train", A.loc[i].round(5).to_dict(), flush=True)
del model; engine.release()
for M in ["deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "allenai/Olmo-3-7B-Think"]:
    case = c[c.model_id == M].iloc[1]
    t0 = time.time()
    print("FULL", M, "label", case.is_robust, "->", engine.are_robust(M, "default", [case.problem]), f"{time.time() - t0:.0f}s", flush=True)
PY
