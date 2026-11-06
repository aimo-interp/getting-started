"""'Wait' count per token decile of every Skywork greedy trace. Run with the pyine venv. -> case_wait_counts.csv"""
import json, re
from pathlib import Path
import pandas as pd
from transformers import PreTrainedTokenizerFast
ROOT = Path(__file__).resolve().parents[2]
tok = PreTrainedTokenizerFast.from_pretrained("Skywork/Skywork-OR1-Math-7B", local_files_only=True)
rows = []
for l in open(ROOT / "data/cvbench/greedy/Skywork_Skywork-OR1-Math-7B.jsonl"):
    r = json.loads(l); ids = tok(r["generated_text"], add_special_tokens=False)["input_ids"][:4096]; n = len(ids)
    if n < 10: continue
    b = [round(i * n / 10) for i in range(11)]
    rows.append({"group": r["group"], **{f"d{k}": len(re.findall(r"\b[Ww]ait\b", tok.decode(ids[b[k]:b[k + 1]]))) for k in range(10)}})
pd.DataFrame(rows).to_csv(Path(__file__).parent / "case_wait_counts.csv", index=False)
