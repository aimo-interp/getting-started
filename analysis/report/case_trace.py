"""Variant outcomes and token-decile text for the Skywork case pair. Run with the pyine venv (needs the tokenizer)."""
import json, re, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast.from_pretrained("Skywork/Skywork-OR1-Math-7B", local_files_only=True)
R = [json.loads(l) for l in open(ROOT / "data/anchors/Skywork_Skywork-OR1-Math-7B/screen.jsonl")]
G = {json.loads(l)["group"]: json.loads(l) for l in open(ROOT / "data/cvbench/greedy/Skywork_Skywork-OR1-Math-7B.jsonl")}
for g, gold in (("335cfaaf1b6e", 588), ("7ee9d4437899", 26244)):
    x = [r for r in R if r["group"] == g]; print(f"\n==== {g}: {len(x)} prompts, gold {gold}, variants {[r['variant'] for r in x]}")
    tot = Counter()
    for r in x:
        fam = "original" if r["variant"] == "X" else r["variant"].split("-")[0]; c = Counter("right" if a == gold else ("unfinished" if a is None else "wrong") for a in r["answers"])
        print(f"  {r['variant']:12s} {dict(c)} wrong answers {[a for a in r['answers'] if a not in (gold, None)]} tokens {min(r['n_tokens'])}-{max(r['n_tokens'])}")
        if fam not in ("original", "rename") or g != "335cfaaf1b6e":
            if fam != "original": tot.update(c)
    print("  variants excluding broken rename:", dict(tot))
    ids = tok(G[g]["generated_text"], add_special_tokens=False)["input_ids"][:4096]; n = len(ids); b = [round(i * n / 10) for i in range(11)]
    print("  tokens", n)
    for k in range(10):
        t = tok.decode(ids[b[k]:b[k + 1]]); w = len(re.findall(r"\b[Ww]ait\b", t)); print(f"  d{k + 1} wait={w:2d} equals={t.count('='):3d} | {t[:150]!r} ... {t[-110:]!r}")
