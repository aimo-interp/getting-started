# Which components write the seg-probe direction? Skywork-OR1-Math-7B (2026-11-02)

- Code: `cvbench/circuit_attr.py` (`collect` on GPU, `analyze` on CPU).
- Data: `data/cvbench/circuit/`.
- Raw output: `circuit_skywork.txt`.
- Method: the residual after layer 14 equals the embedding + 14x28 attention heads + 14 MLPs, so the robust-minus-fragile
  projection splits exactly. Reconstruction error vs the stored segs is 0.1% median.
- n = 48 (32 robust, 16 fragile).

## Results
- **No small circuit.**
  - Attention writes ~95-100% of the gap; MLPs net ~0.
  - Layer 13, the last before the probe, writes ~21%. The rest builds up gradually from layer 6.
  - The top 5 components hold 10-11% of |gap|, the top 20 hold 30%, and it takes ~50 to reach half.
  - The OOF AUC of the residual is already .60-.63 at layer 0 and climbs only to .67-.68 at layer 14.
- Single heads with OOF AUC ~.75-.82: L13.H4, L11.H24, L13.H20, L10.H19, L6.H9.
  - L13.H18, L11.H22 and L10.H14 point the other way (AUC .18-.31).
  - None dominates.
- **Tokens:** the top heads (L13.H16/17/18, L10.H19) read **topic vocabulary**.
  - Fragile-pushing: `-axis`, `θ`, `cos`, `circle`, `coordinates`, names (`Carl`, `Sergey`, `Jon`), `probability`.
  - Robust-pushing: `log`, `gcd`, `prime`, `-digit`, `rem`, `integer`.
- **But topic is not the signal.**
  - A keyword topic score has AUC .51 for Skywork robustness (.49 pooled).
  - Geometry problems are fragile 40% of the time vs 37% for number theory.
  - p_seg correlates .37 with the topic score, yet still separates robust from fragile *within* a topic:
    - geometry: AUC .87 (n=15)
    - number theory: AUC .73 (n=19)
    - other: AUC .67 (n=14)
  - So with n=48 the diff-in-means direction carries a topic component that adds variance without adding signal.

## Reading
"Robustness" here is a distributed, mostly attention-written feature that builds over depth. It is not a circuit
you can isolate with a handful of heads. The clearest structure found is a nuisance topic axis.

## Next ideas
- Project the topic axis out of the direction before scoring. Earlier PC removal hurt Skywork, so this needs a
  supervised topic direction, not top PCs.
- Read attention patterns of the high-AUC heads (L13.H4, L11.H24) to see what they attend to: the problem statement
  vs. their own CoT.

## Follow-ups (same day)

### Topic removal
`stack.py` blocks `seg_notopic` / `seg_notopic4` project keyword-topic class-mean directions out of the seg windows,
per model and per layer.
- Paired vs seg_safe+traj+tx: **-.041 ± .023**. Qwen drops .17, Skywork .06. Raw output: `topic_paired_*.txt`.
- So the topic axis is not pure nuisance: in small per-model samples, topic and robustness are partly tied
  (Qwen: all 4 fragile cases are number theory).
- Not shipped.

### Attention patterns
Code: `circuit_attr.py attn`. Output: `attn_skywork.txt`. These scores are averaged over ALL heads (no head
selection), so they are honest.
- Robust traces put more attention on the BOS sink and chat template: AUC .74-.79, perm p .001-.007.
- Fragile traces look further back into their own CoT (>64 tokens back): AUC .70, p .025.
- Attention to the problem text: no signal (.44-.49).
- These correlate ~.5 with p_seg. On Skywork alone, OOF AUC goes from .743 (p_seg) to .771 (p_seg + far + sink).
- Reading: in fragile traces the heads are busier retrieving earlier reasoning; in robust traces they idle on the
  sink. This fits "fragile = the model is struggling".
- Focus heads have no distinctive targets. Their problem attention is mostly a secondary sink on the first problem
  tokens ("Let", "A"). The layer-13 heads read the question tail ("?", "Find") and type words ("prime", "integers").

### Math vs narrative words (`math_words.py`, `math_words.txt`)
- Share of non-sink problem attention on math tokens vs narrative words: robust .622 vs fragile .609, AUC .54.
  No real difference.
- Narrative fraction of the problem text vs fragility:
  - pooled AUC .51
  - Skywork .63
  - correlation with labelgen distract drop .16
