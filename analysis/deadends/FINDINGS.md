# Dead-end audit, part 1 (2026-11-02): distractor attention, difficulty fill, PC removal

Scripts and raw outputs are in this directory: `fill_audit.py`, `paired.py`, `pc_geometry.py`, `perm_null.py`,
plus `*.txt` and `*.npz`.
Method note: ±.016 is the across-repeat sd of ONE method. Two methods run on the same folds should be
compared by the sd of the per-repeat difference (`paired.py`), which is often 2x smaller.

## 1. Distractor attention: the null result is uninformative (the experiment could not test the hypothesis)
- The organiser's `permutations_causing_decay` come from **one sample per variant**
  (n_permuted_predictions = 10 across 10 variants). "variant accuracy 0.00" means a single wrong sample.
- Our 8 resamples per variant: variant error rate .23 vs original-problem error rate .21 (Fisher p .78).
  Excluding 349493 it is .14 vs .14. **The distractors do not hurt gpt-oss low.** The "decay" was
  ordinary 10-30% error sampled once.
- So "wrong vs right traces" contrasts ordinary errors with successes, not distractor-induced failures.
  A null on clause attention is expected either way.
- More problems:
  - 9 of the 26 mixed variants are 349493. Its reference answer is the modal answer on only 7/16 samples
    (5 more gave no answer), so the right/wrong labels there are mostly noise.
  - The word-level diff spans also catch reformatted problem text:
    - 71beb6_0/3: `N=10^{100}-2`, `S(S(1)+...)`
    - 1fce4b_1/2: scattered words
    - every span: the token before the clause, e.g. `1$`, `a_N$`
- Text check: wrong traces echo clause-only words no more than right traces (12/26, p .65).
- **To actually test it:** first find variants that really hurt. Resample about 8 per variant and keep those
  where variant error is much higher than original error. Then compare attention on those variants
  against the original problem.

## 2. Difficulty fill: real signal, but almost no fragile cases where it would apply
- The label-free difficulty score (E2H ridge, `diff_last`) separates robust cases on exactly the models where
  the seg gate switches off:

  | model | AUC |
  |---|---|
  | DeepSeek | .81 |
  | gpt-oss high | .85 |
  | gpt-oss low | 1.00 (n=7) |
  | pooled | .74 |

- But both seg windows are gated off for only 23 cases, 78% of them robust (5 fragile). The most a perfect
  fill could add is .042 accuracy. gpt-oss high's best in-sample threshold equals always-True (.87).
- Paired, `diff_fill` = +.0025 ± .007. It never makes a correct prediction wrong (0.2 vs 0.5 flips per repeat).
  Its coefficient is a reasonable +.31, but those cases sit far from the 0.5 boundary.
- The plain `diff` column (all models): log-loss −.017, AUC +.019, Olmo +.07, Qwen +.02, but Skywork −.04.
  The shared logistic coefficient is a compromise across models. `seg_fill` also moves Qwen (−.04), which is
  never gated, because its fill values change the shared seg coefficient.
- **Reading:** difficulty is not useless. Thresholded accuracy at n=118 cannot see a gain that lives in 5
  fragile cases, and a shared coefficient spreads one model's fit onto the others. Use log-loss or AUC on the
  affected subset, or give difficulty a per-model coefficient (or per-model gate option).

## 3. PC removal: genuinely worse for most models, because the signal IS the top PC
- Paired Δacc vs seg_safe+traj+tx:

  | variant | Δacc |
  |---|---|
  | k=1 | −.037 ± .019 |
  | k=3 | −.028 ± .016 |
  | k=5 | −.028 ± .021 |

  This is a real loss, not a null. The pooled AUC *rises* (+.006 to +.018) while log-loss worsens
  (k=5: +.088). Pooled AUC mixes rankings across models and is misleading here.
- Where seg works (Qwen, Skywork, gpt-oss medium, Olmo), 60-98% of the DIM direction's squared norm lies in
  the top 3 PCs. PC1 alone has AUC .88 (Qwen) and .94/.98 (gpt-oss medium). For gpt-oss medium window 5-9,
  PC1 correlates |r| = .82 with greedy log length.
- **DeepSeek is the exception:** the OOF AUC goes .31 → .72 (k=3) → .81 (k=5). Permutation p = .03 at k=5
  (4 k values tried, so marginal). In the stack: DeepSeek .65 → .79 (k=3) / .83 (k=5), while every other
  model loses. A per-model k picked by inner CV, specifically k≈5 for DeepSeek, is a real lead.
- **Permutation null (within-model label shuffles, 30x):**
  - Qwen p .03, Skywork p .00, gpt-oss medium p .00 at k=0. These are real signal.
  - Olmo p .07.
  - gpt-oss high and low: no signal at any k.
  - With 2-4 fragile cases, the null 90% interval for OOF AUC is roughly [.05, .90]. Single-model AUCs (and
    the inner-CV AUC ≥ .6 gate) are very noisy for those models. gpt-oss high's ungated OOF AUC of 0.00 sits
    at the bottom of its own null.
- Greedy log length is constant (4k cap) for Qwen and DeepSeek, and missing for Olmo. Length-PC
  correlations exist only for Skywork and gpt-oss.
