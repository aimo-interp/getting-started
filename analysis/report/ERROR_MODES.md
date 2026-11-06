# Error analysis of the full stack (2026-11-06)

All 118 cases, same 20 x 5 grouped folds as the report. Fragile is the positive class. A case counts as wrong if wrong in
at least half of the repeats, which reproduces results_tables.py. On that case-level vote accuracy is .839 (99/118);
the report's .831 is the mean accuracy across repeats. They differ because a vote rounds unstable cases.
Reproduce: errors_collect.py (pyine venv), then errors_table.py, errors_modes.py, errors_stack.py, errors_figure.py
(/usr/bin/python3). Earlier per-method tables in errors_analysis.txt and errors_modes.txt filtered cases and are superseded.

| | Called fragile | Called robust |
|---|---|---|
| True fragile (35) | 20 | 15 |
| True robust (83) | 4 | 79 |

Errors are systematic: 13 of 19 are wrong in every repeat, 4 are wrong in only 50-80%.

Groups are mutually exclusive, applied in this order:

| Trace group | Fragile caught | Robust flagged |
|---|---|---|
| Finished within 4k | 0/3 | 0/27 |
| Solved every time, not within 4k | 2/7 | 1/30 |
| Never finished at 16k | 4/8 | 1/10 |
| Solved sometimes | 14/17 | 2/16 |

## Success
- Finished within 4k and robust: 27/27, the easy group.
- Solved sometimes and fragile: 14/17, where the trace shows repeated self-correction at the cap.

## Missed fragile cases (15)
- Solved every time (8, across the first two groups): every original sample is right and transcript agreement is 1.0.
  Five of these are generated and rest on truncated variants (see below).
- Never finished (4): solve rate measures the token budget here, not ability.
- Solved sometimes (3): two on Skywork, one on gpt-oss high.
- One problem, 7ee9d4437899, is three of the 15, on Skywork, Qwen3.5 and DeepSeek.
- 11 of 15 misses are missed by every block, so stacking did not lose them. Attention is scored as intercept without data.
- By configuration: Skywork 8/16 missed, Qwen3.5 2/4, gpt-oss high and low 4/4, DeepSeek 1/4, Olmo 0/2, gpt-oss medium 0/5.

## Why the errors happened (errors_why_labels.py, errors_why_relabel.py, errors_why_contrib.py)
Generated labels (38 cases) come from 8k runs where a truncated sample counts as wrong. datagen/label_local.py already notes
this bias; its 32k rerun left no output (no data/labelgen/l32k), so the benchmark still uses the 8k labels.
- Failed variant samples on generated fragile cases: 49 of 52 unfinished at 8192 tokens, 3 wrong answers. On the 7 generated
  misses: 21 of 21 unfinished.
- Counting finished samples only, 13 of 14 generated fragile labels become robust, including all 7 misses and 6 of 7 catches.
  Using the answer forced at the cap (Skywork, separate samples): 11 of 13 become robust, 1 stays fragile, 1 unlabelled.
  Ten more samples at the same 8k cap: 3 stay fragile, 4 become robust, 3 unlabelled, 4 have no pooled label.
- Censoring caveat: a slower variant may be a real effect of perturbation under a budget, and a finished-only view can
  hide wrong answers that would have come later. Only a longer run settles it. No robust label changes in any check.
- So the "solved every time, then fails" reading does not hold for generated cases: those samples ran out of tokens.
  Original trace length predicts the generated label (AUC .78), so catches there may partly be near-cap detection.
- Organiser cases give only max_drop, so this cannot be checked for them, and their budget is unknown here.
- Stack accuracy by source, majority vote: organiser 70/80, generated 29/38.

Decision breakdown (fixed final options, 116/118 same outcomes): the intercept gives +1.26 logits to robust. Misses mostly lack
evidence rather than carry opposing evidence: features sum to about -.2 on misses, -2.0 on catches. False alarms have strong
fragile-pointing evidence, from transcript on the two gpt-oss high cases and seg on the Skywork cases.

## False alarms (4)
- All 4 are capped traces in which at least one block is right, so these come from a component, not the combination.
- Transcript is wrong on 3. Both gpt-oss high cases have agreement near 0 because few samples finish: a budget artefact.
- Seg is wrong on 2 Skywork cases. 3f2e57075a10 is robust from one perturbation type only.
- Seg is gated off on nearly all of gpt-oss high, so its right call there is the default.

## Limits
Solve rate: 8 samples at 16k for organiser cases, 4 samples for generated cases. Trace text was read for the 19 errors'
final passages plus a few examples, so descriptions are illustrative. Group p-values are descriptive: repeated problems
are not independent. Labels agree across configurations on 16 of 18 repeated problems; there are 69 problems in all.

## Block view and case study (case_study.py, case_trace.py, case_figure.py)
Capped traces, mean within-model z: caught fragile cases differ on every block, misses sit near passed robust cases, false
alarms look fragile (seg lowest of all). On Skywork seg falls where the model writes "Wait" (decile r -.40, p<.001; across
cases r -.65); step falls more weakly (r -.17). Pair on Skywork, both organiser fragile and capped:
- 335cfaaf1b6e, caught in every repeat: visible confusion, restart, seg lowest at the Wait peak; 31 right, 7 wrong, 2 unfinished on
  valid variants, 300 recurring. The two rename variants are excluded because they rename points but not lengths.
- 7ee9d4437899, missed in every repeat: orderly solution, large steps, few Waits until the end; 8/8 original and 56/56 on seven
  variants correct, so the organiser's drop of 1.0 was not reproduced.
