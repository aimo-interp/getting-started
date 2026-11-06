# Seg probe / stack error analysis (2026-11-02)

`collect.py` (run from cvbench/) -> `oof_cases.csv`: per-case OOF probability and error rate over 20x5 grouped CV for
seg_safe (acc .774), ungated seg (.765), traj+tx (.797), and the stack (.819, reproduces the baseline).
`analyze.py` -> `analyze.txt`, `cases_annotated.csv`. "Wrong" = wrong in >= half the repeats.

## Errors are almost all missed fragile cases
- seg_safe: 24/35 fragile missed, 2/83 false alarms. Stack: 16/35 missed, 4/83 false alarms.
- So the probes act as "is this case unusually fragile-looking". Their default is robust (70% base rate).
- Seg misses come in two kinds:
  - gated off (6 cases; DeepSeek and gpt-oss high, where the window is dropped)
  - a DIM score that points to robust (18)
- Missed vs caught fragile cases do not differ in max_drop, transcript agreement, difficulty, length or logprob
  (MWU p > .18). Nothing in our features flags them.

## By model
| model | n | frag | stack wrong |
|---|---|---|---|
| Skywork | 48 | 16 | 9 |
| DeepSeek | 12 | 4 | 4 (all fragile missed) |
| gpt-oss high | 15 | 2 | 4 (2 missed + 2 false alarms from traj/tx) |
| gpt-oss low | 7 | 2 | 2 |
| Qwen | 12 | 4 | 1 |
| Olmo, gpt-oss medium | 9, 15 | 2, 5 | 0 |

## Which perturbation broke it (labelgen cases only: 14 of 35 fragile)
train-main-v2 only gives max_drop. Labelgen has per-type accuracy, but only from 1 variant x 2 samples per type,
against 4 samples on the original.
- Caught by the stack, by culprit type:

  | type | caught |
  |---|---|
  | distract | 4/6 |
  | domain | 2/3 |
  | typos | 4/8 |
  | rephrase | 3/7 |
  | expert | 2/5 |
  | rename | 2/5 |

  The groups overlap and are tiny: no type-specific pattern can be claimed.
- Base accuracy matters more than perturbation type:
  - fragile cases where the original was solved 4/4: caught 1/5, mean p_stack .71
  - fragile cases where the original was solved < 4/4: caught 6/9, mean p_stack .42 (Fisher p .27)
- **Reading:** the probes detect weak competence on the original problem, not sensitivity to perturbation. A model
  that solves the original perfectly but drops one of two samples on a variant looks robust. That label is also
  the noisiest: one wrong sample out of 2.
- This matches the fickle-anchor result (no signal once p0 >= .9).

## Repeated misses
- Problem 7ee9d4437899 (divisors of 17017^17 that are 5 mod 12) is fragile on all 4 small models. The stack misses it
  on Qwen, Skywork and DeepSeek (p .67-.83).
- On Skywork it has max_drop 1.0 and agreement 1.0: the model consistently solves the original, then fails
  every variant.
