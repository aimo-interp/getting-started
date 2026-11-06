# Organiser-only competence analysis

Higher measured original-problem solve rate predicts organiser robustness labels,
but these data do not establish that the segment probe mostly reads competence.

All 80 labelled organiser cases (33 problems, 59 robust / 21 fragile) are included.
Correctness is measured on eight cached original solutions at 16,384 tokens per
case, using locally verified reference answers. Unfinished samples count as
failures. Reference provenance and extracted answers are saved for every case.
These are our budget-limited solve rates, not the organiser's original accuracies.

| Finding | Result |
|---|---|
| Solve-rate AUC, pooled | .683 |
| Solve-rate AUC, within-configuration pairs | .753; problem-bootstrap 95% CI [.625, .882] |
| Fragile among cases solved 8/8 | 5/39 (12.8%) |
| Fragile among cases solved <8/8 | 16/41 (39.0%) |
| Organiser-only CV accuracy, solve rate + configuration indicators | .754 |
| Same classifier plus SegProbe | .772 |
| Paired accuracy gain | +.018; problem-bootstrap 95% CI [-.007, .051] |
| Gain with quadratic solve-rate control | +.019; CI [-.015, .058] |
| Gain with completion-share control | +.019; CI [-.016, .065] |

The within-configuration AUC weights model AUCs by robust–fragile pair counts;
it is not a pooled AUC after z-scoring. Its bootstrap resamples whole problems,
keeping repeated model configurations together. Five-fold CV likewise groups
problems, and is repeated 20 times. Probe directions, gates, and solve-rate
normalisation are fitted using training cases only. The classifier uses C=.1
and the existing probe convention of in-sample projection scores for training
cases. The quadratic and completion controls are sensitivity checks, not an
attempt to select a best-performing model.

The group bootstrap captures uncertainty over observed problems with fixed OOF
predictions; it does not refit the entire pipeline in every bootstrap sample.
Descriptive Fisher and within-model permutation p-values in the raw output do
not preserve repeated-problem dependence and are not the primary evidence.

On gpt-oss low effort the association reverses (AUC .300): both fragile cases
are solved 8/8. Qwen and DeepSeek finish few samples, so their measured solve
rates are strongly limited by the budget. Eight samples cannot determine latent
competence. Solve-rate adjustment is a predictive comparison, not causal
mediation or a decomposition of how much the probe reads competence.

Existing full-data OOF predictions flag 2/5 fragile organiser cases solved 8/8
with the stack, compared with 1/5 for TransBlackBox and 1/5 for white-box features.
These descriptive subgroup results use both label sources in training (with
problem grouping); the fresh organiser-only fits above do not. Subgroup AUCs
are .882, .941, and .700 respectively, with only five fragile cases.

Reproduce with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u analysis/report/organiser_competence.py > analysis/report/organiser_competence.txt 2> analysis/report/organiser_competence.err
```

Artifacts:

- `organiser_competence.py`: scoring, reference calculations, and evaluation.
- `organiser_competence_cases.csv`: answers, provenance, solve rates, and OOF scores.
- `organiser_competence_predictions.npz`: per-repeat OOF predictions and features.
- `organiser_competence.txt`: full results, including per-model and subgroup checks.

Validation: all counted completed answers have boxed-answer markers; 80/80 cases
have reference answers; divisor-residue counting is independently checked by
exhaustive exponent enumeration; the polygon calculation checks the cached
MathArena answer. On the first held-out fold, accelerated Gram-matrix probe
features match `stack.safe_seg_scores` using native vector calculations to a
maximum absolute error of 4.2e-14. The analysis script compiles successfully.

The report's main competence paragraph, fragility interpretation, and a new
appendix were updated to reflect these results. No new model generation or
network calls were needed.
