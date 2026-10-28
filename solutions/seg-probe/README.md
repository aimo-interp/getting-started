# CoT-window diff-in-means probe (`seg_probe`, windows 0-5 + 5-9)

The best method from the `cvbench` harness, packaged for Codabench, for every
evaluated model including gpt-oss-120b. It is
`act-probe` with the response pooled over two windows of the CoT instead of all
of it, motivated by Thought Anchors (a few sentences decide the answer) and by
the finding that the final 10% of the CoT (the answer wrap-up) carries no signal.

1. Greedy-decode the problem (user-only chat template, 2,048 prompt / 4,096 new
   tokens, static cache, batches of 16), as the uncertainty-profiling baseline does.
   For gpt-oss-120b the `reasoning_effort` goes into the chat template (`"default"`
   is the template's default, medium); batches that run out of GPU memory are halved.
2. One forward pass over prompt + response; cut the response into 10 equal token
   segments and average the hidden state at layer `round(0.5 * n_layers)` within each.
3. Two windows: segments 0-4 (first half) and 5-8 (second half without the final 10%),
   each the mean of its segment means.
4. Project each window onto the model's direction for that window (gpt-oss-120b: one
   set of directions per reasoning effort),
   `mean(h | robust) - mean(h | not robust)`, centred and scaled with that model's
   training statistics.
5. A shared logistic regression (C = 0.3) on the two projections gives the label.

## Artifact

`artifacts/seg_probe.json` comes from `research/train.py`, using the cached
`data/cvbench` segments: 118 cases, 81 small-model cases (`cases.jsonl`: train-main-v2
plus labelgen; Qwen3.5-4B, Skywork-OR1-Math-7B, Olmo-3-7B-Think, DeepSeek-R1-0528-Qwen3-8B)
and 37 gpt-oss-120b cases (`cases_oss.jsonl`: train-main-v2 at low 7, medium 15, high 15).
It holds, per model (and per gpt-oss effort), layer and, per window, mean, direction and
scale, plus the shared logistic coefficients. No pickles.

| CV (20 repeats, 118 cases) | Accuracy | AUC | Always-True |
|---|---|---|---|
| seg-probe | 0.776 ± 0.016 | 0.696 | 0.703 |
| seg-probe, one gpt-oss direction for all efforts | 0.767 ± 0.019 | 0.712 | 0.703 |
| act-probe (whole response) | 0.750 ± 0.014 | 0.697 | 0.703 |
| cot_text (black-box) | 0.725 ± 0.019 | 0.715 | 0.703 |

Per model (accuracy / always-True): Qwen 0.871 / 0.667, Skywork 0.794 / 0.667,
Olmo 0.778 / 0.778, DeepSeek 0.567 / 0.667, gpt-oss low 0.671 / 0.714,
gpt-oss medium 0.880 / 0.667, gpt-oss high 0.753 / 0.867. The gpt-oss splits are
tiny (2, 5 and 2 non-robust cases), so only medium's gain is more than noise; high
is not an evaluated effort.

The windows and layer are fixed, not selected inside the CV: they come from an
earlier window sweep on the 82 small-model cases (nested selection over windows
scored 0.720), so the small-model figures are mildly optimistic.

## Fallbacks

Each of these predicts robust (True):

- any model (or gpt-oss effort) without directions;
- a failed model load or batch;
- batches that would overrun the run's time budget. Set it with
  `SEG_PROBE_BUDGET_S`; the default is 3,300 s, measured from import. Each model's
  next batch is budgeted by its own slowest batch so far (first guess 420 s, or
  900 s for gpt-oss).

Responses under 10 tokens use the full response mean for both windows, as in training.

## Cost

Small models: each batch of 16 takes about 1.5–3.5 min on an A100 (4,096 greedy
tokens). gpt-oss-120b at medium effort mostly runs to the 4,096 cap: about 11 min
for 15 problems on two A100s; low effort averages ~1,600 tokens.

## Reproduce

```bash
uv run solutions/seg-probe/research/train.py                     # artifact + CV report
uv run solutions/seg-probe/research/check_fidelity.py --model Qwen/Qwen3.5-4B   # GPU
uv run solutions/seg-probe/research/check_fidelity.py --model openai/gpt-oss-120b:medium   # 80 GB GPU
```
