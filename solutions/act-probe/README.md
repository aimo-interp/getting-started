# CoT-activation diff-in-means probe (`act_probe`, site `resp_mean`)

The best method from the `cvbench` harness, packaged for Codabench.

1. Greedy-decode the problem (user-only chat template, 2,048 prompt / 4,096 new
   tokens, static cache, batches of 16), as the uncertainty-profiling baseline does.
2. One forward pass over prompt + response; average the hidden state at layer
   `round(0.4 * n_layers)` over the response tokens.
3. Project onto the model's direction `mean(h | robust) - mean(h | not robust)`,
   centred and scaled with that model's training statistics.
4. A shared logistic regression (C = 0.3) on the projection gives the label.

The layer fraction 0.4 is picked by the same grouped inner CV used in the
nested estimate, over the grid 0.2 to 0.9.

## Artifact

`artifacts/act_probe.json` comes from `research/train.py`, using the cached
`data/cvbench` activations (82 cases: train-main-v2 plus labelgen; Qwen3.5-4B,
Skywork-OR1-Math-7B, Olmo-3-7B-Think and DeepSeek-R1-0528-Qwen3-8B). It holds
per-model mean, direction, scale and layer, plus the logistic coefficients. No
pickles.

| Nested CV (20 repeats) | Accuracy | Always-True |
|---|---|---|
| Qwen + Skywork (60 cases) | 0.7525 ± 0.028 | 0.667 |
| All four models (82 cases) | 0.725 ± 0.022 | 0.671 |

Per-model results on all four: Qwen 0.800, Skywork 0.745, Olmo 0.750, DeepSeek 0.565
(DeepSeek's always-True is 0.615).

## Fallbacks

Each of these predicts robust (True):

- `openai/gpt-oss-120b`, or any model without a direction;
- a failed model load or batch;
- batches that would overrun the run's time budget. Set it with
  `ACT_PROBE_BUDGET_S`; the default is 3,300 s, measured from import.

## Cost

Each batch of 16 takes about 1.5–3.5 min on an A100 (4,096 greedy tokens), so
roughly 15–20 batches fit in the hour.

## Reproduce

```bash
uv run solutions/act-probe/research/train.py                     # artifact + nested-CV report
uv run solutions/act-probe/research/check_fidelity.py --model Qwen/Qwen3.5-4B   # GPU
```
