# Uncertainty-Profiling Baseline

This directory is a self-contained Codabench solution. It generates one
deterministic response per input problem, summarizes the generated-token
distributions with 14 uncertainty features, and uses a bundled regression
artifact to predict whether the evaluated model is robust.

## Motivation

This baseline measures robustness through the model's uncertainty over its
generated reasoning trace. Each next-token distribution exposes how
concentrated or diffuse the model's predictive belief is as the response
unfolds. The working hypothesis is that robust reasoning has a different
uncertainty signature from brittle or spurious reasoning, even when both traces
look superficially plausible.

This motivation follows the perspective of
[Tracing Uncertainty in Language Model "Reasoning"](https://arxiv.org/abs/2605.07776),
which treats intermediate reasoning tokens as evolving model states and uses
the shape of their uncertainty signal to predict answer correctness. Our
baseline adapts that idea to robustness prediction: it summarizes token-level
uncertainty with 14 aggregate features and trains a regressor to predict
`absolute_accuracy_decay`. It is a simpler related baseline, not a reproduction
of the paper's temporal trace-profile features such as slope and linearity.

## Submission layout

```text
uncertainty-profiling/
├── solution.py
├── scripts/
│   ├── compute_uncertainty_features.py
│   └── train_uncertainty_regressor.py
├── uncertainty_profile/
│   ├── artifact.py
│   ├── config.py
│   ├── extraction.py
│   ├── inference.py
│   └── metrics.py
└── uncertainty_artifacts/
    └── deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.joblib
```

`solution.py` defines the required
`are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]` entry point. When
Codabench supplies `qwen3-8b:low`, the runtime resolves it to the cached
`deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` checkpoint before artifact lookup and
model loading. If no supported alias or matching artifact exists, the solution
returns `False` for every problem so the batch remains valid. Codabench supplies
only the original problem strings; dataset-side `permutation_type` metadata is
not provided to the regressor.

## Inference

The fitted artifact records and enforces the inference configuration:

- Codabench model alias: `qwen3-8b:low`;
- Hugging Face checkpoint:
  [`deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B);
- user-only chat-template formatting;
- greedy decoding with at most 2,048 prompt and 4,096 generated tokens;
- batch size 8 with a static KV cache;
- Min-K fraction 0.20 and high/low confidence cutoffs 0.90/0.10.

For each generated response, the runtime computes mean probability and
log-probability, NLL, perplexity, log-probability range and standard deviation,
Min-K log-probability, entropy, top-1 probability, top-2 margin, selected-token
top-1 frequency, and high/low confidence token frequencies. The fitted
regressor predicts `absolute_accuracy_decay`; a score below the artifact's
decision threshold maps to `True` (robust).

The evaluation path uses `local_files_only=True`, performs no training, and
does not require the Hugging Face dataset or network access.

## Reproduce the artifact

From the repository root, first generate the reusable feature cache:

```bash
uv run solutions/uncertainty-profiling/scripts/compute_uncertainty_features.py \
  --cache-dir /path/to/huggingface/cache
```

The command pins
[`aimo-interp/augmented-sample-math-agg@f972ced0705096f8d7ca7fac30825900b8b7fb6a`](https://huggingface.co/datasets/aimo-interp/augmented-sample-math-agg/tree/f972ced0705096f8d7ca7fac30825900b8b7fb6a)
and writes its intermediate parquet under the ignored
`data/uncertainty-profiling/` directory. A `.partial.parquet` file is updated
after every generated batch, so an interrupted run resumes without regenerating
completed prompts. To use several GPUs without duplicating prompts, run
one process per GPU with the same `--num-shards N` and distinct
`--shard-index` values; the prompt-level partition is deterministic. Pass the
output directory, rather than a single parquet file, to the training command.

Then perform grouped cross-validation and export the winner:

```bash
uv run solutions/uncertainty-profiling/scripts/train_uncertainty_regressor.py \
  --feature-data-path \
  data/uncertainty-profiling/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B_confidence_features.parquet
```

The training script uses five-fold `StratifiedGroupKFold`, grouping identical
problem texts together. It selects among Random Forest, Extra Trees, and
Histogram Gradient Boosting regressors using out-of-fold balanced accuracy,
then refits the selected pipeline on all feature rows. The artifact includes
the feature order, generation settings, decision threshold, selected
hyperparameters, complete CV summary, source revision, and package versions.

## Validate and package

```bash
uv run python -m unittest discover -s tests -v
uv run scripts/run_local.py solutions/uncertainty-profiling \
  --input-dir data/val-sample/input \
  --reference-dir data/val-sample/reference
uv run scripts/build.py components
```

The build output is `dist/solution-uncertainty-profiling.zip`, with
`solution.py` at the archive root. The ZIP contains the fitted regressor but no
training rows, generated responses, or language-model weights.

## Exported artifact

The checked-in artifact selected Random Forest with 500 estimators, unlimited
depth, `min_samples_leaf=1`, and `max_features=1.0`. Its grouped out-of-fold
decision threshold is `0.5760`; balanced accuracy is `0.6262`,
plain accuracy is `0.6879`, and regression MAE is `0.3429`.
