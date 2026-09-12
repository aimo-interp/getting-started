# AIMO Interpretability Challenge

This repo contains a guideline for developing and submitting a method into the [AIMO Interpretability Challenge](https://aimo-interp.github.io/). It uses the
**representation classifier probe** as a complete, working example.

## Contents

- [What your method does](#what-your-method-does)
- [Choose a track](#choose-a-track)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Which data is available?](#which-data-is-available)
- [Example method: trained representation probe](#example-method-trained-representation-probe)
- [Example method: uncertainty profiling](#example-method-uncertainty-profiling)
- [Runtime and available libraries](#runtime-and-available-libraries)
- [Evaluation environment](#evaluation-environment)
- [What to submit](#what-to-submit)
- [Check the submission locally](#check-the-submission-locally)
- [How to submit](#how-to-submit)
- [Repository layout](#repository-layout)

## What your method does

For each model and batch of mathematical problems, your method predicts whether
the model's behavior is robust:

```python
are_robust(model_id: str, problems: list[str]) -> list[bool]
```

Your submission must expose a callable named `are_robust` with these arguments.
The annotations are recommended for clarity but are not required at runtime.
The returned list must:

- contain exactly one prediction per input problem;
- preserve the input order;
- contain real Python `bool` values, not integers or NumPy boolean scalars;
- use `True` for robust and `False` for not robust.

Your method can load the specified language model (`model_id`) through Hugging
Face, run inference on the supplied `problems`, and predict whether the model is
robust on each problem. You may use or extend any available Hugging Face
functionality and combine signals from the forward pass to produce the
robustness predictions.

Model weights for every evaluated `model_id` are available through the
evaluation worker's offline Hugging Face cache; see
[Evaluation environment](#evaluation-environment) for the list. As the evaluation runs offline, you will not have
access to other HF models.

## Choose a track

The competition has a **Main Track** and a **Small Model Track**. Select exactly
one task when uploading each submission.

- A main-track ZIP must not contain `small.txt`.
- A small-model-track ZIP must contain `small.txt` at its root.
- The marker name is case-insensitive and its contents are ignored. A missing or
  unexpected marker fails the submission before `solution.py` is imported.

Main-track scores are reported as `accuracy`, `coverage`, and
`invalid_predictions`. Small-model-track scores use `accuracy_small`,
`coverage_small`, and `invalid_predictions_small`. Completed submissions are
added to the shared leaderboard automatically, so several entries from the same
participant may appear.

## Prerequisites

You need:

- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) for the local tooling;
- `git` to clone the repository;
- Docker, if you want to reproduce the offline evaluation container locally
  (recommended before submitting).

Clone the repository and install dependencies:

```bash
git clone git@github.com:aimo-interp/getting-started.git
cd getting-started
uv sync
```

## Quick start

The fastest path from a fresh clone to a scored local run:

```bash
# 1. Import the public validation sample into data/val-sample/.
uv run scripts/import_hf_dataset.py

# 2. Score the example method on that sample.
uv run scripts/run_local.py solutions/trained-probe \
  --input-dir data/val-sample/input \
  --reference-dir data/val-sample/reference
```

Without a trained probe artifact, the example predicts `False` for every case
(a valid but trivial baseline). The rest of this guide explains how to plug in a
real artifact, package a submission, and validate it. To write your own method,
copy `solutions/trained-probe/` and replace the implementation behind
`are_robust`.

## Which data is available?

### Public development data

The public validation sample is
[`aimo-interp/val-sample`](https://huggingface.co/datasets/aimo-interp/val-sample).
It contains examples with:

- `model_id`: the model whose robustness is being evaluated;
- `original_problem`: the mathematical problem text;
- `permutation_type`: the relevant perturbation categories;
- `model_is_robust`: the public development label.

Import it into this repository with:

```bash
uv run scripts/import_hf_dataset.py
```

This creates:

```text
data/val-sample/
├── input/
│   └── cases.jsonl
├── reference/
│   └── labels.jsonl
└── metadata.json
```

The public labels are for development only. The final evaluation uses separate
private problems and labels.

### Data available during evaluation

For each call, the trained probe receives a `model_id` and a list of `str`.

Your submission does not receive case IDs, reference labels, private dataset
files, or network access.

## Example method: trained representation probe

This baseline demonstrates how to compose a solution.

Start with this directory:

```text
trained-probe/
├── solution.py
├── probe_inference.py
└── probe_artifacts/
    └── probe_artifact.pkl
```

The `solution.py` file defines only the Codabench interface:

```python
from probe_inference import predict_robustness

def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    return predict_robustness(model_id, problems)
```

`probe_inference.py` contains the implementation in three sections:

1. **Artifact loading:** find and validate `probe_artifact.pkl`.
2. **Hidden-state extraction:** load the offline model, format the problem,
   perform a forward pass, and select the final prompt-token representation
   from the model layer.
3. **Probe scoring:** apply a collection of linear probes and average their
   margins.

For one probe, the decision is:

```python
margin = weights @ hidden_state + bias - threshold
prediction = margin >= 0
```

The ensemble averages these margins and returns `True` when the mean is
non-negative. The artifact stores weights, biases, thresholds, the selected
layer, the Hugging Face model ID, and the system prompt. It does not contain
precomputed hidden states.

## Example method: uncertainty profiling

The repository also provides `solutions/uncertainty-profiling/`. This baseline
generates a deterministic response to each original problem and summarizes the
generated-token distributions with 14 confidence and uncertainty statistics.
A pretrained scikit-learn regressor maps those features to predicted absolute
accuracy decay; the artifact's cross-validated threshold converts the score to
the required robustness boolean.

The motivation is to measure robustness through uncertainty over the model's
generated reasoning trace: robust and spurious reasoning may produce different
token-level uncertainty signatures. This follows the general perspective of
[Tracing Uncertainty in Language Model "Reasoning"](https://arxiv.org/abs/2605.07776),
while using a simpler set of 14 aggregate trace features.

Training is deliberately separate from submission inference. Generate a
resumable feature cache from the pinned public aggregate dataset:

```bash
uv run solutions/uncertainty-profiling/scripts/compute_uncertainty_features.py \
  --cache-dir /path/to/huggingface/cache
```

Then select the regressor and threshold with grouped cross-validation, refit on
all feature rows, and export the joblib artifact directly into the solution:

```bash
uv run solutions/uncertainty-profiling/scripts/train_uncertainty_regressor.py \
  --feature-data-path \
  data/uncertainty-profiling/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B_confidence_features.parquet
```

The resulting solution is offline and self-contained. During evaluation it
resolves the Codabench alias `qwen3-8b:low` to the cached
`deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` checkpoint, then loads the matching
bundled regressor artifact. It uses greedy decoding with a 2,048-token prompt
cap and a 4,096-token response cap. See
`solutions/uncertainty-profiling/README.md` for the complete feature list,
artifact provenance, and CV search.

## Runtime and available libraries

The evaluation container is offline. It does not run `pip install`, `uv sync`,
or a submitted `requirements.txt`.

**You may rely only on the Python standard library and packages provided by
[`Dockerfile.competition`](Dockerfile.competition).** The current runtime
provides:

- PyTorch `2.12.1` with CUDA `13.2`;
- Accelerate `1.13.0`;
- Hugging Face Hub `1.22.0`;
- joblib `1.5.3`;
- pandas `3.0.3`;
- safetensors `0.8.0`;
- scikit-learn `1.8.0`;
- tokenizers `0.22.2`;
- Transformers `5.13.0`.

Important consequences:

- A package installed only on your laptop or cluster login environment will not
  exist during evaluation.
- Adding `requirements.txt` to the ZIP does not install anything.
- Artifacts serialized with scikit-learn or joblib should be trained with the
  runtime versions above.
- Model and tokenizer loading must work offline, so rely only on the model ID
  passed to `are_robust`. Do not call external APIs or download from Hugging
  Face during evaluation.
- You may include your own Python source files and pretrained data artifacts in
  the ZIP, such as pretrained probes, provided they comply with the competition
  rules.

## Evaluation environment

Each submission is evaluated on a dedicated GPU worker with the following
resources:

- **GPU:** one NVIDIA RTX PRO 6000 (Blackwell, compute capability 12.0) with
  96 GB of VRAM. A single card is exposed to the evaluation container, so
  `torch.cuda.device_count()` returns 1 and `device_map="auto"` places the whole
  model on that card. Every evaluated checkpoint fits, `openai/gpt-oss-120b`
  included. The submissions share the compute environment, so the free VRAM at
  any moment may be less than the nominal total.
- **CPU and RAM:** 192 cores with ample system memory; there is no per-container
  memory cap.
- **Time limit:** 3600 seconds per evaluation run, applied separately to the
  prediction run and the scoring run. The limit covers the *entire* prediction
  run -- every model batch and every problem together, not one hour per problem.
  Loading a cached 4-8B checkpoint takes about two seconds from the warm cache,
  and `openai/gpt-oss-120b` about four.
- **Network:** disabled inside evaluation containers. `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1` are preset.
- **Model cache:** a read-only Hugging Face cache is mounted at
  `/app/data/huggingface`, and `HF_HOME` already points to it, so
  `from_pretrained(...)` resolves cached models with no extra configuration.
  Pass `local_files_only=True` to fail fast on anything not cached.

Models evaluated in the Competition phase, and cached on the workers:

| `model_id` | Main Track | Small Models Track |
| --- | :---: |:------------------:|
| `Qwen/Qwen3.5-4B` | yes |        yes         |
| `Skywork/Skywork-OR1-Math-7B` | yes |        yes         |
| `allenai/Olmo-3-7B-Think` | yes |        yes         |
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | yes |        yes         |
| `openai/gpt-oss-120b` | yes |         no          |

`are_robust` receives these identifiers verbatim. There are no aliases to
resolve: the `:low`-suffixed names and the closed API models that appear in the
public sample datasets are never passed during the Competition phase, so every
evaluated case is a local open-weights checkpoint.

The warm-up phase evaluated a different set, which included
`allenai/Olmo-3-7B-Instruct` and `google/gemma-3-27b-it`. Neither is used in the
Competition phase, and `allenai/Olmo-3-7B-Think` is a distinct checkpoint from
`allenai/Olmo-3-7B-Instruct`.

Every `model_id` the harness passes to `are_robust` is cached, so a method can
always load the model it is asked to judge. All of them load onto the single
visible GPU with `device_map="auto"` and `local_files_only=True`.
If you encounter an OOM error, it may be a clash with another process on our Codabench server
-- please try resubmitting and if the problem persists, please let us know.

`openai/gpt-oss-120b` is cached in its upstream MXFP4 form, and the runtime
image ships the Triton kernels needed to run it natively, so
`from_pretrained("openai/gpt-oss-120b", local_files_only=True)` yields a model
of roughly 62 GiB that fits on the single card with headroom to spare.

Models outside this list cannot be loaded during evaluation. Contact the
organizers if your method needs another model cached on the workers.

## What to submit

Submit one ZIP archive containing:

- `solution.py` at the archive root;
- a root-level `small.txt` only when submitting to the Small Model Track (its
  contents are ignored);
- any helper modules or Python packages imported by `solution.py`;
- all pretrained artifacts needed, such as the trained probes;
- optional configuration files.

Do not submit:

- training datasets or private labels;
- model weights already supplied by the evaluation worker;
- virtual environments, caches, `__pycache__`, or installed packages;
- credentials, API keys, or code that requires network access.

Correct ZIP layout, for the probing example:

```text
baseline-submission-bundle.zip
├── solution.py
├── probe_inference.py
└── probe_artifacts/
    └── probe_artifact.pkl
```

Correct Small Model Track layout:

```text
baseline-submission-bundle-small.zip
├── solution.py
├── small.txt
├── probe_inference.py
└── probe_artifacts/
    └── probe_artifact.pkl
```

Incorrect ZIP layout:

```text
baseline-submission-bundle.zip
└── trained-probe/
    └── solution.py
```

Build the archive from inside the method directory:

```bash
cd trained-probe
zip -r ../baseline-submission-bundle.zip solution.py probe_inference.py probe_artifacts
cd ..
```

Verify that `solution.py` is at the root:

```bash
unzip -l baseline-submission-bundle.zip
```

For any method under `solutions/`, the build helper can also create both track
variants without modifying the source directory:

```bash
uv run scripts/build.py solutions
uv run scripts/build.py solutions --small
```

This creates a solution ZIP. For the probing example, the main-track output is
`dist/solution-trained-probe.zip` and the small-track output is
`dist/solution-trained-probe-small.zip`.

## Check the submission locally

The repository already includes the method under `solutions/trained-probe/`.
Copy the exported artifact into it:

```text
solutions/
└── trained-probe/
    ├── solution.py
    ├── probe_inference.py
    └── probe_artifacts/
        └── probe_artifact.pkl
```

```bash
mkdir -p solutions/trained-probe/probe_artifacts
cp /path/to/probe_artifact.pkl \
  solutions/trained-probe/probe_artifacts/probe_artifact.pkl
```

Run the public validation sample:

```bash
uv run scripts/run_local.py solutions/trained-probe \
  --input-dir data/val-sample/input \
  --reference-dir data/val-sample/reference
```

Expected output has this shape:

```json
{
  "accuracy": 0.7,
  "coverage": 1.0,
  "invalid_predictions": 0
}
```

The accuracy shown above is only an example. Before submission, require:

- `coverage` equal to `1.0`;
- `invalid_predictions` equal to `0`;
- deterministic predictions across repeated runs;
- successful execution in the competition Docker image, not only in a local
  virtual environment.

To check the track-specific packaging guards and score names separately, run:

```bash
uv run scripts/build.py solutions
uv run scripts/build.py solutions --small
uv run scripts/run_local.py dist/solution-always-true.zip
uv run scripts/run_local.py dist/solution-always-true-small.zip --small
```

The small-track run reports:

```json
{
  "accuracy_small": 0.375,
  "coverage_small": 1.0,
  "invalid_predictions_small": 0
}
```

The public sample is a contract and packaging check; it is not a preview of the
private track scores.

## How to submit

1. Sign in to the AIMO Interpretability Challenge on [Codabench](https://www.codabench.org/competitions/16180/#/pages-tab).
2. Open the active submission phase and select exactly one task.
3. For the Main Track, upload a ZIP without `small.txt`, such as
   `dist/solution-trained-probe.zip`.
4. For the Small Model Track, upload the corresponding ZIP with the marker,
   such as `dist/solution-trained-probe-small.zip`.
5. Wait for ingestion and scoring to finish.
6. Check the track's coverage and invalid-prediction columns before interpreting
   its accuracy.

An invalid prediction usually means the method raised an exception, returned a
non-boolean value, or returned the wrong number of predictions. Detailed
participant exceptions may be hidden by the evaluator, so reproduce failures
locally with the same Docker image.


### Repository layout

```text
components/             the Codabench ingestion and scoring programs, used by run_local.py
data/val-sample/        generated public validation import (ignored)
scripts/                dataset importer, local runner, and archive builder
solutions/              example methods, including the trained probe
Dockerfile.competition  the evaluation runtime (exact package versions)
pyproject.toml          local dependencies mirroring the runtime versions
```
