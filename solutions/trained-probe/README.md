# Trained Probe Baseline

This baseline is the submission-side inference wrapper for the
representation-based baseline in the `codex/simple-encode-probe-guide` branch
of `https://github.com/aimo-interp/baselines`.

The implementation is split by responsibility:

- `solution.py` defines `are_robust(model_id, reasoning_effort, problems: list[str])`.
- `probe_inference.py` loads artifacts and models, extracts hidden states, and
  scores the probe ensemble.
- `probe_artifacts/` contains the exported pretrained artifact when packaging a
  real submission.

The baseline repo trains probes from final input-token hidden states. This
Codabench solution expects the final exported pickle artifact:

```text
results/<run-name>/probe_artifacts/probe_artifact.pkl
```

Copy it next to `solution.py` as `probe_artifact.pkl`, or keep it at
`probe_artifacts/probe_artifact.pkl`. A model-specific
`probe_artifacts/<safe-model-id>.pkl` is also supported. No artifact is checked
in here. When no matching artifact is present, the solution returns `False` for
every case so it remains a valid build-time baseline.

The expected pickle schema is version 2 from the baseline branch:

- `artifact_type`: `all_folds_layers_seed_probe_ensemble`
- `model_id`: Hugging Face model id used to extract hidden states
- `system_prompt`: prompt prefix used during encoding
- `best_layer_index`: globally selected layer across validation metrics
- `recommended_strategy.name`: `best_layer_mean_margin`
- `groups[*]["probes"][layer]["weights"]`: plain Python list with shape
  `(n_seeds, hidden_dim)`
- `groups[*]["probes"][layer]["bias"]`: plain Python list with shape
  `(n_seeds,)`
- `groups[*]["probes"][layer]["threshold"]`: plain Python list with shape
  `(n_seeds,)`

`<safe-model-id>` replaces `/`, `:`, and whitespace with `_`, so
`qwen3-8b:low` maps to `probe_artifacts/qwen3-8b_low.pkl`.

The inference path mirrors the baseline encoder: build a prompt, run a single
forward pass with `output_hidden_states=True`, take the final prompt-token
hidden state from `best_layer_index`, compute all stored seed/fold probe
margins for that layer, and return `mean(score - threshold) >= 0`.
At inference, the list-backed probe arrays are converted with
`np.asarray(..., dtype=np.float32)`.

The submission exposes the batched contract
`are_robust(model_id, reasoning_effort, problems) -> list[bool]`. It preserves problem order and
returns one prediction per input problem.

Single-probe `.npz` artifacts from the earlier wrapper are still accepted as a
compatibility fallback, but the pickle ensemble is the expected format.
