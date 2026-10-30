"""Offline model loading and memory-conscious uncertainty feature extraction."""

from __future__ import annotations

import gc
import sys
from collections.abc import Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from tqdm.auto import tqdm

from uncertainty_profile.config import GenerationConfidenceConfig
from uncertainty_profile.metrics import (
    ConfidenceMetrics,
    compute_generation_confidence_metrics,
)


FeatureValue = float | int | str
FeatureRow = dict[str, FeatureValue]


def load_model_and_tokenizer(
    model_id: str,
    *,
    cache_dir: str | None = None,
    local_files_only: bool,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal language model and compatible tokenizer.

    Args:
        model_id: Hugging Face model identifier or local model path.
        cache_dir: Optional Hugging Face cache directory.
        local_files_only: Whether loading must avoid all network access.

    Returns:
        Evaluation-mode model and left-padding tokenizer.

    Raises:
        OSError: If the requested model or tokenizer cannot be loaded.
        RuntimeError: If the tokenizer has neither a padding nor an EOS token.
    """

    tokenizer_kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "trust_remote_code": False,
    }
    try:
        # The pinned runtime otherwise chooses a legacy Llama conversion that
        # does not round-trip this checkpoint's byte-level BPE vocabulary.
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            model_id,
            **tokenizer_kwargs,
        )
    except (OSError, TypeError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            **tokenizer_kwargs,
        )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(f"tokenizer for {model_id!r} has no pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            dtype=dtype,
            device_map="auto",
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            dtype=torch.float32,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        model.to("cpu")
    model.eval()
    return model, tokenizer


def release_model() -> None:
    """Collect dereferenced model resources and empty the CUDA allocator cache."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_model_input_device(model: PreTrainedModel) -> torch.device:
    """Determine the device on which model inputs must be placed.

    Args:
        model: Loaded causal language model.

    Returns:
        Device holding the input embeddings, or the first model parameter.
    """

    embeddings = model.get_input_embeddings()
    if embeddings is not None:
        return embeddings.weight.device
    return next(model.parameters()).device


def format_problem(
    tokenizer: PreTrainedTokenizerBase,
    problem_text: str,
    *,
    role: str = "user",
) -> str:
    """Apply a user-only chat template, falling back to plain text.

    Args:
        tokenizer: Tokenizer supplying the model chat template.
        problem_text: Original mathematical problem statement.
        role: Role assigned to the single chat message.

    Returns:
        Formatted prompt text, or ``problem_text`` if no compatible template is
        available.
    """

    try:
        formatted = tokenizer.apply_chat_template(
            [{"role": role, "content": problem_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except (AttributeError, TypeError, ValueError):
        return problem_text
    return formatted if isinstance(formatted, str) else problem_text


def _token_id_set(value: int | Sequence[int] | None) -> set[int]:
    """Normalize a scalar or sequence of token IDs into a set.

    Args:
        value: Token ID value from a tokenizer or generation configuration.

    Returns:
        Integer token IDs, or an empty set for ``None``.
    """

    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {int(item) for item in value}
    return {int(value)}


class GenerationConfidenceCollector(LogitsProcessor):
    """Accumulate generation statistics without retaining vocabulary logits.

    The collector receives one vocabulary-sized score tensor at a time from
    ``generate``. It immediately reduces that tensor to per-response scalars and
    retains only one step of log-probabilities until the selected token is known.

    Attributes:
        selected_log_probs: Selected-token log-probability tensors by step.
        entropy: Predictive entropy tensors by step.
        top1_probs: Most-likely-token probability tensors by step.
        top2_margins: Top-one minus top-two probability tensors by step.
        top1_token_ids: Most-likely token IDs by step.
        pending_log_probs: Previous step retained until its token is selected.
    """

    def __init__(self) -> None:
        """Initialize empty online generation statistics."""

        self.selected_log_probs: list[torch.Tensor] = []
        self.entropy: list[torch.Tensor] = []
        self.top1_probs: list[torch.Tensor] = []
        self.top2_margins: list[torch.Tensor] = []
        self.top1_token_ids: list[torch.Tensor] = []
        self.pending_log_probs: torch.Tensor | None = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Reduce the current generation scores and retain them for one step.

        Args:
            input_ids: Input and generated token IDs available before selection.
            scores: Next-token vocabulary scores for each response in the batch.

        Returns:
            Unmodified ``scores`` so greedy generation semantics are preserved.
        """

        if self.pending_log_probs is not None:
            previous_token_ids = input_ids[:, -1].to(self.pending_log_probs.device)
            self.selected_log_probs.append(
                self.pending_log_probs
                .gather(-1, previous_token_ids.unsqueeze(-1))
                .squeeze(-1)
                .detach()
            )

        log_probs = functional.log_softmax(scores.float(), dim=-1)
        probabilities = log_probs.exp()
        entropy_terms = torch.nan_to_num(
            probabilities * log_probs,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        top2_log_probs, top2_token_ids = log_probs.topk(k=2, dim=-1)
        top2_probs = top2_log_probs.exp()

        self.pending_log_probs = log_probs.detach()
        self.entropy.append((-entropy_terms.sum(dim=-1)).detach())
        self.top1_probs.append(top2_probs[:, 0].detach())
        self.top2_margins.append((top2_probs[:, 0] - top2_probs[:, 1]).detach())
        self.top1_token_ids.append(top2_token_ids[:, 0].detach())
        return scores

    def compute_features(
        self,
        *,
        sequences: torch.Tensor,
        prompt_length: int,
        pad_token_id: int | None,
        eos_token_ids: set[int],
        config: GenerationConfidenceConfig,
    ) -> tuple[list[ConfidenceMetrics], torch.Tensor]:
        """Finalize selected-token values and compute batch feature rows.

        Args:
            sequences: Complete prompt and generated token sequences.
            prompt_length: Padded prompt length shared by the batch.
            pad_token_id: Token ID excluded as generation padding.
            eos_token_ids: EOS token IDs excluded from feature summaries.
            config: Uncertainty metric configuration.

        Returns:
            Per-response scalar metrics and generated token IDs on CPU.

        Raises:
            RuntimeError: If generation produced no statistics or step alignment
                is inconsistent.
        """

        if not self.top1_probs or self.pending_log_probs is None:
            raise RuntimeError("generation produced no token statistics")

        num_steps = len(self.top1_probs)
        final_token_ids = sequences[:, prompt_length + num_steps - 1].to(
            self.pending_log_probs.device
        )
        self.selected_log_probs.append(
            self.pending_log_probs
            .gather(-1, final_token_ids.unsqueeze(-1))
            .squeeze(-1)
            .detach()
        )
        self.pending_log_probs = None
        if len(self.selected_log_probs) != num_steps:
            raise RuntimeError("generated token statistics are misaligned")

        generated_token_ids = sequences[
            :, prompt_length : prompt_length + num_steps
        ].detach().cpu()
        selected_log_probs = torch.stack(self.selected_log_probs, dim=1).float().cpu()
        selected_probs = selected_log_probs.exp()
        entropy = torch.stack(self.entropy, dim=1).float().cpu()
        top1_probs = torch.stack(self.top1_probs, dim=1).float().cpu()
        top2_margins = torch.stack(self.top2_margins, dim=1).float().cpu()
        top1_token_ids = torch.stack(self.top1_token_ids, dim=1).cpu()
        selected_is_top1 = generated_token_ids == top1_token_ids
        valid_mask = build_valid_token_mask(
            generated_token_ids,
            pad_token_id=pad_token_id,
            eos_token_ids=eos_token_ids,
        )

        rows: list[ConfidenceMetrics] = []
        for batch_index in range(generated_token_ids.shape[0]):
            mask = valid_mask[batch_index]

            def masked_numpy(values: torch.Tensor) -> np.ndarray:
                """Select valid values for the current response as an array."""

                return values[batch_index][mask].numpy()

            rows.append(
                compute_generation_confidence_metrics(
                    log_probs=masked_numpy(selected_log_probs),
                    probs=masked_numpy(selected_probs),
                    entropy=masked_numpy(entropy),
                    top1_probs=masked_numpy(top1_probs),
                    top2_margins=masked_numpy(top2_margins),
                    selected_is_top1=masked_numpy(selected_is_top1),
                    min_k_fraction=config.min_k_fraction,
                    high_conf_threshold=config.high_conf_threshold,
                    low_conf_threshold=config.low_conf_threshold,
                )
            )
        return rows, generated_token_ids


def build_valid_token_mask(
    generated_token_ids: torch.Tensor,
    *,
    pad_token_id: int | None,
    eos_token_ids: set[int],
) -> torch.Tensor:
    """Mask padding and EOS tokens from generated-token summaries.

    Args:
        generated_token_ids: Batch of generated token IDs.
        pad_token_id: Padding token ID, if configured.
        eos_token_ids: Every EOS token ID used by the model or tokenizer.

    Returns:
        Boolean tensor with ``True`` exactly for summarized tokens.
    """

    mask = torch.ones_like(generated_token_ids, dtype=torch.bool)
    if pad_token_id is not None:
        mask &= generated_token_ids != int(pad_token_id)
    for token_id in sorted(eos_token_ids):
        mask &= generated_token_ids != token_id
    return mask


def compute_batch_features_from_scores(
    *,
    sequences: torch.Tensor,
    scores: Sequence[torch.Tensor],
    prompt_length: int,
    pad_token_id: int | None,
    eos_token_ids: set[int],
    config: GenerationConfidenceConfig,
) -> tuple[list[ConfidenceMetrics], torch.Tensor]:
    """Compute batch features from retained per-step scores.

    This compatibility helper is used to verify that the online collector
    preserves the original feature values. Runtime generation uses
    :class:`GenerationConfidenceCollector` instead.

    Args:
        sequences: Complete prompt and generated token sequences.
        scores: Per-step vocabulary score tensors.
        prompt_length: Padded prompt length shared by the batch.
        pad_token_id: Token ID excluded as generation padding.
        eos_token_ids: EOS token IDs excluded from feature summaries.
        config: Uncertainty metric configuration.

    Returns:
        Per-response scalar metrics and generated token IDs on CPU.

    Raises:
        RuntimeError: If no score tensors were supplied.
    """

    if not scores:
        raise RuntimeError("generation produced no score tensors")

    generated_token_ids = sequences[:, prompt_length : prompt_length + len(scores)]
    selected_log_probs_steps: list[torch.Tensor] = []
    entropy_steps: list[torch.Tensor] = []
    top1_prob_steps: list[torch.Tensor] = []
    top2_margin_steps: list[torch.Tensor] = []
    selected_is_top1_steps: list[torch.Tensor] = []

    for step_index, step_scores in enumerate(scores):
        logits = step_scores.float()
        log_probs = functional.log_softmax(logits, dim=-1)
        token_ids = generated_token_ids[:, step_index].to(log_probs.device)
        selected_log_probs = log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        probabilities = log_probs.exp()
        entropy_terms = torch.nan_to_num(
            probabilities * log_probs,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        entropy = -entropy_terms.sum(dim=-1)
        top2_log_probs, top2_token_ids = log_probs.topk(k=2, dim=-1)
        top2_probs = top2_log_probs.exp()

        selected_log_probs_steps.append(selected_log_probs.cpu())
        entropy_steps.append(entropy.cpu())
        top1_prob_steps.append(top2_probs[:, 0].cpu())
        top2_margin_steps.append((top2_probs[:, 0] - top2_probs[:, 1]).cpu())
        selected_is_top1_steps.append((token_ids == top2_token_ids[:, 0]).cpu())

        del logits, log_probs, probabilities, entropy_terms, top2_log_probs, top2_probs

    selected_log_probs = torch.stack(selected_log_probs_steps, dim=1)
    selected_probs = selected_log_probs.exp()
    entropy = torch.stack(entropy_steps, dim=1)
    top1_probs = torch.stack(top1_prob_steps, dim=1)
    top2_margins = torch.stack(top2_margin_steps, dim=1)
    selected_is_top1 = torch.stack(selected_is_top1_steps, dim=1)
    generated_token_ids_cpu = generated_token_ids.detach().cpu()
    valid_mask = build_valid_token_mask(
        generated_token_ids_cpu,
        pad_token_id=pad_token_id,
        eos_token_ids=eos_token_ids,
    )

    rows: list[ConfidenceMetrics] = []
    for batch_index in range(generated_token_ids_cpu.shape[0]):
        mask = valid_mask[batch_index]

        def masked_numpy(values: torch.Tensor) -> np.ndarray:
            """Select valid values for the current response as an array."""

            return values[batch_index][mask].detach().float().numpy()

        rows.append(
            compute_generation_confidence_metrics(
                log_probs=masked_numpy(selected_log_probs),
                probs=masked_numpy(selected_probs),
                entropy=masked_numpy(entropy),
                top1_probs=masked_numpy(top1_probs),
                top2_margins=masked_numpy(top2_margins),
                selected_is_top1=masked_numpy(selected_is_top1),
                min_k_fraction=config.min_k_fraction,
                high_conf_threshold=config.high_conf_threshold,
                low_conf_threshold=config.low_conf_threshold,
            )
        )
    return rows, generated_token_ids_cpu


@torch.inference_mode()
class GenerationProgress(LogitsProcessor):
    """Advance a progress bar once per generated token; scores pass through unchanged."""

    def __init__(self, bar: tqdm, prompt_length: int, eos_token_ids: set[int]) -> None:
        self.bar = bar
        self.prompt_length = prompt_length
        self.eos_token_ids = torch.tensor(sorted(eos_token_ids)) if eos_token_ids else None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # Each call selects one new token, including the first after the prompt.
        self.bar.update(1)
        if self.eos_token_ids is not None and self.bar.n % 64 == 0:
            generated = input_ids[:, self.prompt_length :]
            finished = torch.isin(generated, self.eos_token_ids.to(generated.device)).any(dim=1)
            self.bar.set_postfix(finished=f"{int(finished.sum())}/{input_ids.shape[0]}")
        return scores


def iter_generation_feature_batches(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    problem_texts: Sequence[str],
    config: GenerationConfidenceConfig,
    include_generated_text: bool = False,
) -> Iterator[tuple[int, list[FeatureRow]]]:
    """Yield ordered feature rows one inference batch at a time.

    Args:
        model: Loaded causal language model.
        tokenizer: Matching tokenizer configured for left padding.
        problem_texts: Problem statements in source order.
        config: Generation and uncertainty metric configuration.
        include_generated_text: Whether to decode and attach model responses.

    Yields:
        Pairs of the batch's start index and its feature rows.

    Raises:
        ValueError: If ``config`` is invalid.
        RuntimeError: If generated statistics are incomplete or misaligned.
    """

    config.validate()
    device = get_model_input_device(model)
    eos_value = getattr(model.generation_config, "eos_token_id", None)
    if eos_value is None:
        eos_value = tokenizer.eos_token_id
    eos_token_ids = _token_id_set(eos_value) | _token_id_set(tokenizer.eos_token_id)

    n_batches = -(-len(problem_texts) // config.batch_size)
    for start in range(0, len(problem_texts), config.batch_size):
        batch_texts = list(problem_texts[start : start + config.batch_size])
        batch_number = start // config.batch_size + 1
        formatted = [
            format_problem(tokenizer, text, role=config.chat_template_role)
            for text in batch_texts
        ]
        encoded = tokenizer(
            formatted,
            padding=True,
            truncation=True,
            max_length=config.max_prompt_length,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        prompt_length = int(encoded["input_ids"].shape[1])
        collector = GenerationConfidenceCollector()
        bar = tqdm(
            total=config.max_new_tokens,
            desc=(
                f"batch {batch_number}/{n_batches} "
                f"(problems {start + 1}-{start + len(batch_texts)} of {len(problem_texts)})"
            ),
            unit="tok",
            file=sys.stderr,
            mininterval=1.0 if sys.stderr.isatty() else 15.0,
        )
        progress = GenerationProgress(bar, prompt_length, eos_token_ids)
        try:
            outputs = model.generate(
                **encoded,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                return_dict_in_generate=True,
                output_scores=False,
                logits_processor=LogitsProcessorList([collector, progress]),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_value,
                use_cache=True,
                cache_implementation=config.cache_implementation,
            )
        finally:
            bar.close()
        metric_rows, generated_token_ids = collector.compute_features(
            sequences=outputs.sequences,
            prompt_length=prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_ids=eos_token_ids,
            config=config,
        )
        rows: list[FeatureRow] = [dict(row) for row in metric_rows]
        if include_generated_text:
            generated_texts = tokenizer.batch_decode(
                generated_token_ids,
                skip_special_tokens=True,
            )
            for row, generated_text in zip(rows, generated_texts):
                row["generated_text"] = generated_text
        yield start, rows
        del encoded, outputs, collector, generated_token_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def extract_generation_features(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    problem_texts: Sequence[str],
    config: GenerationConfidenceConfig,
    include_generated_text: bool = False,
) -> list[FeatureRow]:
    """Return ordered feature rows for all supplied problem texts.

    Args:
        model: Loaded causal language model.
        tokenizer: Matching tokenizer configured for left padding.
        problem_texts: Problem statements in source order.
        config: Generation and uncertainty metric configuration.
        include_generated_text: Whether to decode and attach model responses.

    Returns:
        Feature rows in the same order as ``problem_texts``.

    Raises:
        RuntimeError: If generation returns a different number of feature rows.
    """

    rows: list[FeatureRow] = []
    for _, batch_rows in iter_generation_feature_batches(
        model=model,
        tokenizer=tokenizer,
        problem_texts=problem_texts,
        config=config,
        include_generated_text=include_generated_text,
    ):
        rows.extend(batch_rows)
    if len(rows) != len(problem_texts):
        raise RuntimeError(
            f"generated {len(rows)} feature rows for {len(problem_texts)} prompts"
        )
    return rows
