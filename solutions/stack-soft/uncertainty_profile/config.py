"""Shared feature ordering and generation configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


ConfigValue = int | float | bool | str

DEEPSEEK_MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
COMPETITION_MODEL_ALIASES: dict[str, str] = {
    "qwen3-8b:low": DEEPSEEK_MODEL_ID,
}

FEATURE_NAMES: tuple[str, ...] = (
    "generation_mean_logprob",
    "generation_mean_prob",
    "generation_mean_nll",
    "generation_ppl",
    "generation_min_logprob",
    "generation_max_logprob",
    "generation_std_logprob",
    "generation_min_k_logprob",
    "generation_mean_entropy",
    "generation_mean_top1_prob",
    "generation_mean_top2_margin",
    "generation_frac_selected_is_top1",
    "generation_frac_high_conf_tokens",
    "generation_frac_low_conf_tokens",
)


def resolve_checkpoint_model_id(model_id: str) -> str:
    """Resolve a Codabench model alias to its Hugging Face checkpoint.

    Args:
        model_id: Codabench model identifier or exact checkpoint identifier.

    Returns:
        Exact Hugging Face checkpoint identifier when an alias is known,
        otherwise ``model_id`` unchanged.
    """

    return COMPETITION_MODEL_ALIASES.get(model_id, model_id)


@dataclass(frozen=True)
class GenerationConfidenceConfig:
    """Configure deterministic generation and uncertainty features.

    Attributes:
        max_prompt_length: Maximum number of input tokens retained per prompt.
        max_new_tokens: Maximum number of tokens generated per response.
        batch_size: Number of prompts generated in each inference batch.
        cache_implementation: Transformers key-value cache implementation.
        chat_template_role: Role used for the single chat-template message.
        do_sample: Whether generation samples rather than decodes greedily.
        min_k_fraction: Fraction of least-likely tokens used by Min-K.
        high_conf_threshold: Inclusive high-confidence token threshold.
        low_conf_threshold: Inclusive low-confidence token threshold.
    """

    max_prompt_length: int = 2048
    max_new_tokens: int = 4096
    batch_size: int = 8
    cache_implementation: str = "static"
    chat_template_role: str = "user"
    do_sample: bool = False
    min_k_fraction: float = 0.20
    high_conf_threshold: float = 0.90
    low_conf_threshold: float = 0.10

    def validate(self) -> None:
        """Validate every configuration value.

        Raises:
            ValueError: If a value is invalid or would make generation
                nondeterministic.
        """

        if type(self.max_prompt_length) is not int or self.max_prompt_length < 1:
            raise ValueError("max_prompt_length must be a positive integer")
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if self.cache_implementation != "static":
            raise ValueError("cache_implementation must be 'static'")
        if self.chat_template_role != "user":
            raise ValueError("chat_template_role must be 'user'")
        if type(self.do_sample) is not bool or self.do_sample:
            raise ValueError(
                "uncertainty profiling requires deterministic greedy decoding"
            )
        if (
            isinstance(self.min_k_fraction, bool)
            or not isinstance(self.min_k_fraction, (int, float))
            or not 0.0 < self.min_k_fraction <= 1.0
        ):
            raise ValueError("min_k_fraction must be in (0, 1]")
        if (
            isinstance(self.low_conf_threshold, bool)
            or not isinstance(self.low_conf_threshold, (int, float))
            or not 0.0 <= self.low_conf_threshold <= 1.0
        ):
            raise ValueError("low_conf_threshold must be in [0, 1]")
        if (
            isinstance(self.high_conf_threshold, bool)
            or not isinstance(self.high_conf_threshold, (int, float))
            or not 0.0 <= self.high_conf_threshold <= 1.0
        ):
            raise ValueError("high_conf_threshold must be in [0, 1]")

    def to_dict(self) -> dict[str, ConfigValue]:
        """Return a validated, JSON-serializable configuration mapping.

        Returns:
            Configuration values keyed by dataclass field name.

        Raises:
            ValueError: If the configuration is invalid.
        """

        self.validate()
        return {
            "max_prompt_length": self.max_prompt_length,
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
            "cache_implementation": self.cache_implementation,
            "chat_template_role": self.chat_template_role,
            "do_sample": self.do_sample,
            "min_k_fraction": self.min_k_fraction,
            "high_conf_threshold": self.high_conf_threshold,
            "low_conf_threshold": self.low_conf_threshold,
        }

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, object],
    ) -> "GenerationConfidenceConfig":
        """Construct and validate a configuration from artifact metadata.

        Args:
            values: Mapping containing exactly the persisted configuration keys.

        Returns:
            Validated generation configuration.

        Raises:
            ValueError: If keys or values do not match the configuration schema.
        """

        if not isinstance(values, Mapping):
            raise ValueError("generation_config must be a mapping")

        expected = set(cls.__dataclass_fields__)
        if set(values) != expected:
            missing = sorted(expected - set(values))
            extra = sorted(set(values) - expected)
            raise ValueError(
                f"invalid generation_config keys; missing={missing}, extra={extra}"
            )

        config = cls(**dict(values))  # type: ignore[arg-type]
        config.validate()
        return config
