"""Sample n reasoning traces per problem with Transformers, under a wall-clock limit.

Sampling matches the training transcripts: the model's own chat template, the
problem plus SOLVE, temperature 1.0, no top-p/top-k truncation. Problems are
batched as far as the KV cache fits. ``generate(max_time=...)`` stops a batch
at its deadline, so the number of tokens actually generated can fall short of
the requested budget; callers then score at the largest trained budget that
was reached.
"""

from __future__ import annotations

import gc
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SOLVE = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
HEADROOM_BYTES = 6 * 1024**3
KV_FRACTION = 0.5    # share of free memory the KV cache may use
MAX_BATCH_PROBLEMS = 8


def load_model(model_id: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {"local_files_only": True}
    if torch.cuda.is_available():
        kwargs.update(dtype=torch.bfloat16, device_map="auto")
        free = {i: max(torch.cuda.mem_get_info(i)[0] - HEADROOM_BYTES, 0) for i in range(torch.cuda.device_count())}
        kwargs["max_memory"] = {**free, "cpu": 1024**4}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return tokenizer, model


def release() -> None:
    """Return freed GPU memory; call after the caller has dropped its model reference."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def chat_prompt(tokenizer, problem: str, effort: str | None) -> str:
    kwargs = {"reasoning_effort": effort} if effort in ("low", "medium", "high") else {}
    return tokenizer.apply_chat_template([{"role": "user", "content": problem + SOLVE}], tokenize=False,
                                         add_generation_prompt=True, **kwargs)


def _eos_ids(tokenizer, model) -> list[int]:
    ids = {tokenizer.eos_token_id}
    configured = getattr(model.generation_config, "eos_token_id", None)
    ids.update(configured if isinstance(configured, list) else [configured])
    for token in ("<|return|>", "<|call|>", "<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:
            ids.add(token_id)
    return sorted(i for i in ids if i is not None)


def kv_bytes_per_token(model) -> int:
    config = getattr(model.config, "text_config", model.config)
    heads = getattr(config, "num_attention_heads", 32)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
    kv_heads = getattr(config, "num_key_value_heads", None) or heads
    return 2 * config.num_hidden_layers * kv_heads * head_dim * 2


def problems_per_batch(model, n: int, budget: int, prompt_tokens: int) -> int:
    if not torch.cuda.is_available():
        return 1
    free = torch.cuda.mem_get_info()[0]
    per_problem = n * (budget + prompt_tokens) * kv_bytes_per_token(model)
    return max(1, min(MAX_BATCH_PROBLEMS, int(KV_FRACTION * free // per_problem)))


@torch.no_grad()
def sample(model, tokenizer, prompts: list[str], n: int, budget: int, max_time: float, seed: int = 0):
    """Per prompt, n samples as (token ids, finished). Also returns the number of decoding steps run."""
    device = model.get_input_embeddings().weight.device
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    eos = _eos_ids(tokenizer, model)
    torch.manual_seed(seed)
    out = model.generate(**encoded, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, num_return_sequences=n,
                         max_new_tokens=budget, max_time=max_time, pad_token_id=tokenizer.pad_token_id,
                         eos_token_id=eos)
    generated = out[:, encoded["input_ids"].shape[1]:].tolist()
    steps = len(generated[0]) if generated else 0
    eos_set = set(eos)
    rows = []
    for ids in generated:
        end = next((i for i, t in enumerate(ids) if t in eos_set), None)
        rows.append((ids if end is None else ids[:end], end is not None))
    return [rows[i * n:(i + 1) * n] for i in range(len(prompts))], steps
