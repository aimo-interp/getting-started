"""Shared helpers for label generation (standard library only).

Imported both by scripts that run in the project's ``uv`` environment and by
scripts that run in a separate vLLM environment, so it must not import any
third-party package.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_DIR = ROOT / "data" / "labelgen"

# Label rule inferred from aimo-interp/train-main-v2: robust at max_drop <= 0.10,
# not robust at >= 0.25, unlabelled in between.
ROBUST_MAX_DROP = 0.10
NOT_ROBUST_MIN_DROP = 0.25

PERTURBATION_TYPES = ("rephrase", "rename", "domain", "distract", "typos", "expert")

SOLVE_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def problem_id(text: str) -> str:
    """Stable id from normalized problem text."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()[:12]


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace, for de-duplication."""
    return re.sub(r"\s+", " ", text).strip().lower()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def parse_integer(text: str | None) -> int | None:
    """Parse an integer answer such as ``"033"``, ``"142.0"``, or ``"1,024"``.

    Commas count only as thousands separators, so a list like ``"1,11"`` is
    not an integer.
    """
    if text is None:
        return None
    cleaned = text.strip().strip("$").replace("\\!", "").replace("{,}", ",").strip()
    if "," in cleaned:
        if not re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.0+)?", cleaned):
            return None
        cleaned = cleaned.replace(",", "")
    cleaned = re.sub(r"^\\text\{(.*)\}$", r"\1", cleaned).strip()
    match = re.fullmatch(r"([+-]?\d+)(?:\.0+)?", cleaned)
    return int(match.group(1)) if match else None


def _last_boxed(text: str) -> str | None:
    start = text.rfind("\\boxed")
    if start < 0:
        return None
    brace = text.find("{", start)
    if brace < 0:
        return None
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : index]
    return None


def parse_boxed(content: str) -> int | None:
    """Integer in a ``\\boxed{}``, allowing working such as ``18 \\times 41 = 738``.

    Thin-space separators (``742\\,899``) are removed, and when the content is
    not itself an integer the value after the last ``=`` is used.
    """
    content = re.sub(r"\\[,;: ]|~", "", content).strip()
    value = parse_integer(content)
    if value is None and "=" in content:
        value = parse_integer(content.rsplit("=", 1)[1])
    return value


def extract_answer(completion: str) -> int | None:
    """Final integer answer: last ``\\boxed{}``, else ``ANSWER:``, else the last integer."""
    boxed = _last_boxed(completion)
    if boxed is not None:
        return parse_boxed(boxed)
    match = re.findall(r"ANSWER:\s*([+-]?[\d,]+)", completion)
    if match:
        return parse_integer(match[-1])
    tail = re.findall(r"[+-]?\d[\d,]*", completion[-300:])
    return parse_integer(tail[-1]) if tail else None


def label_for(max_drop: float) -> bool | None:
    if max_drop <= ROBUST_MAX_DROP:
        return True
    if max_drop >= NOT_ROBUST_MIN_DROP:
        return False
    return None


def tokenizer_path(model_id: str) -> str:
    """Where to load ``model_id``'s tokenizer from (pass to vLLM ``tokenizer=`` or ``AutoTokenizer``).

    transformers 5 loads a checkpoint whose ``tokenizer_config.json`` names
    ``LlamaTokenizerFast`` over a byte-level BPE ``tokenizer.json``
    (DeepSeek-R1-0528-Qwen3-8B) as a Llama tokenizer that mis-encodes text and
    decodes ``Ġ``/``Ċ`` literally. For such checkpoints this returns a copy under
    ``data/tokenizers/`` that names ``PreTrainedTokenizerFast``; otherwise ``model_id``.
    """
    if Path(model_id).is_dir():
        snap = Path(model_id)
    else:
        hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
        snaps = sorted((hub / f"models--{model_id.replace('/', '--')}" / "snapshots").glob("*"))
        if not snaps:
            return model_id
        snap = snaps[-1]
    cfg_path = snap / "tokenizer_config.json"
    if not cfg_path.exists() or not (snap / "tokenizer.json").exists():
        return model_id
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("tokenizer_class") != "LlamaTokenizerFast":
        return model_id
    if '"ByteLevel"' not in (snap / "tokenizer.json").read_text(encoding="utf-8"):
        return model_id
    out = ROOT / "data" / "tokenizers" / model_id.replace("/", "_")
    if not (out / "tokenizer_config.json").exists():
        out.mkdir(parents=True, exist_ok=True)
        for name in ("tokenizer.json", "special_tokens_map.json", "chat_template.jinja", "generation_config.json"):
            if (snap / name).exists():
                shutil.copyfile(snap / name, out / name)
        (out / "tokenizer_config.json").write_text(json.dumps({**cfg, "tokenizer_class": "PreTrainedTokenizerFast"}, indent=1))
    return str(out)
