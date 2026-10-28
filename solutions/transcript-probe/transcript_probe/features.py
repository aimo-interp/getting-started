"""Gold-free transcript features of a model's own samples on the original problem.

Each sample is the generated text (cut at the thinking budget) and whether the
model finished within that budget. Three features, averaged over samples:

* ``pre_answer_logchars``: log characters of reasoning before the sample's own
  final answer first appears (the whole reasoning if it never answers). Long
  searches before a first answer go with fragility; long re-checking after it
  does not.
* ``anchor``: within the first ``ANCHOR_CHARS`` characters, how often the most
  repeated ``= N`` value (2-8 digits, not a number from the problem) occurs.
  Robust solves lock onto one concrete quantity early and keep returning to it.
* ``agree``: share of samples giving the majority final answer (unfinished
  samples count as disagreeing).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from statistics import mean

FEATURES = ("pre_answer_logchars", "anchor", "agree")
ANCHOR_CHARS = 16000
_ANCHOR = re.compile(r"=\s*\$?(\d{2,8})\b(?![\^.\d])")
_THINK_END = ("</think>", "<|end|>")


# ---------------------------------------------------------------- answers
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


def parse_integer(text: str | None) -> int | None:
    """Parse an integer answer such as ``"033"``, ``"142.0"``, or ``"1,024"`` (commas only as thousands)."""
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


def extract_answer(completion: str) -> int | None:
    """Final integer answer: last ``\\boxed{}``, else ``ANSWER:``, else the last integer."""
    boxed = _last_boxed(completion)
    if boxed is not None:
        content = re.sub(r"\\[,;: ]|~", "", boxed).strip()
        value = parse_integer(content)
        if value is None and "=" in content:
            value = parse_integer(content.rsplit("=", 1)[1])
        return value
    match = re.findall(r"ANSWER:\s*([+-]?[\d,]+)", completion)
    if match:
        return parse_integer(match[-1])
    tail = re.findall(r"[+-]?\d[\d,]*", completion[-300:])
    return parse_integer(tail[-1]) if tail else None


# ---------------------------------------------------------------- features
def reasoning_of(text: str) -> str:
    for marker in _THINK_END:
        if marker in text:
            return text.split(marker, 1)[0]
    return text


def transcript_features(problem: str, texts: list[str], finished: list[bool]) -> dict:
    """Features of n samples on the original problem (see the module docstring)."""
    answers = [extract_answer(t) if done else None for t, done in zip(texts, finished)]
    in_problem = set(re.findall(r"\d+", problem))
    pre, anchors = [], []
    for text, answer in zip(texts, answers):
        think = reasoning_of(text)
        hit = re.search(r"(?<!\d)%s(?!\d)" % re.escape(str(answer)), think) if answer is not None else None
        pre.append(math.log((hit.start() if hit else len(think)) + 1))
        head = [m.group(1) for m in _ANCHOR.finditer(text[:ANCHOR_CHARS]) if m.group(1) not in in_problem]
        anchors.append(Counter(head).most_common(1)[0][1] if head else 0)
    given = [a for a in answers if a is not None]
    agree = Counter(given).most_common(1)[0][1] / len(answers) if given else 0.0
    return {"pre_answer_logchars": mean(pre), "anchor": float(mean(anchors)), "agree": agree,
            "answers": answers}
