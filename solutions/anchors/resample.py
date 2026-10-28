#!/usr/bin/env python3
"""Resampling analyses (vLLM env): screen, select, anchors (analysis 1), transplant (analysis 2), summary.

Stages, each resumable, all under ``work/``:

* ``screen``: sample X and every dX variant of every case in ``cases.json``
  (``--screen-n`` samples each, T=1.0) -> ``screen.jsonl``.
* ``select``: pick the cases -> ``selection.json``. Roles:
  ``spurious`` (label not robust, probe right, largest observed drop),
  ``robust`` (label robust, probe right, closest to the spurious case in base
  accuracy and trace length, same perturbation type when possible),
  ``probe_fail`` (probe wrong, largest margin). Per case it fixes one dX (the
  variant with the largest drop), one X trace (a correct sample of median
  length) and one dX trace (a wrong sample if there is one, else correct).
  Override any role with ``--spurious/--robust/--probe-fail GROUP``.
* ``anchors``: Thought Anchors. Split each trace's thinking into sentences;
  from the end of sentence k (up to ``--max-points`` evenly spaced k, plus the
  empty and full prefix) resample ``--n`` continuations and record
  P(correct | prefix). Curves ``X`` (X prompt + X trace) and ``dX``
  (dX prompt + dX trace) -> ``anchors.jsonl``.
* ``transplant``: Thought Branches. Curve ``X->dX``: X-trace prefixes attached
  to the dX prompt. Curve ``dX->X``: dX-trace prefixes on the X prompt (does
  a bad dX start stay bad on the clean prompt?) -> ``anchors.jsonl``.
* ``summary``: per curve, when it reaches/locks at 0.8, its largest jump and
  the anchor sentences -> ``summary.json`` and a printed report.

    python solutions/anchors/resample.py --stages screen select anchors transplant summary
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "datagen"))

from common import SOLVE_SUFFIX, append_jsonl, extract_answer, read_jsonl, tokenizer_path  # noqa: E402

WORK = Path(os.environ.get("ANCHORS_WORK", ROOT / "data" / "anchors"))
THINK_END = "</think>"
# Perturbation types in order of preference when drops tie; distract/typos come
# first because analysis 5 (attention to edited tokens) is clearest for them.
TYPE_ORDER = ("distract", "typos", "typos2", "rename", "rephrase", "llm")


# ---------------------------------------------------------------- text helpers
def thinking_of(trace: str) -> str:
    """The reasoning part of a completion (everything before ``</think>``)."""
    end = trace.find(THINK_END)
    return trace if end < 0 else trace[:end]


def sentence_ends(text: str, min_chars: int = 20) -> list[int]:
    """Character offsets where sentences end, so ``text[:offset]`` is a clean prefix.

    A boundary is a run of newlines, or whitespace after ``.!?`` outside
    ``$...$`` maths. Fragments shorter than ``min_chars`` merge into the next.
    """
    ends, last = [], 0
    for m in re.finditer(r"\n+|(?<=[.!?])[ \t]+", text):
        if m.group().startswith(("\n",)) or text[: m.start()].count("$") % 2 == 0:
            if m.end() - last >= min_chars:
                ends.append(m.end())
                last = m.end()
    if len(text) - last > 0:
        if ends and len(text) - last < min_chars:
            ends[-1] = len(text)
        else:
            ends.append(len(text))
    return ends


def choose_points(ends: list[int], max_points: int) -> list[tuple[int, int]]:
    """(k, offset) pairs: k sentences kept. Always includes k=0 and k=len(ends)."""
    n = len(ends)
    if n + 1 <= max_points:
        ks = list(range(n + 1))
    else:
        ks = sorted({round(i * n / (max_points - 1)) for i in range(max_points)})
    return [(k, 0 if k == 0 else ends[k - 1]) for k in ks]


def type_of(key: str) -> str:
    return "llm" if key.startswith("llm") else key.rsplit("-s", 1)[0]


# ---------------------------------------------------------------- vLLM
class Sampler:
    def __init__(self, args):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # CUDA is already initialised in this process
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # its JIT build needs nvcc, absent on the nodes
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.budget = args.budget
        kwargs = dict(seed=args.seed, max_model_len=args.budget + 4096, enable_prefix_caching=True,
                      gpu_memory_utilization=args.gpu_memory_utilization, limit_mm_per_prompt={"image": 0, "video": 0},
                      tokenizer=tokenizer_path(args.model))
        try:
            self.llm = LLM(args.model, **kwargs)
        except TypeError:  # older vLLM without limit_mm_per_prompt for this model
            kwargs.pop("limit_mm_per_prompt")
            self.llm = LLM(args.model, **kwargs)
        self.tok = self.llm.get_tokenizer()

    def prompt(self, problem: str) -> str:
        return self.tok.apply_chat_template([{"role": "user", "content": problem + SOLVE_SUFFIX}],
                                            tokenize=False, add_generation_prompt=True)

    def n_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def run(self, prompts: list[str], n: int, prefix_tokens: list[int] | None = None) -> list[list]:
        """Per prompt, n (text, finish_reason, n_tokens) continuations within the token budget."""
        prefix_tokens = prefix_tokens or [0] * len(prompts)
        params = [self.SamplingParams(n=n, temperature=1.0, top_p=1.0, top_k=-1,
                                      max_tokens=max(16, self.budget - p)) for p in prefix_tokens]
        outs = self.llm.generate(prompts, params)
        return [[(c.text, c.finish_reason, len(c.token_ids)) for c in o.outputs] for o in outs]


def grade(text: str, finish: str, gold: int | None) -> tuple[int | None, bool]:
    answer = extract_answer(text) if finish == "stop" else None
    return answer, answer is not None and answer == gold


# ---------------------------------------------------------------- stages
def stage_screen(args, cases, sampler):
    out = WORK / "screen.jsonl"
    done = {(r["group"], r["variant"]) for r in read_jsonl(out)}
    jobs = []
    for c in cases:
        for key, text in [("X", c["problem"]), *c["variants"].items()]:
            if (c["group"], key) not in done:
                jobs.append((c["group"], key, text))
    print(f"[screen] {len(jobs)} prompts x {args.screen_n} samples ({len(done)} done)", flush=True)
    for start in range(0, len(jobs), args.chunk_size):
        chunk = jobs[start:start + args.chunk_size]
        results = sampler.run([sampler.prompt(t) for *_, t in chunk], args.screen_n)
        append_jsonl(out, [dict(group=g, variant=k, text=t, texts=[r[0] for r in res], finish=[r[1] for r in res],
                                n_tokens=[r[2] for r in res],
                                answers=[extract_answer(r[0]) if r[1] == "stop" else None for r in res])
                           for (g, k, t), res in zip(chunk, results)])
        print(f"  {min(start + args.chunk_size, len(jobs))}/{len(jobs)}", flush=True)


def screen_stats(cases):
    rows = {}
    for r in read_jsonl(WORK / "screen.jsonl"):
        rows.setdefault(r["group"], {})[r["variant"]] = r
    stats = {}
    for c in cases:
        by = rows.get(c["group"])
        if not by or "X" not in by:
            continue
        x_answers = [a for a in by["X"]["answers"] if a is not None]
        gold = c["gold"] if c["gold"] is not None else (Counter(x_answers).most_common(1)[0][0] if x_answers else None)
        acc = {k: sum(a is not None and a == gold for a in r["answers"]) / len(r["answers"]) for k, r in by.items()}
        base = acc.pop("X")
        drops = {k: base - a for k, a in acc.items()}
        stats[c["group"]] = dict(gold=gold, gold_source="dataset" if c["gold"] is not None else "majority_X",
                                 base_acc=base, acc=acc, drops=drops, rows=by,
                                 mean_tokens=sum(by["X"]["n_tokens"]) / len(by["X"]["n_tokens"]),
                                 truncated=sum(f != "stop" for r in by.values() for f in r["finish"])
                                 / sum(len(r["finish"]) for r in by.values()))
    return stats


def worst_variant(stat, prefer_type: str | None = None) -> str:
    def rank(k):
        t = type_of(k)
        return (-stat["drops"][k], TYPE_ORDER.index(t) if t in TYPE_ORDER else 99)
    keys = sorted(stat["drops"], key=rank)
    if prefer_type:
        same = [k for k in keys if type_of(k) == prefer_type]
        if same and stat["drops"][same[0]] >= stat["drops"][keys[0]] - 0.25:
            return same[0]
    return keys[0]


def pick_trace(row, gold, want_correct: bool) -> tuple[str, bool]:
    items = [(t, f, n, a) for t, f, n, a in zip(row["texts"], row["finish"], row["n_tokens"], row["answers"])]
    good = [i for i in items if (i[3] == gold) == want_correct]
    if not good:
        good = items
    finished = [i for i in good if i[1] == "stop"] or good
    finished.sort(key=lambda i: i[2])
    t, f, n, a = finished[len(finished) // 2]
    return t, a == gold


def stage_select(args, cases):
    stats = screen_stats(cases)
    by_group = {c["group"]: c for c in cases}
    print("[select] screening results (base acc on X, worst dX drop):")
    for g, s in stats.items():
        c = by_group[g]
        w = worst_variant(s)
        print(f"  {g} robust={c['is_robust']!s:5} probe_ok={c['probe_correct']!s:5} base={s['base_acc']:.2f} "
              f"worst={w}:{s['drops'][w]:+.2f} tokens={s['mean_tokens']:.0f} truncated={s['truncated']:.0%} "
              f"gold={s['gold']}({s['gold_source']})")
    eligible = {g: s for g, s in stats.items() if s["base_acc"] >= 0.5}

    def best(pool, key):
        pool = [g for g in pool if g in eligible]
        return min(pool, key=key) if pool else None

    spurious = args.spurious or best(
        [g for g, c in by_group.items() if not c["is_robust"] and c["probe_correct"]],
        lambda g: -eligible[g]["drops"][worst_variant(eligible[g])])
    if spurious is None:
        spurious = best([g for g, c in by_group.items() if not c["is_robust"]],
                        lambda g: -eligible[g]["drops"][worst_variant(eligible[g])])
    if spurious is None:  # no not-robust case reaches base accuracy 0.5: take the most accurate one
        pool = [g for g, c in by_group.items() if not c["is_robust"] and g in stats]
        if not pool:
            raise SystemExit("[select] no not-robust case was screened")
        spurious = max(pool, key=lambda g: stats[g]["base_acc"])
        print(f"  spurious: no eligible case, using {spurious} (base {stats[spurious]['base_acc']:.2f})")
        eligible.setdefault(spurious, stats[spurious])
    s_stat = stats[spurious]
    s_key = worst_variant(s_stat)
    robust = args.robust or best(
        [g for g, c in by_group.items() if c["is_robust"] and c["probe_correct"]],
        lambda g: abs(eligible[g]["base_acc"] - s_stat["base_acc"])
        + 0.5 * abs(math.log(eligible[g]["mean_tokens"] / s_stat["mean_tokens"])))
    fail = args.probe_fail or best([g for g, c in by_group.items() if not c["probe_correct"]],
                                   lambda g: -abs(by_group[g]["probe_score"]))

    selection = []
    for role, g, prefer in (("spurious", spurious, None), ("robust", robust, type_of(s_key)),
                            ("probe_fail", fail, None)):
        if g is None:
            print(f"  no case for role {role}")
            continue
        s, c = stats[g], by_group[g]
        key = worst_variant(s, prefer)
        x_trace, _ = pick_trace(s["rows"]["X"], s["gold"], want_correct=True)
        dx_trace, dx_ok = pick_trace(s["rows"][key], s["gold"], want_correct=False)
        selection.append(dict(role=role, group=g, is_robust=c["is_robust"], max_drop=c["max_drop"],
                              probe_score=c["probe_score"], probe_correct=c["probe_correct"], gold=s["gold"],
                              gold_source=s["gold_source"], base_acc=s["base_acc"], dx_key=key,
                              dx_acc=s["acc"][key], variant_acc=s["acc"], x_text=c["problem"],
                              dx_text=c["variants"][key], x_trace=x_trace, dx_trace=dx_trace,
                              dx_trace_correct=dx_ok, mean_tokens=s["mean_tokens"]))
        print(f"  {role:10} {g} dX={key} base={s['base_acc']:.2f} dX_acc={s['acc'][key]:.2f} "
              f"dX trace {'correct' if dx_ok else 'WRONG'}")
    (WORK / "selection.json").write_text(json.dumps(selection, indent=1))


def curves_for(sel, which: list[str]):
    """(curve name, prompt problem text, trace) for the requested curves."""
    spec = {"X": (sel["x_text"], sel["x_trace"]), "dX": (sel["dx_text"], sel["dx_trace"]),
            "X->dX": (sel["dx_text"], sel["x_trace"]), "dX->X": (sel["x_text"], sel["dx_trace"])}
    return [(name, *spec[name]) for name in which]


def stage_curves(args, sampler, which: list[str]):
    out = WORK / "anchors.jsonl"
    done = {(r["group"], r["curve"]) for r in read_jsonl(out)}
    for sel in json.loads((WORK / "selection.json").read_text()):
        for curve, problem, trace in curves_for(sel, which):
            if (sel["group"], curve) in done:
                continue
            thinking = thinking_of(trace)
            ends = sentence_ends(thinking)
            points = choose_points(ends, args.max_points)
            base = sampler.prompt(problem)
            prefixes = [thinking[:off] for _, off in points]
            n_prefix = [sampler.n_tokens(p) for p in prefixes]
            print(f"[{curve}] {sel['role']} {sel['group']}: {len(ends)} sentences, {len(points)} points "
                  f"x {args.n} samples", flush=True)
            results = sampler.run([base + p for p in prefixes], args.n, n_prefix)
            rows = []
            for (k, off), npf, res in zip(points, n_prefix, results):
                graded = [grade(t, f, sel["gold"]) for t, f, _ in res]
                rows.append(dict(k=k, offset=off, prefix_tokens=npf, answers=[a for a, _ in graded],
                                 correct=[c for _, c in graded], p=sum(c for _, c in graded) / len(graded),
                                 n_tokens=[n for *_, n in res]))
            append_jsonl(out, [dict(group=sel["group"], role=sel["role"], curve=curve, n_sentences=len(ends),
                                    sentence_ends=ends, points=rows)])


def curve_features(points, thinking: str, ends: list[int], threshold: float = 0.8) -> dict:
    ps = [pt["p"] for pt in points]
    ks = [pt["k"] for pt in points]
    n = max(1, ks[-1])
    reach = next((ks[i] for i, p in enumerate(ps) if p >= threshold), None)
    lock = next((ks[i] for i in range(len(ps)) if all(p >= threshold for p in ps[i:])), None)
    jumps = []
    for i in range(1, len(ps)):
        lo, hi = ks[i - 1], ks[i]  # sentences lo+1..hi were added between these points
        start = 0 if lo == 0 else ends[lo - 1]
        jumps.append(dict(k_from=lo, k_to=hi, delta=ps[i] - ps[i - 1], text=thinking[start:ends[hi - 1]].strip()[:400]))
    jumps.sort(key=lambda j: -abs(j["delta"]))
    return dict(p=ps, k=ks, frac=[k / n for k in ks], p_start=ps[0], p_end=ps[-1], mean_p=sum(ps) / len(ps),
                reach_frac=None if reach is None else reach / n, lock_frac=None if lock is None else lock / n,
                max_jump=jumps[0]["delta"] if jumps else 0.0, anchors=jumps[:3])


def stage_summary():
    selection = {s["group"]: s for s in json.loads((WORK / "selection.json").read_text())}
    summary = {}
    for r in read_jsonl(WORK / "anchors.jsonl"):
        sel = selection.get(r["group"])
        if sel is None:
            continue
        trace = sel["x_trace"] if r["curve"] in ("X", "X->dX") else sel["dx_trace"]
        feats = curve_features(r["points"], thinking_of(trace), r["sentence_ends"])
        summary.setdefault(r["group"], dict(role=sel["role"], dx_key=sel["dx_key"], base_acc=sel["base_acc"],
                                            dx_acc=sel["dx_acc"], probe_score=sel["probe_score"],
                                            is_robust=sel["is_robust"], curves={}))["curves"][r["curve"]] = feats
    for g, s in summary.items():
        c = s["curves"]
        if "X->dX" in c and "dX" in c:
            # Recovery: does a clean-trace prefix rescue the dX prompt? Compare at matched depth.
            s["transplant_gap"] = c["X"]["mean_p"] - c["X->dX"]["mean_p"] if "X" in c else None
            s["transplant_recovery"] = c["X->dX"]["mean_p"] - c["dX"]["p_start"]
    (WORK / "summary.json").write_text(json.dumps(summary, indent=1))

    fmt = lambda v: "  -  " if v is None else f"{v:5.2f}"
    for g, s in summary.items():
        print(f"\n=== {s['role']} {g} (label robust={s['is_robust']}, probe={s['probe_score']:+.3f}, "
              f"dX={s['dx_key']}, base acc {s['base_acc']:.2f} -> dX acc {s['dx_acc']:.2f})")
        for name, f in s["curves"].items():
            curve = " ".join(f"{p:.1f}" for p in f["p"])
            print(f"  {name:6} start={f['p_start']:.2f} end={f['p_end']:.2f} mean={f['mean_p']:.2f} "
                  f"reach0.8@{fmt(f['reach_frac'])} lock0.8@{fmt(f['lock_frac'])} maxjump={f['max_jump']:+.2f}")
            print(f"         P(correct|k): {curve}")
            for a in f["anchors"][:2]:
                print(f"         anchor k{a['k_from']}->{a['k_to']} {a['delta']:+.2f}: {a['text'][:160]!r}")
        if "transplant_recovery" in s:
            print(f"  transplant: X-prefix on dX prompt mean P={s['curves']['X->dX']['mean_p']:.2f} "
                  f"(dX from scratch {s['curves']['dX']['p_start']:.2f}; X on X {s['curves'].get('X', {}).get('mean_p', float('nan')):.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", default=["screen", "select", "anchors", "transplant", "summary"],
                        choices=["screen", "select", "anchors", "transplant", "summary"])
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--screen-n", type=int, default=8)
    parser.add_argument("--n", type=int, default=10, help="Resamples per prefix.")
    parser.add_argument("--max-points", type=int, default=40, help="Prefix points per curve.")
    parser.add_argument("--budget", type=int, default=16384, help="Max completion tokens, counted from <think>.")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spurious")
    parser.add_argument("--robust")
    parser.add_argument("--probe-fail")
    args = parser.parse_args()

    cases = json.loads((WORK / "cases.json").read_text())
    needs_gpu = {"screen", "anchors", "transplant"} & set(args.stages)
    sampler = Sampler(args) if needs_gpu else None
    if "screen" in args.stages:
        stage_screen(args, cases, sampler)
    if "select" in args.stages:
        stage_select(args, cases)
    if "anchors" in args.stages:
        stage_curves(args, sampler, ["X", "dX"])
    if "transplant" in args.stages:
        stage_curves(args, sampler, ["X->dX", "dX->X"])
    if "summary" in args.stages:
        stage_summary()


if __name__ == "__main__":
    main()
