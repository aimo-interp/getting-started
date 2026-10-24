# Thought anchors x activation probe: findings and status (2026-10-25)

## 1. Where in the CoT does the robustness probe's signal live? (cvbench, all 82 cases)

`cvbench/generate.py --stages segs` caches per-layer means over 10 equal token
segments of each greedy response; `cv.py --methods seg_probe` runs the
diff-in-means probe on a window of them (layer frac 0.5, 20x5-fold grouped CV).

| Pooling window | cv acc | AUC |
|---|---|---|
| always_true | 0.671 | 0.500 |
| 0-10 (= act_probe resp_mean, previous best) | 0.734 | 0.758 |
| first 10% / 30% / 50% | 0.690 / 0.741 / 0.743 | 0.684 / 0.707 / 0.715 |
| middle 40% (3-7) | 0.737 | 0.754 |
| last 50% / 30% / 10% | 0.723 / 0.690 / 0.670 | 0.743 / 0.733 / 0.670 |
| nested choice of window x layer | 0.720 | 0.695 |
| halves 0-5 + 5-10, one score each | 0.749 | 0.767 |
| thirds 0-3 + 3-7 + 7-10 | 0.742 | 0.785 |
| 0-9 (drop final 10%) | 0.749 | 0.754 |
| **halves without final 10%: 0-5 + 5-9** | **0.763** | 0.750 |

* The final 10% of the CoT (answer wrap-up) carries no robustness signal (chance);
  averaging it in dilutes the probe. Signal is in the first half and middle.
* 0-5 + 5-9 beats resp_mean at every layer tried (frac .4/.5/.6/.7:
  0.749/0.763/0.763/0.756 vs 0.733/0.734/0.734/0.727), but AUC is flat: the gain is
  calibration, not ranking. The drop-final-10% rule came from the window sweep, so
  it is mildly post hoc; confirm on new cases before shipping.
* Nested window selection over-fits at n=82 (0.720); use fixed windows.
* DeepSeek-R1-0528-Qwen3-8B stays below always_true (~0.55 vs 0.615) for every
  variant: the probe does not transfer to it at this sample size (13 cases).

Rows are in `cvbench/results.csv` (method `seg_probe`); loop log in
`autoresearch/loop-261025-0356/results.tsv`.

## 2. Thought-anchor diagnostics (running as batch jobs)

`bash analysis/anchors/run.sh` (screen -> select -> anchor/transplant curves ->
whitebox probe/lens/attn/steer) was smoke-tested end to end at a 2k budget and
submitted at full settings (N=10, 40 points, 16k budget):

| Model | where | work dir |
|---|---|---|
| Qwen/Qwen3.5-4B | interactive H100, then job `anchors` on gpu-l40s (delayed start, resumes) | `analysis/anchors/work/` |
| Skywork/Skywork-OR1-Math-7B | `anchors_sky`, feit-gpu-a100 | `analysis/anchors/work/Skywork_Skywork-OR1-Math-7B/` |
| allenai/Olmo-3-7B-Think | `anchors_olmo`, gpu-a100 | `analysis/anchors/work/allenai_Olmo-3-7B-Think/` |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | `anchors_ds`, deeplearn | `analysis/anchors/work/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B/` |

One pending job per partition (do not fan each model out to all four partitions).
Logs: `analysis/anchors/work/logs/slurm_<jobid>.txt` and `<work dir>/logs/`.
Every stage resumes; resubmit with the same `MODEL=... JOB_NAME=... TIME=24:00:00
bash analysis/anchors/submit.sh` after a timeout. The attention stage
(receiver heads) supports Qwen3.5 only and is skipped for the other models.

What to read when they finish: `wb_probe*.json` (per-sentence probe projection
along X and dX traces, next to P(correct|prefix)). The question for section 1:
does the probe score separate robust vs not before the anchor/commitment
sentence, and is it flat in the wrap-up? If receiver heads (Qwen) point at the
same sentences, a receiver-attention-weighted pooling site is the next cvbench
method to try.

## 3. gpt-oss-120b fixed-trace hidden shift (analysis/thought_anchors/hidden_shift.py)

On the two hand-labelled cases, teacher-forcing the original trace under the
distractor prompt shifts the robust case ~3x more than the fragile one (KL over
noise, relative to a number-free control: 5.15 vs 1.85), on all 8 traces; no KL
spike at the step where the fragile failure happens. The shift tracks distractor
salience, not fragility, matching the prompt-level `shift_resid` null in
cvbench (cv 0.66-0.67, AUC 0.52-0.58). Direction dropped.

## 4. Semantic-entropy probe (cvbench `sep_probe`)

Direct answer entropy over the 4 labelgen samples on X predicts Skywork robustness
with AUC 0.739 (n=36; rank-corr +0.48 with the resp_mean probe; z-sum AUC 0.839),
but those samples are the ones the labels were built from (leaky, and partly label
noise: mid-accuracy problems get noisy drops).

`sep_probe` (ridge from last-prompt h to that entropy, trained on 512 Skywork /
128 DeepSeek labelgen prompts minus test-fold problems) does not work: cv 0.666,
AUC 0.484; with seg_probe 0.759 (vs 0.763 without). The probe barely predicts
entropy on held-out problems (grouped-CV Spearman 0.08-0.16): with 4 samples per
prompt 85% of targets are 0, over only 100 distinct problems. The SEP needs >= 8-10
full-length samples per prompt on more problems (Kossen et al. used 10); the honest
test is entropy from fresh full-length samples on the cvbench cases.
