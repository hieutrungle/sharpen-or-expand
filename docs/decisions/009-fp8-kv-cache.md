# 009 — FP8 KV cache for every sampling run and RL rollout; suite soe-eval-v3

- Date: 2026-10-28
- Status: accepted (user decision, against the recommendation to keep BF16)
- Milestone: M0/M1 (amends 004's sampling protocol)

## Context

Sampling Qwen3.5-4B-Base with 16K thinking responses is bound by KV-cache memory. An FP8 KV
cache halves the cache's bytes per token. It changes which tokens are sampled, so it is a
protocol setting, not a speed knob. Published vLLM results show small accuracy changes (sub-point
on Qwen3.5-27B, 1–2 points on a 30B thinking model), mostly measured on Hopper GPUs.

Probe on an A40 (vLLM 0.24.0, `scripts/throughput_probe.py`, same settings as the BF16 probe;
jobs 3222365/3222366, commit 84c5722):

| | BF16 (`auto`) | FP8 (e4m3) |
| --- | --- | --- |
| KV-cache capacity | about 960K tokens | 1.69M tokens (1.75×) |
| Steady generation | about 1,280–1,370 tokens/s | about 1,450–1,840 tokens/s (about 1.25×) |
| Attention backend | default | FlashInfer (chosen by vLLM for FP8 on Ampere) |

Scales cannot be calibrated: vLLM 0.24 disables `calculate_kv_scales` for hybrid models (Qwen3.5
has linear-attention layers) and the checkpoint ships no KV scales, so keys and values are stored
with scale 1.0 (e4m3 range ±448). Only the 8 full-attention layers have a KV cache; the linear-
attention state keeps the model dtype.

## Decision

`sampling.kv_cache_dtype: fp8` in the suite (`soe-eval-v3`), passed to every vLLM engine:
evaluation, preference generation and verl rollouts. v3 is v2 with only this change (pinned by a
test); v2 and v1 stay as `eval_suite_v2.yaml` and `eval_suite_v1.yaml`. `merge_eval` refuses to
combine runs whose sampling differs.

## Consequences (including what could confound the algorithm comparison)

- The base reference is re-sampled on v3 (`docs/reports/qwen35-4b-base_soe-eval-v3_report.json`)
  and compared problem by problem with the BF16 v2 reference (`scripts/compare_eval.py`,
  `docs/reports/m0_fp8_vs_bf16.json`). FP8 minus BF16, mean ± paired standard error:

  | Group | Problems | pass@1 | pass@16 (hard tier: pass@64) |
  | --- | --- | --- | --- |
  | Level A | 1,224 | +0.35 ± 0.21 pts | +1.72 ± 0.51 |
  | Level B | 983 | +0.15 ± 0.32 | +0.20 ± 0.66 |
  | Level C1 | 62 | +0.6 ± 1.3 | +4.8 ± 3.6 |
  | Level C2 | 183 | +0.1 ± 0.7 | −4.9 ± 2.4 |
  | Hard tier | 233 | +0.11 ± 0.27 | +3.0 ± 2.2 |

  Seed control, two BF16 runs (selection seed 1 minus evaluation seed 0): Level A +0.28 ± 0.20
  (pass@1) and +1.06 ± 0.47 (pass@16); Level B +0.23 ± 0.34 and +0.53 ± 0.68. The seed-0 BF16
  draw sits slightly low on Level A pass@16, so a second draw of either kind lands about a point
  higher. **No FP8 effect on pass@1 is detectable at this size (all |diff| < 1 point, within seed
  noise).** pass@k at large k is noisier; the Level C2 drop (2 SE, 183 problems, one of ten
  comparisons) is weak evidence and is not treated as an FP8 effect. From here on every
  checkpoint is compared against the v3 (FP8) base reference, never against v2.
- Cost: 90.3 worker-hours on 6 A40, 4 A30 and 4 A10 for 50,416 samples (about 1.8 hours per 1,000
  samples, against about 2.75 for the BF16 v1 run on a similar GPU mix).
- All checkpoints are sampled the same way, so the shift does not favour any algorithm in
  evaluation. In RL, rollouts (FP8 KV) and the trainer's log-probs (BF16) differ more than with a
  BF16 cache; importance ratios and clip fractions are logged and must be read with that in mind,
  since algorithms with different clipping react differently to the mismatch.
- The attention kernel differs from the BF16 runs (FlashInfer); A100/A30 (sm80) and A40/A10 (sm86)
  may differ in kernel details. Per-shard run info records the GPU.
- Throughput gain is about 1.25×, smaller than the 1.75× capacity gain.

## Alternatives considered

- **BF16 everywhere** (recommended): no shift against the v2 reference and no extra rollout/trainer
  mismatch, at about 25% lower sampling throughput.
- **FP8 for evaluation only, after a paired test**: keeps RL rollouts exact; rejected by the user.
- **`fp8_per_token_head`** (dynamic per-token scales, vLLM 0.24): avoids the fixed scale but is
  newer and untested here.
