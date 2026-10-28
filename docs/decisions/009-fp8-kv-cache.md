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

- The base reference is re-sampled on v3. Comparing it problem by problem with the BF16 v2
  reference measures the FP8 shift for this model; the result is added here when it is in.
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
