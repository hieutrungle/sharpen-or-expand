# 021 — RL step metrics: soe metrics from per-rollout records; importance ratios from verl's off-policy diagnostics

- Date: 2026-10-29
- Status: accepted (user, 2026-10-29: "Per-rollout records from the reward hook", "Use verl's
  off-policy diagnostics")
- Milestone: M1 (task M1.6)
- Code: `soe/trainer/rl/metrics.py`, `soe/trainer/main_metrics.py`,
  `soe/workers/reward_manager/math_reward.py`, `soe/workers/rollout/agent_loop.py`,
  `scripts/check_run_metrics.py`, `scripts/transitions.py`

## Context

The project rules list the metrics every run logs. verl 0.9.1 logs about 90 keys per step (reward,
entropy, grad norm, clip fraction, response lengths, timings, `rollout_corr/*`), but not the
group-level and behaviour metrics: group reward variance, zero-variance groups, boxed rate, stop
reasons, length percentiles, answer diversity. Its training generation dumps hold only the text,
score and uid. Its trainer class is fixed by `trainer_mode`, so per-step metrics inside verl would
need another import-time patch of its internals.

verl computes importance-weight statistics (mean, max, fraction above a threshold) only when
rollout correction is applied (`algorithm.rollout_correction.rollout_is`), which changes the loss.
With correction off it logs distribution-level diagnostics of the rollout-vs-trainer gap.

## Decision

1. **Per-rollout records.** The agent loop adds the step to each rollout's extra fields. The reward
   hook appends one `RolloutOutcome` per training rollout (step, problem, score, truncated, boxed,
   turn end, extracted answer) to `runs/<id>/rollouts/segment-<job>.jsonl`, inside the failure
   guard of decision 016. Lengths come from the rollout-token records of decision 018.
2. **Per-step soe metrics at each save.** The checkpoint callback starts
   `python -m soe.trainer.main_metrics` in the background, and main_rl runs it when a job segment
   ends. It writes `runs/<id>/rollout_metrics.jsonl` (`soe/*` keys, one line per step) and fills
   `efficiency.json` (rollout and training tokens, wall-clock and GPU-hours). Answer diversity
   groups final answers by verifier equivalence like maj@k; a process of its own keeps math-verify's
   time limits on a main thread.
3. **Importance ratios** are covered by verl's off-policy diagnostics with correction off:
   `rollout_corr/kl`, `k3_kl`, `chi2_token` (E[rho^2] - 1 per token), `chi2_seq`,
   `log_ppl_diff_max/min` and `training/rollout_probs_diff_max`. Ratio mean, max and the fraction
   above 2 are not logged.
4. **The checker** (`scripts/check_run_metrics.py`) fails a run that misses any verl or soe key on
   any step, holds other than prompts x G records per step, or whose soe reward and length means
   differ from verl's (`critic/score/mean`, `response_length/mean`): verl would then have trained
   other rollouts than were generated (decision 016's known gap). On the M1.5 control run the
   rollout-token mean equalled verl's `response_length/mean` exactly at all 12 steps.

## Consequences (including what could confound the algorithm comparison)

- Every algorithm logs the same metrics from the same records.
- soe metrics appear at each save, not live every step; verl's own keys stay live.
- A step whose rollouts were replayed after a resume counts its latest segment only (as tokens do).
- No direct "fraction of token ratios > 2": a heavy tail shows in `chi2_token`/`chi2_seq` and the
  log-perplexity extremes instead.

## Alternatives considered

- **Offline from verl's generation dumps:** no worker change, but the dumps lack truncation and stop
  reasons, depend on verl's format, and the verifier would run again.
- **Patching verl's driver to log them live:** another patch of verl internals to re-check on every
  upgrade.
- **Patching the actor for ratio statistics:** exact roadmap metrics, but a patch inside the FSDP
  actor next to decision 019's.
