# 016 — A rollout failure stops the run; RL rewards are scored in a process pool

- Date: 2026-10-28
- Status: proposed
- Milestone: M1 (task M1.3)
- Code: `soe/workers/rollout/failures.py`, `soe/workers/reward_manager/math_reward.py`,
  `soe/workers/rollout/agent_loop.py`, `soe/trainer/main_rl.py` (`run_until_failure`)

## Context

Smoke attempt 2 (job 3222642, 2B, LoRA, 2×A40, 3 steps) exited 0, but its batches were not real:
- verl's naive reward manager calls `compute_score` from executor threads. math-verify's 5 s time
  limits use `signal.alarm`, which only works on a process's main thread, so every response with a
  `\boxed` answer raised `ValueError` (104 times in the log).
- verl 0.9.1's sync trainer logs such an error, marks the group `failure`, and then trains on the
  group's other trajectories. No metric counts the lost ones (`rollout_failure/evicted_samples` is
  only written when failed groups are refilled, which is off).
- Only boxless responses survived, so every logged rollout was truncated at 16,384 tokens and every
  reward was 0. The run looked like a model that never stops; it was a reward crash.

Evaluation scores on the main thread, so M0 never hit this.

## Decision

1. **Scoring.** `compute_score` runs `score_response` in a per-reward-worker pool of 2 `spawn`
   processes, so math-verify keeps evaluation's time limits on the pool workers' main threads.
   Rejected: disabling math-verify's timeouts in RL (a comparison evaluation times out could then
   be accepted in RL, and a pathological answer could hang a reward worker).
2. **Failures stop the run.** The agent loop's `run` and `compute_score` append every exception to
   `runs/<id>/rollout_failures-<job>.jsonl` (path in `SOE_ROLLOUT_FAILURES`, set by `main_rl`).
   `main_rl` checks the file every 30 s and, at the first entry, terminates the trainer and exits
   with the first error. No DONE marker is written. Rejected: verl's
   `sync_refill_failed_groups` (it replaces failed groups with new prompts, which hides a
   systematic failure and biases the batch towards prompts that do not fail), and patching verl's
   `_settle_session_tasks` (verl looks it up before the first agent loop imports soe, so the
   first wave would escape).

## Consequences (including what could confound the algorithm comparison)

- Every algorithm uses the same scorer and the same stop rule, so neither can favour one.
- Rewards equal evaluation verdicts, time limits included (the M1.3 smoke check re-scores the
  stored rollout text with `score_response`).
- Errors outside the two recorded places (for example a TransferQueue write) would still be dropped
  by verl. The M1.6 checker therefore also asserts that each step trained exactly
  `prompts_per_step × group_size` trajectories.
- A transient failure costs the job segment; the resume path (M1.5) restarts from the last
  checkpoint.
