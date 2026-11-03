# 020 — RL runs stop at their rollout-token budget: the last step is re-estimated at each save

- Date: 2026-10-29
- Status: accepted (user, 2026-10-29: "Re-estimate the last step at each save")
- Milestone: M1 (task M1.5; implements decision 013's budget, defined in decision 018)
- Code: `soe/trainer/rl/budget.py`, `soe/trainer/rl/checkpoint_callback.py`,
  `soe/trainer/main_rl.py` (`segment_overrides`), `soe/trainer/rl/verl_overrides.py`

## Context

Runs are compared at equal rollout-token budgets (decision 013). verl 0.9.1 stops only at
`trainer.total_training_steps`, has no per-step hook, and its trainer class cannot be swapped. soe
sees the token count at each save, every 2–4 hours. The M1.9 dev run also needs a wall-clock cap
("30M rollout tokens or 20 h").

## Decision

1. At each save the checkpoint callback computes the step at which the run stops: the first step
   whose cumulative rollout tokens reach `budget.rollout_tokens`, using the mean tokens per step
   since the previous save; or, for development runs, the last step that fits in
   `budget.max_hours` (training wall-clock over all job segments) or `budget.max_steps`, whichever
   comes first. It sets verl's `trainer.total_training_steps` to that step. verl's loop reads the
   total before every step, so that step runs as its last: it is saved and validated.
2. The estimate goes to `runs/<id>/stop_step.json`; a resumed job starts with it (main_rl replaces
   `trainer.total_training_steps` in that job's override list). Without `max_steps` the frozen list
   gives verl no step total (null), only the 1,000-epoch bound.
3. The learning-rate schedule is pinned to constant after a fixed number of warm-up steps
   (`lr_scheduler_type=constant`, `lr_warmup_steps_ratio=0`), so changing the step total cannot
   change the schedule.
4. A resumed job skips verl's validation before training: its checkpoint step was validated when
   saved, and a second record at that step would replace the step's training metrics in the merge.

## Consequences (including what could confound the algorithm comparison)

- Every algorithm stops the same way. Runs end within about one step of the budget, and the exact
  count of each checkpoint is recorded (`checkpoints.jsonl`), so comparisons use the recorded
  tokens, not the nominal budget.
- The estimate assumes the tokens per step of the last save interval continue. If responses grow
  fast, the run can stop a few steps before the budget; the next save corrects the estimate, so the
  error is bounded by the change within one save interval. A run never stops before the step just
  saved.
- A cosine or other step-dependent schedule would need a new decision.

## Alternatives considered

- **Stop at the first save past the budget:** simplest, but overshoots by up to one save interval
  (2–4 h of tokens, 10–20% of a 30M-token run), and runs end at different token counts.
- **A verl trainer subclass with a per-step check:** exact, but verl picks its trainer class from
  `trainer_mode`, so it would need a deeper patch.

## Revisions

- 2026-10-30 (user, M1.9): the development run's budget becomes 220M rollout tokens with the 20 h
  cap, so the cap binds (~36 steps); at ~6.2M rollout tokens per 512-rollout step (M1.5), the
  30M-token budget would have ended it after ~5 steps.
- 2026-11-03 (user, M1.11: "Fix now in M1.11"): a save can lower the stop step but could not raise
  it once verl had marked the current step as its last. verl 0.9.1 sets `is_last_step` when a step
  starts and returns through it. So when an earlier estimate landed exactly on a later save step,
  the run ended there even if that save's re-estimate asked for more (M1.11 B-L run: estimate 12 at
  step 6, re-estimate 13 at step 12, ended at 12). With a token budget, a run could end up to one
  step short. `soe.trainer.main_rl.train_to_stop` now launches verl again in the same job when
  `stop_step.json` is later than the last saved step (the last step is always saved); the launch
  resumes from that save, skips the validation before training and starts with the new stop. Cost:
  one start-up (~10 min) and one more validation pass, only in runs that hit the case. CPU tests
  `tests/trainer/test_main_rl_on_cpu.py`; first use on GPUs in M2.
