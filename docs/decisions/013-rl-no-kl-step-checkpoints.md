# 013 — RL runs: no KL term; a checkpoint every N steps with its rollout-token count

- Date: 2026-10-28
- Status: accepted (user, 2026-10-28: D5 option (a) and D7 option (a))
- Milestone: M1 (task M1.1; amends the plan's "checkpoints at fixed rollout-token intervals")
- Enforced in: `soe/trainer/config/experiment.py` (`PolicyLossSpec` has no KL field;
  `CheckpointSpec.save_every_steps`)

## Context

Two settings shared by all RL algorithms had to be fixed before any RL config exists.
- **KL.** The original GRPO adds a KL penalty to a reference model; DAPO and Dr.GRPO drop it. Keeping
  it for some algorithms only would add one more difference between them.
- **Checkpoints.** verl 0.9.1's v1 trainer saves every `save_freq` steps; it has no token-based
  trigger. The plan asked for checkpoints at fixed rollout-token intervals and every 2–4 hours.

## Decision

1. **No KL term and no reference model for any RL algorithm** (GRPO, Dr.GRPO, DAPO, RS-GRPO). The
   config schema has no field for one; the verl overrides set `use_kl_loss=false` and
   `use_kl_in_reward=false` (M1.3).
2. **A checkpoint every N steps.** N (`checkpoint.save_every_steps`) is set from the measured step
   time so that saves land every 2–4 hours. Each save appends its step, cumulative rollout tokens,
   wall-clock time and hardware to `runs/<id>/checkpoints.jsonl` (M1.4).
   - Runs are compared at equal rollout-token budgets by taking, for each token mark, the checkpoint
     nearest to it. Lane B evaluates those checkpoints.

## Consequences (including what could confound the algorithm comparison)

- The algorithms differ only in the decision-001 switches. Memory and time are lower without a
  reference model (no second weight copy, no reference forward pass).
- Without KL, the policy can drift further from the SFT start. Entropy, length and the
  rollout/trainer log-prob gap are logged every step, so drift is visible.
- **Token marks are approximate.** Runs with different response lengths reach a token mark at
  different steps, so the nearest checkpoint can be off by up to half a save interval in tokens.
  `checkpoints.jsonl` records the exact count, and comparisons report it. If G1 needs exact marks, a
  token-triggered save is a new decision.

## Alternatives considered

- **KL for GRPO only:** faithful to the original GRPO, but one more difference between algorithms, and
  the memory study would need a reference model in the GRPO cells.
- **Token-triggered saves through a verl trainer subclass:** exact marks, but a subclass of verl's new
  v1 trainer to maintain and test.
