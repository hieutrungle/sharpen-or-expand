# 022 — RL steps make 4 optimizer updates (mini-batches), the same for every algorithm

- Date: 2026-10-29
- Status: accepted (user, 2026-10-29: "4 updates per step")
- Milestone: M1 (found in task M1.6; before the M1.9 run and the M1.10/M1.11 measurements)
- Code: `soe/trainer/config/experiment.py` (`rollout.updates_per_step`),
  `soe/trainer/rl/verl_overrides.py` (`actor.ppo_mini_batch_size`)

## Context

Until now each training step made exactly one optimizer update: `actor.ppo_mini_batch_size` was the
step's whole batch (`prompts_per_step`) with `ppo_epochs=1`. The update then runs on the policy that
generated the rollouts, so the PPO ratio pi_theta / pi_old is 1 for every token. Every M1.5 step
logged `actor/pg_clipfrac = 0` and `actor/ppo_kl = 0` exactly. With a ratio of 1:
- clipping never acts, so `clip_ratio_low/high` and DAPO's Clip-Higher (a leave-one-out ablation in
  the milestone plan) change nothing;
- GRPO, DAPO and RS-GRPO lose the trust-region part that distinguishes them in their papers.

The published recipes split each step into mini-batches: DAPO samples 512 prompts per step and
updates on mini-batches of 32 prompts (16 updates per step). DeepSeekMath's GRPO made a single update
per exploration stage, which is the setup we had.

verl 0.9.1 splits the step's batch into `ppo_mini_batch_size` prompts (times G rollouts), runs one
optimizer step per mini-batch, and steps the LR scheduler once per training step (on the last
mini-batch), so warm-up stays counted in training steps. Each rollout is still used once per epoch,
so the forward/backward tokens per step do not change; only the optimizer steps (3 more per
training step) are added.

## Decision

1. Every RL config states `rollout.updates_per_step`; `prompts_per_step` must be divisible by it.
   soe sets `actor.ppo_mini_batch_size = prompts_per_step / updates_per_step`, with
   `ppo_epochs = 1` and verl's default `shuffle = false`, so each mini-batch holds whole groups.
2. **4 updates per step** for GRPO, Dr.GRPO, DAPO and RS-GRPO, in every 2B and 4B run, including
   the G = 4 / 8 / 16 check: the number of updates, and so the staleness of the last mini-batch
   (3 updates), is the same in every comparison. DPO is not affected.
3. Development configs follow: the dev and resume configs (64 prompts) get mini-batches of 16
   prompts (128 rollouts); the smoke config (16 prompts) mini-batches of 4; the identity configs
   (learning rate 0) keep 1 update, since they test the export, not training.

## Consequences (including what could confound the algorithm comparison)

- `actor/pg_clipfrac`, `actor/pg_clipfrac_lower` and `actor/ppo_kl` become informative, and the
  clip ratios (and Clip-Higher) act on the later mini-batches.
- The policy moves about 4 times as many optimizer steps per rollout token as before, at the same
  learning rate. The M1.9 run watches KL, entropy and grad norm; the learning rate is still the same
  for every algorithm, so the comparison stays matched.
- Step time: the same tokens are trained, plus 3 optimizer steps. M1.10/M1.11 measure throughput
  with this setting, so the G0/G1 numbers include it.
- Runs before this decision (M1.3–M1.6 development runs) made one update per step; none counts
  toward conclusions.

## Alternatives considered

- **1 update per step (status quo, DeepSeekMath's GRPO):** fully on-policy and simplest, but the
  clip settings do nothing, so the DAPO ablation "minus Clip-Higher" would measure noise.
- **16 updates per step (DAPO's recipe):** DAPO's mini-batch is 32 prompts of a 512-prompt step; our
  steps are smaller (64 prompts in the 2B development config, set for 4B at G1), so 16 updates would
  mean 4-prompt mini-batches with noisy gradients, and the last one 15 updates stale.
- **A fixed mini-batch size in prompts instead of a fixed number of updates:** the number of updates
  (and the staleness) would then change with the batch size between configs; a fixed number keeps
  every comparison at the same off-policy degree.
