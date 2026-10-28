# 014 — RL rollouts sample like evaluation; a truncated rollout gets reward 0

- Date: 2026-10-28
- Status: accepted (user, 2026-10-28: D3 "use the 0.6 / 0.95, equal to evaluation"; D4 the
  recommendation, reward 0)
- Milestone: M1 (task M1.1/M1.3; extends the "One yardstick" rule to the RL sampling distribution)
- Enforced in: `soe/trainer/config/experiment.py` (`RolloutSpec` has no temperature or top-p;
  `rollout_protocol` takes them from the suite) and the reward manager (M1.3)

## Context

The RL rollout distribution was open. verl and DAPO default to temperature 1.0 / top-p 1.0; the
frozen evaluation protocol (decision 004, Yue et al.) samples at 0.6 / 0.95. verl's reward hook also
scores a response cut off at the length cap, while evaluation counts it as incorrect.

## Decision

1. **Rollout sampling = evaluation sampling:** temperature 0.6, top-p 0.95, top-k off, read from the
   suite file, so the two cannot drift. This applies to every RL algorithm and to the in-loop
   validation.
   - The suite's **per-request seed (0) is not used for rollouts.** It would give a prompt the same
     random numbers each time it is drawn. Rollouts take vLLM's engine seed from the run seed;
     only evaluation uses seed 0.
2. **A rollout that reaches the 16,384-token cap gets reward 0**, whatever its last box says, as in
   evaluation (truncated = incorrect). DAPO's overlong shaping (M3) is applied on top of this, and its
   ablation keeps the base rule.

## Consequences (including what could confound the algorithm comparison)

- The policy is trained on the distribution it is evaluated on, and pass@k at the end measures the
  same sampler RL optimised. Every algorithm uses the same rollout distribution.
- Lower temperature means less exploration than the published recipes (DAPO and most GRPO work use
  1.0). M3's "verified against published behaviour" check must allow for this: entropy, zero-variance
  groups and the clip fraction will differ from the papers' curves. That is a difference shared by all
  arms, not between them, but the report states it.
- Algorithms that lean on exploration (Clip-Higher, RS-GRPO's risk seeking) may gain less at 0.6 than
  at 1.0. Any "expansion" conclusion is conditional on this sampling temperature.
- Reward 0 at the cap pushes toward shorter answers, a pressure evaluation also applies. Truncation is
  logged per step, and transitions separate truncation-driven failures (decision 006).

## Alternatives considered

- **1.0 / 1.0 (verl and DAPO default):** matches the published recipes and explores more, but trains on
  a different distribution from the one evaluated.
- **Scoring truncated rollouts by their box (verl default):** rewards an answer evaluation rejects.
