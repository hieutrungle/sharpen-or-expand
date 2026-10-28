# 014 — RL rollout sampling and truncation: rollouts at 1.0 / 1.0, a truncated rollout gets reward 0

- Date: 2026-10-28 (revised the same day by decision 017)
- Status: accepted. Current form: user, 2026-10-28, decision 017 (rollouts and DPO preference data
  at 1.0 / 1.0, in-loop validation at evaluation sampling); D4 the recommendation, reward 0. The
  first form (rollouts at evaluation's 0.6 / 0.95) is kept under History.
- Milestone: M1 (tasks M1.1/M1.3)
- Enforced in: `soe/trainer/config/experiment.py` (`RolloutSpec.temperature` / `top_p` are required;
  `rollout_protocol` gives rollouts the config's sampler and validation the suite's),
  `soe/trainer/rl/verl_overrides.py`, and the reward hook
  `soe/workers/reward_manager/math_reward.py` (reward 0 when truncated)

## Context

The RL rollout distribution was open. verl and DAPO default to temperature 1.0 / top-p 1.0; the
frozen evaluation protocol (decision 004, Yue et al.) samples at 0.6 / 0.95. verl's reward hook also
scores a response cut off at the length cap, while evaluation counts it as incorrect.

## Decision (current)

1. **Sampling** (decision 017 holds the reasons and the literature):
   - RL rollouts sample at temperature 1.0, top-p 1.0, top-k off, for every algorithm, seed and
     ablation; the values are in each experiment YAML and must be equal across compared runs;
   - DPO preference pairs are sampled from the SFT policy at 1.0 / 1.0 as well;
   - in-loop validation (the G1 stopping signal) and evaluation sample at the suite's 0.6 / 0.95;
   - the suite's **per-request seed (0) is not used for rollouts**: it would give a prompt the same
     random numbers each time it is drawn. Rollouts take vLLM's engine seed from the run seed; only
     evaluation uses seed 0.
2. **A rollout that reaches the 16,384-token cap gets reward 0**, whatever its last box says, as in
   evaluation (truncated = incorrect). DAPO's overlong shaping (M3) is applied on top of this, and
   its ablation keeps the base rule.

## Consequences (including what could confound the algorithm comparison)

- Every algorithm, and DPO's preference data, uses the same sampler, so it cannot favour one.
- RL optimises the 1.0 / 1.0 sampler while pass@k measures the 0.6 / 0.95 one. This is the usual
  setup; the report states it.
- At 1.0 / 1.0 the sampler is the policy, so PPO's importance ratio has no sampler/policy mismatch
  beyond numerics. The M1.3 smokes measured the rollout/trainer gap (`rollout_corr/kl`) at 0.0009
  per step at 1.0 / 1.0, against 0.008–0.010 at 0.6 / 0.95.
- Reward 0 at the cap pushes toward shorter answers, a pressure evaluation also applies. At
  1.0 / 1.0 more rollouts hit the cap (about half for the 2B base model in the M1.3 smoke).
  Truncation is logged per step, and transitions separate truncation-driven failures (decision 006).

## Alternatives considered

- **0.6 / 0.95, equal to evaluation (the first form of this decision):** trains on the evaluated
  distribution, but explores less than the published recipes, may hold back exploration-leaning
  algorithms (Clip-Higher, RS-GRPO), and its top-p truncation makes rollouts slightly off-policy.
- **Scoring truncated rollouts by their box (verl default):** rewards an answer evaluation rejects.

## History

- 2026-10-28, first form: rollouts and in-loop validation sampled at evaluation's 0.6 / 0.95, read
  from the suite (user, D3 "use the 0.6 / 0.95, equal to evaluation"). M1.3 smoke attempts 1–3 ran
  with it.
- 2026-10-28, revised by decision 017 (user): rollouts at 1.0 / 1.0 for exploration; validation
  stays at evaluation sampling; DPO preference data at 1.0 / 1.0 (amendment, following
  Online-DPO-R1). Smoke attempt 4 ran with it.
