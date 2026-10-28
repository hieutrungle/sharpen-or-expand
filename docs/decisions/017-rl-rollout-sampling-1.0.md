# 017 — RL rollouts sample at temperature 1.0, top-p 1.0; validation stays at evaluation sampling

- Date: 2026-10-28
- Status: accepted (user, 2026-10-28: "use the standard default to temperature 1.0 / top-p 1.0 for
  algorithm during RL roll-out because we want more exploration and more diversity"; validation at
  evaluation sampling, the recommendation). Amended the same day: preference (DPO) data also
  samples at 1.0 / 1.0 (user: "check if the literature use 1.0 / 1.0, if so use 1.0 / 1.0")
- Milestone: M1 (task M1.3)
- Supersedes: decision 014, point 1 (rollout sampling = evaluation sampling). Point 2 of 014, reward 0
  for a rollout that reaches the 16,384-token cap, stands.
- Enforced in: `soe/trainer/config/experiment.py` (`RolloutSpec.temperature` / `top_p` are required;
  `rollout_protocol` gives rollouts the config's sampler and validation the suite's) and
  `soe/trainer/rl/verl_overrides.py`

## Context

Decision 014 sampled rollouts at evaluation's 0.6 / 0.95, so RL optimised the distribution that is
later evaluated. Its own consequences named the cost: less exploration than the published recipes,
which may hold back the algorithms that lean on it (Clip-Higher, RS-GRPO), and curves that cannot
be compared directly with the papers in M3. The study asks whether RL expands what a model can
solve, so exploration during training matters more than matching the evaluation sampler.

## Decision

1. **Rollouts** sample at temperature 1.0, top-p 1.0, top-k off, for every algorithm, seed and
   ablation. The values are in each experiment YAML (`rollout.temperature`, `rollout.top_p`); runs
   compared with each other must have equal values (checked by the M1.6 run checker).
2. **In-loop validation** (the G1 stopping signal, decision 012) keeps the suite's evaluation
   sampling, 0.6 / 0.95, top-k off, so it measures what evaluation measures.
3. **Evaluation** is unchanged (decision 004: 0.6 / 0.95, 16,384 tokens).
4. As in 014, rollouts take no per-request seed; the vLLM engine seed is the run seed.
5. **Preference (DPO) pairs** are sampled from the SFT policy at 1.0 / 1.0 too (amendment, see
   below), so DPO learns from the same sampler as the RL algorithms it is compared with.

## Amendment, 2026-10-28: preference data

The user asked to follow the literature for DPO pair sampling and to use 1.0 / 1.0 if it does.
- Closest analogue: Online-DPO-R1 (Zhang et al. 2025, RLHFlow), iterative DPO with rule-based
  rewards from a Qwen2.5-Math base, compared with RL. Its pipeline (`run_iter_dpo.sh`) generates
  with `--temperature 1.0`, and its generator (`generation/gen_hf.py`) fixes `top_p=1.0`. A 2026
  study of iterative math alignment (arXiv 2602.05370) uses it as the standard baseline, sampling at
  T = 1.0.
- Other reasoning-DPO work varies: Iterative RPO (Pang et al. 2024) samples at 0.8, then 1.3 in
  later iterations; RLHF Workflow (Dong et al. 2024) uses half 1.0, half 0.7; DPO-VP (2025)
  anneals 0.7 → 1.0 → 1.2. No common value exists, so the closest setup decides.
- Preference pairs therefore use 1.0 / 1.0, top-k off, the 16,384-token cap, and the suite's prompt
  and thinking mode. Pair construction (which correct and incorrect responses form a pair, and
  whether truncated responses can be losers) is decided with the DPO pipeline in M4.

## Consequences (including what could confound the algorithm comparison)

- Every algorithm shares the rollout sampler, so it cannot favour one. The 014 handicap on
  exploration-leaning algorithms is gone, and M3 can compare entropy, zero-variance groups and the
  clip fraction with the published curves directly.
- At 1.0 / 1.0 the sampler is exactly the policy (verl's trainer applies the rollout temperature to
  its log-probs, and there is no top-p truncation), so PPO's importance ratio has no
  sampler/policy mismatch beyond numerics (FP8 KV cache, kernels).
- RL optimises a different sampler from the one evaluated. This is the usual setup (public RL
  models are trained near 1.0 and evaluated at 0.6 / 0.95), but the report states that pass@k
  measures the evaluation sampler, not the training one.
- Responses are expected to be longer and to hit the cap more often, especially for the base and SFT
  policies early in training. Truncated rollouts get reward 0 (014), rollout tokens per step rise,
  and runs fit fewer steps into an equal token budget. DAPO's overlong ablation (run only if
  truncation appears) becomes more likely. Truncation is logged per step.
- M1.10 (memory) and M1.11 (throughput) are measured at this setting. Smoke attempts 1–3 ran at
  0.6 / 0.95; the rollout/trainer log-prob gap and entropy are re-checked at 1.0 / 1.0.

## Alternatives considered

- **Keep 0.6 / 0.95 (decision 014):** trains on the evaluated distribution but explores less, and
  its top-p truncation makes rollouts slightly off-policy.
- **Validation at 1.0 / 1.0 as well:** follows the training sampler, but the stopping signal would
  then measure something other than evaluation.
