# M1.7: each RL test fails on a deliberately broken version

Recorded 2026-10-29 at commit `b629f2b` (branch `m1/rl-tests`).

## Automatic (every run of the container tests)

`tests/trainer/rl/test_loss_and_advantage_in_container.py` runs each check on verl 0.9.1's function
and on broken variants, which must fail (FASTER CPU job 3223657: 6 passed):

| Check | verl function | Broken variants rejected |
| --- | --- | --- |
| token-mean, seq-mean-token-mean, seq-mean-token-sum-norm equal hand values (2.5, 3.0, 1.25) | `core_algos.agg_loss` | mask shifted by one token; padding counted |
| aggregation is the same split over 2 ranks and 3 micro-batches as at once | `core_algos.agg_loss` with the engine's global counts | (equality, all three modes) |
| GRPO advantages = (r − mean) / (std + 1e-6), zero-variance group 0, Dr.GRPO r − mean, zero on padding | `v1.utils.compute_advantage_for_multi_trajectories` (GRPO) | std without epsilon (NaN on the zero-variance group); Dr.GRPO still dividing by std |

## By hand (scratch copy of the repo, one bug at a time, CPU tests)

| Bug introduced | Tests run | Result |
| --- | --- | --- |
| truncated rollouts rewarded (`correct = verdict.correct`) | `tests/workers/reward_manager` | 3 failed |
| answer read before the last `</think>` | `tests/workers/reward_manager`, `tests/utils/reward_score` | 1 failed |
| checkpoint manifest keeps the first entry of a step | `test_checkpoints_on_cpu.py` | 1 failed |
| keep-last pruning off by one | `test_checkpoints_on_cpu.py` | 1 failed |
| pruning ignores a missing export | `test_checkpoints_on_cpu.py` | 1 failed |
| resume always from step 0 | `test_segments_on_cpu.py` | 1 failed |
| a replayed step does not replace the killed job's record | `test_segments_on_cpu.py` | 1 failed |
| a step with a lost rollout passes the metrics | `test_metrics_on_cpu.py` | 1 failed |

The unmodified tree passes all of them.

## Found while writing the tests

- verl's batch keys are `{uid}_{session}_{output}`: each rollout of a group is its own session, and
  only a session's last output gets an advantage. Keys with one session per group would give every
  group a single member (advantage = reward); the toy batch follows verl's format.
- Without `loss_scale_factor`, `seq-mean-token-sum-norm` (Dr.GRPO's aggregation) divides by the
  padded width of each micro-batch, which varies with dynamic batching. Dr.GRPO's config must set a
  constant factor (`actor.loss_scale_factor`, e.g. the 16,384-token cap) when it is wired in.
- No stopped rollout in the M1.5 validation dumps lacks a box (0 of 1,232), so the reward fixtures
  take "no box" from truncated rollouts; the stopped no-box case is a hand-written unit test.
