# 015 — Backport vLLM's FP8 wake-up fix for hybrid models into the pinned vLLM 0.24.0

- Date: 2026-10-28
- Status: proposed
- Milestone: M1 (task M1.3; amends decision 001's "pinned versions only" with one backported fix)
- Code: `soe/workers/rollout/vllm_patches.py`, selected by the verl override
  `+actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls`

## Context

The first verl smoke run (job 3222631, 2B, LoRA, 2×A40) failed on the first vLLM wake-up, before any
rollout:
- `init_fp8_kv_scales` raised `'list' object has no attribute 'zero_'`;
- verl puts vLLM to sleep during training and wakes it for each rollout;
- with an FP8 KV cache, vLLM 0.24.0 zeroes every per-layer cache entry on wake-up, assuming a
  tensor, but Qwen3.5's linear-attention (DeltaNet) layers hold a list of state tensors.

Evaluation never sleeps, so M0 did not hit this. Upstream fixed it in PR #41602 (merged 2026-07-29),
after v0.25.1; verl 0.9.1 pins vLLM 0.24.0. Issue #49237 (open) reports a related failure in later
builds.

## Decision

Apply the upstream fix as a backport, without touching the image:
- `vllm_patches.py` wraps `GPUModelRunner.init_fp8_kv_scales` so that the cache entries are a flat
  list of tensors for the duration of the call, then restores the runner's own structure. vLLM's
  function itself still runs, including the reset of the FP8 k/v scales to 1.0, which is the fixed
  scale evaluation uses (decision 009).
- The patch is applied in each vLLM worker process by importing the module: verl names a worker
  extension class, and soe's overrides name `SoeColocateWorkerExtension`, a subclass of verl's with
  no other change.
- The module refuses to load on any vLLM other than 0.24.0, so a version upgrade must drop it.
- Tests: the upstream PR's own cases, run against the patched 0.24.0 in the container
  (`tests/workers/rollout/test_vllm_patches_in_container.py`); a pure helper test in CI.

## Consequences (including what could confound the algorithm comparison)

- Every algorithm runs the same patched engine, so the patch cannot favour one.
- Sampling semantics are unchanged: the zeroed caches are rebuilt by prefill, and the scale stays
  1.0 as in evaluation. An FP8 run without sleep mode (evaluation) and with it (RL) therefore
  quantise the same way.
- One more thing to remove at the next vLLM upgrade; the version check makes that impossible to
  forget.

## Alternatives considered

- **Keep vLLM resident (no sleep mode):** no wake-up path, but the KV cache memory stays reserved
  during training. That changes the memory picture the G0 study measures and hurts 4B on 4×A40.
- **Upgrade vLLM:** breaks verl 0.9.1's tested set and needs an image rebuild; not in mid-milestone.
- **BF16 KV cache for rollouts:** reverses decision 009 (your choice of FP8 everywhere).
