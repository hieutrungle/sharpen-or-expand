# 024 — M1.10 memory study: descending knob search, per-phase memory marks, fused log-prob

- Date: 2026-10-31
- Status: proposed (search order: user, 2026-10-31, "Descending search")
- Milestone: M1 (task M1.10)
- Code: `examples/grpo_trainer/m1_memory_{bf0,bl}_qwen35-4b.yaml`, `scripts/memory_probe_config.py`,
  `slurm/memory_probe.sbatch`, `soe/workers/rollout/phase_memory.py`,
  `soe/workers/rollout/verl_patches.py`, `slurm/common.sh` (`start_monitors`),
  `soe/trainer/config/experiment.py` (`HardwareSpec.fused_log_prob`), `scripts/memory_study_report.py`

## Context

G0 picks full BF16 or BF16 LoRA from peak memory, KV-cache headroom and speed on 4×A40 (M1.10). The
roadmap tunes two knobs per cell, rollout TP ∈ {1, 2} and `gpu_memory_utilization` ∈ {0.4 … 0.7},
on one-step probes. A 4B step on 4×A40 takes about 35–45 min, so the full grid is 8 probes per cell.
In the colocated engine, verl keeps the FSDP parameters and optimizer state on the GPU while vLLM
generates (unless offloaded), so the two compete for the same 48 GB: full BF16 holds about 13 GB of
FP32 master weights and Adam state per GPU. verl 0.9.1 logs one running peak of torch memory per
step, never reset, so it cannot say which phase needs the memory, and vLLM allocates outside
torch's allocator. Its log-prob pass also builds each micro-batch's full logits (15.5 GiB at
36,864 tokens and a 248K vocabulary; M1.8 job 3223874), unless `use_fused_kernels` is on.

## Decision

1. **Search.** For each TP, probe `gpu_memory_utilization` from the highest value down and stop at
   the first probe that finishes every phase; the chosen setting is the faster TP winner by rollout
   tokens/s. Lower values at that TP are recorded as "not run, dominated": more KV cache never slows
   vLLM, and each probe's peak KV-cache usage and preemption count show whether the cache was the
   limit. One job per cell runs the search (`slurm/memory_probe.sbatch`); each probe is a normal run
   directory, its checkpoint and export deleted afterwards.
2. **Probe.** One training step with an 8-problem validation pass; every other field is the cell's.
   One step still covers weight sync, generation, old log-prob, update, the save and its export.
3. **Phase memory.** The verl worker hook wraps `update_weights`, `compute_log_prob` and
   `update_actor`; at each entry and return it writes torch's peak allocated and reserved memory
   since the previous mark, per rank, then resets the peak (`runs/<id>/phase_memory/`). A sidecar
   outside the container samples nvidia-smi `memory.used` per GPU and the node's
   MemTotal − MemAvailable every 2 s (`runs/<id>/{gpu_memory,host_memory}/`); the report takes each
   phase's nvidia-smi peak inside rank 0's mark windows. Every run gets both, and its own copy of
   the job log (`runs/<id>/logs/`), from which vLLM's KV-cache size, maximum concurrency, usage and
   preemptions are parsed; vLLM's stats logging is therefore on in every run.
4. **Fused log-prob.** `hardware.fused_log_prob` sets verl's `use_fused_kernels` (torch backend). It
   is on in every M1.10 cell, as the roadmap's fixed "fused/chunked log-prob" intends, and off by
   default so earlier configs load unchanged. G0 decides whether it becomes the default.
5. **Updates per step.** The cells use decision 022's 4 updates per step, not the roadmap's older
   "one optimizer step per rollout batch": memory per micro-batch is the same, and the timings are
   then the ones M1.11 and the real runs need.

## Consequences

- About 3–4 probes per cell instead of 8: roughly 2 h of fc098 per cell, plus the measured run.
- verl's own `perf/max_memory_*` now means the peak of the update phase, not of the whole run.
- The sidecar adds about 70 KB per hour to a run directory; the job log copy is the job's output.
- A probe that fails names the phase after rank 0's last mark and the failing allocation from the
  log; a failure before the first mark is start-up, model load or the first weight sync.

## Alternatives considered

- **Full 8-point grid:** measures every point, at about 3× the probe cost for points that cannot be
  the fastest (user chose the descending search).
- **Per-phase peaks from verl's metrics:** verl keeps only the update's running peak and drops the
  old log-prob output's memory figures; vLLM's memory is invisible to torch in any case.
- **Fused kernels always on:** simpler, but would silently change the numerics of the M1 dev
  configs; a knob that G0 can flip keeps the change explicit.
