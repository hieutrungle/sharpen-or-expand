# 018 — RL checkpoints: exported in the training job, LoRA merged by soe, mtp copied from base; rollout tokens are generated tokens

- Date: 2026-10-28
- Status: accepted (user, 2026-10-28: the three recommendations)
- Milestone: M1 (task M1.4; amends the roadmap's "export on a CPU node" and defines decision 013's
  rollout tokens)
- Code: `soe/trainer/rl/checkpoints.py` (manifest and token count), `soe/trainer/rl/checkpoint_callback.py`
  (verl hook), `soe/trainer/rl/export.py` and `scripts/export_checkpoint.py` (HF export),
  `soe/workers/rollout/agent_loop.py` (token records)

## Context

Lane B evaluates RL checkpoints as HF directories. Three facts of verl 0.9.1 shape the export:
- verl deletes old FSDP shards inside the next save (`max_actor_ckpt_to_keep`), so an export queued
  as a separate job can lose its input if it waits longer than a save interval.
- verl's `model_merger` does not merge LoRA: it writes the base weights (with `base_layer` renamed)
  and the adapter separately, so its "HF model" of a LoRA run is the base model. It also casts the
  shards to BF16 before any merge.
- Transformers does not load Qwen3.5's multi-token-prediction head (`mtp.*`), so RL never sees or
  trains it.

Decision 013 compares runs at equal rollout tokens, but verl keeps no token count, and its checkpoint
hook runs before that step's metrics are logged (validation runs in between), so a job killed in that
window would lose the step's count if it came from the metrics.

## Decision

1. **Export in the training job.** After each save, the checkpoint callback appends the manifest
   entry and starts the export as a background CPU process on the training node. `main_rl` also
   exports any checkpoint without an export at the start and end of each job segment, so a job that
   ends mid-export loses nothing. Lane B only evaluates.
2. **soe merges LoRA.** The export reads the FSDP shards in FP32, merges each adapter as
   W + (alpha / r) · B · A in FP32, and writes BF16 tensors with the base's names and layout.
3. **mtp is copied from the base.** Every export holds the base snapshot's `mtp.*` tensors unchanged,
   and its provenance file records them as untrained. The export's file set therefore equals the
   base's, and the identity test covers every tensor.
4. **Rollout tokens are generated tokens of training rollouts:** the response tokens of every
   training rollout, a truncated one counted at its 16,384 tokens. Prompt tokens and validation
   rollouts are excluded. The soe agent loop records each training rollout's step and response
   length; a step replayed after a resume is counted from its latest job segment only, once.
   Prompt tokens are left out because each prompt repeats G times per group, so they would make the
   count depend on the group size the G = 4 / 8 / 16 comparison varies.

## Consequences (including what could confound the algorithm comparison)

- Every algorithm uses the same export and the same count.
- Exporting on the training node uses its host memory and CPUs while training continues (about
  25–30 GB for 4B, no GPU). M1.10/M1.11 measure step time with exports running.
- The token record also checks the batch: each training step must hold exactly
  prompts per step × G rollouts, else the checkpoint callback fails the run (decision 016).
- A step-0 identity test (a 1-step run at learning rate 0, LoRA and full) checks that the export of
  an unchanged model is bit-identical to the base, and a test on a trained checkpoint checks that
  the vision tower and mtp head are unchanged and that only language-model weights were trained.

## Alternatives considered

- **Export as a separate Lane B CPU job (the roadmap's first plan):** keeps the training node free,
  but an export waiting longer than a save interval loses its shards; it needs `keep_last >= 2` and
  a sweeper.
- **verl's `model_merger`:** would evaluate the base model for every LoRA run.
- **Dropping mtp:** exports would differ in layout from the base reference.
- **Prompt + response tokens:** closer to training compute, but depends on G.
