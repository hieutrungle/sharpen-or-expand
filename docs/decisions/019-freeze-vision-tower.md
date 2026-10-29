# 019 — Freeze Qwen3.5's vision tower in verl's actor (verl's flag is not implemented)

- Date: 2026-10-28
- Status: accepted (user, 2026-10-28: "Freeze it in soe")
- Milestone: M1 (task M1.4)
- Code: `soe/workers/rollout/verl_patches.py`, named in the overrides as Ray's
  `ray_kwargs.ray_init.runtime_env.worker_process_setup_hook`

## Context

The M1.4 full-BF16 identity run (job 3222855, 2B, learning rate 0) saved a checkpoint whose
optimizer had state for the vision tower; the export refused it ("model.visual.patch_embed.proj.weight
was trained"). verl 0.9.1's Qwen3.5 forward runs a dummy image through the vision tower on text-only
batches and adds `0.0 * image_embeds.mean()`, so every rank produces gradients for FSDP. The
gradients are exactly zero, so with weight decay 0 the weights stay equal to the base, but AdamW keeps
FP32 gradients and two moment buffers for 0.334B parameters (4B model), about 4 GB, and a non-zero
weight decay would shrink weights nothing trains. verl declares `actor.freeze_vision_tower`, but no
code reads it. LoRA runs are unaffected: the whole base is frozen there.

## Decision

- A Ray worker setup hook wraps verl's `FSDPEngine._build_fsdp_module`, which runs before the
  optimizer is built, and sets `requires_grad=False` on every `model.visual.*` parameter. Torch's
  optimizer then creates no state for them, and FSDP keeps no gradient for them.
- The hook refuses any verl other than 0.9.1, so an upgrade re-checks whether verl's own flag works.
- The export keeps its checks: no vision parameter may have optimizer state, and every vision tensor
  must equal the base bit for bit (decision 018).

## Consequences (including what could confound the algorithm comparison)

- Full BF16 and LoRA now train the same set of weights (the language model), and the M1.10 memory
  study measures full BF16 without about 1 GB per GPU (4 A40s) of state that never changes a weight.
- Every algorithm runs the same patched actor.
- The FP32 vision weights themselves stay on the GPUs (verl keeps the model in FP32), about 1.3 GB
  in total for 4B.

## Alternatives considered

- **Leave the tower in the optimizer:** no patch, but full BF16's measured memory includes useless
  state (tilting G0 toward LoRA), and full-mode configs would need weight decay 0 as a standing rule.
- **verl's `freeze_vision_tower`:** not implemented in 0.9.1.
