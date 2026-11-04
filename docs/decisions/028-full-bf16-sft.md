# 028 — Full-BF16 SFT: FSDP2 through the Trainer, sharded checkpoints, resume, full export

- Date: 2026-11-04
- Status: proposed
- Milestone: M2 (task M2.6; roadmap D6, D12). Supersedes decision 023 items 3 (training mode) and
  5 (no resume); its format, masking, length rule, memory and dtype items stand.
- Code: `soe/trainer/sft/trainer.py`, `soe/trainer/sft/args.py`, `soe/trainer/sft/resume.py`,
  `soe/trainer/main_sft.py`, `soe/trainer/config/experiment.py` (`SftSpec`),
  `examples/sft/m2_dev_full_qwen35-2b.yaml`

## Context

G0 (decision 025) fixed full BF16 for every 4B run, SFT included: FP32 master weights, gradients
and AdamW states, BF16 compute, no offload. M1.8's SFT path (decision 023) trains BF16 LoRA under
DDP, saves only a final export, and refuses a started run. The 4B SFT run is one epoch of 25,551
examples (M2.5), longer than a job is safe to be without saves, and must chain like the RL runs.
DeepSpeed is not in the image (uv.lock), so sharding goes through FSDP. The pinned stack
(transformers 5.9, accelerate 1.15, TRL 0.27) supports FSDP2 in the Trainer, with sharded
checkpoints and resume; three of the settings it needs are not exposed by `TrainingArguments`.

## Decision

1. **Sharding.** `training_mode: full_bf16` SFT runs FSDP2 through the Trainer: `fsdp="full_shard
   auto_wrap"`, `fsdp_config.version: 2`, `Qwen3_5DecoderLayer` as the wrap unit (the frozen vision
   tower stays in the root unit), reshard after forward. LoRA stays for development; QLoRA is
   refused for SFT (no such path).
2. **Precision (D6).** The model loads in FP32: exact master weights, the base's F32 tensors
   included, so decision 023's dtype restore is not needed. Compute is BF16 and gradients reduce in
   FP32 (`MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32, output_dtype=bf16)`). The
   policy is set on accelerate's FSDP plugin after the Trainer is built and before `train()` wraps
   the model: `TrainingArguments` has no field for it, and accelerate's `bf16` default would reduce
   in BF16.
3. **What trains.** The vision tower is frozen before wrapping (decision 019's rule); mtp is not
   loaded by transformers and is copied at export.
4. **Memory.** FSDP-native activation checkpointing (`fsdp_config.activation_checkpointing`); the
   Trainer's `gradient_checkpointing` is off (it adds an all-gather per layer in backward). Batch
   size 1 per GPU, no packing, Liger's fused cross-entropy (decision 023 item 6).
5. **Checkpoints.** Every `sft.save_every_steps` optimizer steps (set from M2.8's step time so saves
   land every 2–4 h), `save_total_limit = sft.keep_last`, sharded state dicts
   (`FSDP_STATE_DICT_TYPE=SHARDED_STATE_DICT`, set before the Trainer builds the plugin, because
   `TrainingArguments` does not forward it), with optimizer, scheduler, RNG and trainer state
   (`save_only_model: false`). A checkpoint is complete when its `trainer_state.json` exists: the
   Trainer writes it after everything else.
6. **Resume.** `main_sft` leaves a finished run (DONE) alone, so a queued successor exits at once.
   A started run resumes from its newest complete checkpoint (`--resume-from`) and records
   `resumed_from_step`; with no complete checkpoint it starts again from step 0. The Trainer
   restores model, optimizer, scheduler and RNG state and skips the batches already trained (fixed
   `data_seed`), so the example order continues. Jobs chain through `slurm/train.sbatch`
   (`CHAIN_LEFT`), unchanged.
7. **Example-id log.** Every training micro-batch appends the train-set indices of its examples and
   the optimizer step it counts toward to `runs/<id>/examples/segment-<job>-rank<r>.jsonl`.
   `soe.trainer.sft.resume.continuity` checks that steps redone after a resume repeat the same
   examples on every rank (a step cut off by a kill may be a prefix).
8. **Export.** After training, the full FP32 state dict is gathered to rank 0 (accelerate's FSDP2
   `get_state_dict`, CPU offload) and written with `like_base`/`write_export`, `training: "full"`,
   after `check_trained(is_lora=False)`. Every tensor outside the language model must equal the
   base, the base's dtypes are written back, and the base's config, tokenizer and chat template are
   copied. A job resumed after training ended only exports; an existing export is not rewritten.
9. **Config.** `SftSpec` gains `save_every_steps` (null only in development: no resume),
   `keep_last`, and a development-only `max_steps`. The M1.8 dev config states `null`/`1`
   explicitly.

## Consequences (including what could confound the algorithm comparison)

- Every RL run and DPO starts from one SFT checkpoint, so nothing here differs between algorithms.
  The SFT checkpoint is in the base layout and dtypes, as with LoRA.
- Resume is continuity, not bitwise equality (D12): BF16 compute and collective order can differ
  across jobs. M2.7 checks example order, learning rate, step count and loss noise.
- FP32 master state costs memory (about 18 GB per A40 for 4B on 4 GPUs, decision 025's arithmetic)
  and FP32 reduction costs bandwidth; M2.8 measures both before run #1.
- **Testing limit.** accelerate enables FSDP only for multi-accelerator setups, never on CPU. The
  container test (`tests/trainer/sft/test_fsdp_in_container.py`) therefore runs the full-mode code
  on CPU in one process without FSDP: FP32 load, frozen vision tower, example log, resume
  (examples, learning rate and loss continue), and the full export through `like_base` on a tiny
  Qwen3.5. Its FSDP2 variant (two ranks, Liger, fla kernels) skips without 2 GPUs, and is the
  first step of M2.7.
- Validation loss is computed once per epoch (`eval_strategy: epoch`), i.e. at the end of training:
  run #1's single epoch, or a capped development run (2B smoke: about 4 min for 967 examples).

## Alternatives considered

- **DeepSpeed ZeRO-3:** not in the pinned image; adding it would change the frozen stack
  (decision 025).
- **FSDP1:** the older wrapper with flat parameters; FSDP2 is what accelerate 1.15 and torch 2.11
  develop, and it handles frozen and trainable parameters in one unit.
- **BF16 load with accelerate's FP32 upcast:** accelerate upcasts trainable parameters to FP32, but
  only after the BF16 load has rounded the base's F32 tensors.
- **Full (rank-0) state-dict checkpoints:** a gather and a 16 GB write at every save; sharded saves
  are written by every rank in parallel, and only the final export is gathered.
