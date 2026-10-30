# 023 — SFT wiring: TRL on pre-tokenized prompt/completion examples, LoRA merged into the base layout

- Date: 2026-10-29
- Status: accepted (user, 2026-10-29: "LoRA now, full BF16 at M2"; "Drop them, and count how many")
- Milestone: M1 (task M1.8)
- Code: `soe/trainer/sft/data.py`, `soe/trainer/sft/trainer.py`, `soe/trainer/main_sft.py`,
  `soe/trainer/config/experiment.py` (`SftSpec`), `soe/trainer/rl/export.py` (shared steps),
  `examples/data_preprocess/m1_sft_dev_data.py`, `slurm/train.sbatch`

## Context

The frozen SFT checkpoint (M2) is the start of every RL run and DPO, so SFT must train on exactly
the format rollouts and evaluation use (decision 006) and save a checkpoint that main_eval and verl
load like the base. TRL 0.27's `SFTTrainer`, given prompt/completion text, tokenizes the prompt and
the prompt + completion separately and builds the loss mask from the prompt's length; it only warns
when the two disagree at the boundary. It also appends the tokenizer's EOS (`<|endoftext|>` for
Qwen3.5 Base) to each completion, while every soe sampler stops on `<|im_end|>`. Saving through
transformers would drop Qwen3.5's mtp head (not loaded by transformers) and could change the
tokenizer files.

## Decision

1. **Format.** An example is the suite-rendered user turn and the completion text (`reasoning
   </think>` then the answer). soe renders the prompt with the chat template and the suite's
   `enable_thinking` (ending in `<think>\n`), encodes prompt and completion apart, appends
   `<|im_end|>`, and gives TRL `input_ids` plus a `completion_mask`; TRL's collator sets every label
   outside the mask to -100. The prompt tokens are exactly the prompt's own encoding, and TRL adds
   nothing.
2. **Length.** Examples longer than `sft.max_length` (2,048 + 16,384 for the dev config) are dropped
   and counted (`runs/<id>/sft_data.json`), never truncated: a cut example would teach reasoning that
   never reaches an answer or a stop.
3. **Training mode.** BF16 LoRA on the language model (the config's lora block, as in RL), DDP over
   `sft.n_gpus` with torchrun, gradient checkpointing and Liger's fused cross-entropy (no 18K x 248K
   logits). Full-BF16 SFT is added with the G0 mode, before M2; until then the config refuses it.
4. **Checkpoint.** Rank 0 merges the adapters in FP32 and writes them with the RL export's steps
   (decision 018): tensors in the base snapshot's layout and dtypes, vision tower checked equal to
   the base, mtp head copied, the base's config, tokenizer and chat template copied, and
   `soe_export.json`, at `$SOE_DATA/checkpoints/<run_id>/step_<N>/hf`.
5. **Launch.** `slurm/train.sbatch` runs `soe.trainer.main_sft` for an `algorithm: sft` config; it
   records the job segment like main_rl and writes DONE. SFT has no resume yet (M2 adds it): a run
   directory that was started before is refused.

## Consequences (including what could confound the algorithm comparison)

- Every RL run and DPO starts from a checkpoint in the same layout as the base, so main_eval, verl
  and the RL export treat SFT and base checkpoints alike.
- The dev data (roadmap D9 (a)) is 4B-Base text used to fine-tune 2B: wiring only.
- The dev SFT ran on 2×A30 (no A40 was free; user, 2026-10-29); it never enters conclusions.

## Alternatives considered

- **TRL's text prompt/completion path:** boundary merges can shift a token between prompt and
  completion, and TRL appends `<|endoftext|>` unless `eos_token` is changed, which also rewrites the
  saved tokenizer.
- **Conversational data with `assistant_only_loss`:** needs `{% generation %}` markers that the
  Qwen3.5 chat template lacks, and renders the assistant turn through the template, not as the
  sampler produces it.
- **Truncating long examples (TRL default):** see 2.
