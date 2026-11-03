# 025 — G0: full BF16 for every 4B run, and one shared 4B base config

- Date: 2026-11-03
- Status: proposed (training modes: user, 2026-11-03, "Full BF16" and SFT "Same as the RL mode";
  the memo `docs/gates/G0.md` awaits sign-off)
- Milestone: M1 (task M1.12, gate G0)
- Evidence: `docs/reports/m1_memory_study.json`, `docs/reports/m1_throughput.json`,
  `docs/reports/g0_schedule.json`

## Context

The algorithm comparison is only clean if every 4B run differs from the others in its algorithm
switches and nothing else. M1.10 and M1.11 measured full BF16 (B-F0) and BF16 LoRA r 64 (B-L) on
4×A40; the configs that ran are `examples/grpo_trainer/m1_throughput_{bf0,bl}_qwen35-4b.yaml`.

## Decision

1. **Training mode.** Qwen3.5-4B SFT, DPO and every RL run train in full BF16: BF16 compute with
   FP32 master weights, gradients and AdamW state (D6), no offload, the vision tower frozen
   (decision 019). QLoRA and LoRA stay development-only.
2. **Shared 4B RL base config.** Every 4B RL config sets these to the values below; a CPU test from
   M3 on checks it. Algorithm configs may differ only in the decision-001 switches (advantage
   estimator, loss aggregation, Clip-Higher, dynamic sampling, overlong shaping, RS-GRPO's β), in
   `name`, and in M7's group size.

   | Group | Values (from `m1_throughput_bf0_qwen35-4b.yaml`) |
   | --- | --- |
   | Start | the frozen M2 SFT checkpoint (`init_checkpoint`), Qwen3.5-4B-Base revision `1001bb4d…` |
   | Batch | 64 prompts × G = 8, 4 updates per step (decision 022), 2,048 prompt tokens |
   | Sampling | rollouts 1.0 / 1.0 (017); validation 0.6 / 0.95; 16,384-token cap, reward 0 when cut (014); FP8 KV (009); thinking mode (006) |
   | Loss | no KL term, no reference model (013); rollout correction off, `rollout_corr/*` logged (021); clip 0.2 unless Clip-Higher |
   | Engine | FSDP2, gradient checkpointing, remove padding, dynamic batch, `max_token_len_per_gpu` 36,864, fused log-prob on, weight-sync buffer 4,096 MiB (024) |
   | Hardware | 4×A40 on fc098, rollout TP 1, `gpu_memory_utilization` 0.7 |
   | Checkpoints | save every 6 steps (~3 h at M1's lengths; re-checked on the first M3 steps, kept within 2–4 h), `keep_last` 2, HF export at each save (018) |
   | Validation | 128 problems × 2 samples at every save (D2); G1 may resize it |
   | Stop | the rollout-token budget (020, with its 2026-11-03 revision) |

   Learning rate, warmup and weight decay are shared too, but M1 did not tune them (its 1e-6 was a
   placeholder); their values are fixed before M3 screening and recorded here as a revision.
3. **SFT and DPO** use the same mode and image on 4×A40 through TRL (decision 023's format); M2 adds
   the full-BF16 SFT path and its checkpoint resume.
4. **Frozen until M9** unless a dated revision here says otherwise: image `soe.sif` sha256
   `15824634257dd71030f07d7657cab9e55b8ab12581e8995a1f840ca9dbfd7196` (checked on FASTER
   2026-11-03, job 3231315), torch 2.11.0+cu130, vLLM 0.24.0, verl 0.9.1, transformers 5.9.0,
   TRL 0.27.0, peft 0.21.2, flash-attn 2.8.3.
5. **No 2×A100 variant yet.** The fallback pair was not measured. Full BF16 would very likely need
   optimizer offload there (12.1 GiB of training state per GPU on 4 GPUs would double on 2). If you
   request a block on the A100s, its variant is measured first and may differ only in speed knobs
   (TP, `gpu_memory_utilization`, offload, micro-batch token limit), never in the global batch.

## Consequences

- One run's hardware, mode and base values are fixed when it starts; a change between runs of one
  comparison block is a confound and needs a revision here first.
- The thin generation margin (1.5 GB of 48 GB free) means an OOM in M2/M3 is possible; the first
  responses are lowering `gpu_memory_utilization` to 0.65, then optimizer offload (B-F1), both as
  a revision here before any compared run starts.

## Alternatives considered

- **BF16 LoRA r 64:** 4 GB more GPU headroom and an easier fit on 2×A100, but 6% slower, host RAM
  at 208 of ~250 GiB, and a low-rank update would let "RL only sharpens" be read as a limit of the
  adapter rather than of RL.
- **A layered base config file that algorithm configs inherit:** fewer repeated lines, but the
  project keeps one flat YAML per experiment; a CPU test that compares the shared keys gives the
  same guarantee.
