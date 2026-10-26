# 001 — GPU stack: verl 0.9.1 for RL, TRL 0.27 for SFT/DPO

- Date: 2026-10-26
- Status: proposed
- Milestone: M0

## Context
Five algorithms must be compared with nothing else changing. DAPO needs dynamic sampling,
Clip-Higher, token-level loss and overlong shaping as independent switches for the leave-one-out
ablations. Hand-written versions of these would themselves be a confound.

## Decision
Pin verl 0.9.1's own tested set (from its `vllm` and `fsdp` extras): torch 2.11.0+cu130,
vLLM 0.24.0, transformers 5.9.0, flash-attn 2.8.3 (verl wheelhouse, cp312), flash-linear-attention
0.5.2 (Qwen3.5's linear-attention layers), TRL 0.27.0, Ray 2.59.0. CUDA 13.0 base image
(`nvidia/cuda:13.0.2-devel-ubuntu24.04`, as in verl's reference image), driver ≥ 580.

Each algorithm is a switch in the verl 0.9.1 config (keys checked in the 0.9.1 wheel; the exact
override set is built and tested in M1):

| Algorithm | verl settings |
| --- | --- |
| GRPO | `algorithm.adv_estimator=grpo`, `actor.loss_agg_mode=seq-mean-token-mean`, symmetric clip |
| Dr.GRPO | GRPO + `algorithm.norm_adv_by_std_in_grpo=false`, `actor.loss_agg_mode=seq-mean-token-sum-norm` |
| DAPO | `algorithm.filter_groups.enable` (dynamic sampling), `actor.clip_ratio_high > clip_ratio_low` (Clip-Higher), `actor.loss_agg_mode=token-mean`, DAPO reward manager `overlong_buffer` |
| RS-GRPO | custom advantage estimator in `soe/trainer/rl/core_algos.py`, registered with `@register_adv_est("rs_grpo")`, β from config |
| SFT, DPO | TRL 0.27 `SFTTrainer` / `DPOTrainer` (same image) |

Rewards for RL and scores for evaluation both go through `soe/utils/reward_score/math_verify.py`
(`reward.custom_reward_function` → `soe/workers/reward_manager/math_reward.py`).

## Consequences
- vLLM is the rollout engine (the plan allows vLLM or SGLang).
- `uv.lock` routes torch to the PyTorch cu130 index; uv applies sources only to direct
  dependencies, so the `train` extra lists torch/torchvision/torchaudio/flash-attn explicitly.
- flash-attn is pinned by **direct URL and sha256** to the wheelhouse release
  `flash-attention-v2.8.3-py3.12-torch2.11.0`, never through the wheelhouse index: the index
  re-points the same filename to builds for newer torch. The first image (2026-10-26) got the
  torch-2.13 build and failed to import (`undefined symbol c10::ValueError`); the original release
  that verl 0.9.1's own lock references has been deleted. `tests/special_sanity/` checks that the
  locked flash-attn build matches the locked torch.
- Library versions are recorded in every `run.json`; upgrades only between milestones, followed
  by `slurm/check_container.sbatch` on A100 and A40.

## Alternatives considered
- TRL throughout (as in Carry-On): has `loss_type` dr_grpo/dapo but no dynamic sampling, which
  would be custom code inside the ablation being measured.
- Latest vLLM 0.32: incompatible with verl 0.9.1 (transformers ≥ 5.16 vs verl < 5.11).
