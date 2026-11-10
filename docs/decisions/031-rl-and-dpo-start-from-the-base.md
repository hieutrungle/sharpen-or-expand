# 031 — Every RL and DPO run starts from the base model; SFT run #2 is an evaluated arm

- Date: 2026-11-10
- Status: proposed (user leaning to it, 2026-11-10; full audit asked for and done first)
- Milestone: M2 (after M2.11). Revises decision 025's "Start" row, decision 014 item 1 and decision
  017 item 5 (preference pairs), and settles the starting point decision 030 left open. Changes the
  plan's experiment tree and its "One starting point" rule.
- Evidence: `docs/reports/m2_base_start_audit.md` with `m2_base_format_audit.json`,
  `m2_prompt_identity.json`, `m2_rollout_groups.json`; `m2_sft_run2_check_run2.json`,
  `m2_sft_diagnostic_run2.json`

## Context

The plan put one frozen SFT checkpoint between Qwen3.5-4B-Base and every algorithm, mainly so that
RL would start from a model that already writes the demonstration format. M2 found two things.

1. **SFT run #2 is a little worse than the base and no better anywhere.** On the RL validation split
   (256 × 16): pass@1 0.637 vs 0.660 (paired drop 0.023, SE 0.008), pass@16 0.855 vs 0.852,
   truncation 34.5% vs 32.1%, loops 10.4% vs 5.1%; accuracy when stopped 0.972 for both. The held-out
   DeepMath split agrees. The whole pass@1 drop is stopping less, none is solving less.
2. **The base already writes the format.** The SFT data are the base's own verified samples, and the
   audit finds every stopped base sample with exactly one `</think>`, a boxed answer after it in
   99.75% (demonstrations 99.85%), the same answer-section length and no leaked turns. On the real
   verl rollout path at 1.0 / 1.0 the untrained base boxes and ends on `<|im_end|>` in 100% of
   stopped rollouts, and verl's rollout prompt ids equal the evaluation prompt ids for all 12,124
   RL problems.

So the SFT step buys no format and no capability, and costs a stopping problem that RL would first
have to undo: early pass@1 gains from the SFT start would partly be repairs, which reads as
sharpening without being it.

## Decision

1. **Start.** Every non-development 4B RL run (GRPO, Dr.GRPO, DAPO, RS-GRPO, every seed and
   ablation) and DPO starts from `Qwen/Qwen3.5-4B-Base` at revision
   `1001bb4d826a52d1f399e183466143f4da7b741b`, the staged Hub snapshot. `init_checkpoint` is null;
   `ExperimentConfig` refuses a non-development RL/DPO config that names one (development configs
   may still start from a dev SFT export).
2. **Freezing the start (M2.12, revised).** `docs/reports/m2_start_frozen.json` records the sha256 of
   every snapshot file and one digest over the sorted `name sha256` lines;
   `soe/trainer/config/frozen_start.yaml` holds the model id, revision and digest, one entry per
   model (Gemma's in M8). Every non-development RL/DPO config must name that model and revision, and
   `main_rl` (and the DPO entry) recompute the digest at launch before any GPU work. The snapshot's
   blob files are made read-only.
3. **SFT run #2 is an evaluated arm.** Its export (decision 030) is evaluated on the full
   soe-eval-v4 suite in M2.13 and reported with base→SFT transitions, as rejection-sampling
   fine-tuning on the base's own correct traces. No RL or DPO run starts from it.
4. **Preference pairs** (DPO, M3/M4) are sampled from the base policy, at 1.0 / 1.0 as before,
   with the suite's prompt and thinking mode.
5. **Prompt guard.** Because the format now rests on the prompt alone, an in-container test
   (`tests/trainer/rl/test_prompt_identity_in_container.py`) checks that verl's rollout prompt ids
   equal the evaluation sampler's for edge-case questions and every RL validation problem; it runs
   at every library or image change and before M3.
6. **Reporting.** Every RL and DPO result is read against the base, which is now also the start.
   Every pass@1 change is split into a stopping part and an accuracy-when-stopped part (the
   `scripts/sft_diagnostic.py` method) and shown with truncation, for every algorithm.
7. **Gemma (M8)** starts from the Gemma 4 E4B base as well, if a format probe before M8 (Lane B,
   about 1 A30-hour: the suite prompt in Gemma's template, 64 problems × 4) finds at least 80% of
   stopped samples with a boxed answer after the thinking (decision 006's halt threshold). If it
   does not, a new decision chooses between a Gemma SFT with the M2 recipe and a prompt change,
   and the asymmetry with Qwen is reported.

## Consequences (including what could confound the algorithm comparison)

- Every algorithm still starts from one checkpoint; the comparison's control is unchanged.
- No SFT artefact sits between the base and RL: the loop-repair confound is gone, and "sharpen" and
  "expand" are measured against the model the algorithms actually started from.
- Early RL from the base is mostly learning to stop within 16,384 tokens: 36% of step-1 rollouts
  are truncated, accuracy when stopped is about 95%, and 90% of the rollouts in all-wrong groups
  are truncated (M1 4B dumps). That is a real RL effect, but a stopping gain can look like a
  capability gain in pass@1; item 6 reports it separately.
- At G = 8 about 44% of groups are all correct and 16% all wrong on RL train v3.1 (pooled over 14
  M1 steps), so about 40% carry an advantage. This is a property of the RL set, the same from
  either start; DAPO's dynamic sampling acts on it and GRPO does not.
- The hard subset was selected by this same base (decision 005) with an independent seed, so its
  base rates are unbiased; it is hard for the start model by construction.
- External validity: the study becomes base-start ("zero") RL, the setting of the closest prior
  work on this question, rather than the SFT-then-RL pipeline. The SFT arm (item 3) keeps a
  measured answer to what rejection-sampling SFT does on its own.
- DPO from a base policy is less conventional than from SFT; its pairs come from the same model and
  sampler as the RL rollouts, which keeps the comparison matched. Pair construction (truncated
  losers) is still M4's decision.
- Cost: none on Lane A for Qwen; M8 saves a Gemma SFT run and its sampling if the probe passes.
  The decision-025 learning rate (D8: 1e-6, 10 warmup steps, weight decay 0.1) comes from DAPO,
  which trains from a base model.
- Decisions 025 (Start row), 014 and 017 (preference pairs) get dated revisions pointing here; the
  milestone plan's experiment tree, M2–M4 and M8 text and its "One starting point" rule change.

## Alternatives considered

- **Start from SFT run #2 (decision 030 as first drafted):** the standard SFT→RL pipeline, no plan
  change; but early RL gains would partly repair SFT's loops, and the start would be worse than the
  base at pass@1.
- **SFT start plus one GRPO control from the base:** tests whether conclusions depend on the start,
  at the cost of one full RL run of Lane A wall-clock, the study's tightest budget.
- **A third SFT run** (traces re-sampled at temperature 1.0): needs revising decision 029, a third
  use of the RL validation split for selection, 89–179 A40-h of sampling and about 7.5 h on fc098,
  for a step that would still add no format the base lacks.
