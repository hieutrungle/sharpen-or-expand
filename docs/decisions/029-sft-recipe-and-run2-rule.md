# 029 — Pre-registered SFT recipe (run #1) and the run-#2 rule

- Date: 2026-11-04
- Status: proposed
- Milestone: M2 (task M2.9; roadmap D5 and D6, both answered (a) on 2026-11-03). Supersedes decision
  028's line "validation loss is computed once per epoch" for runs that save (item 3 below).
- Config and code: `examples/sft/m2_run1_qwen35-4b.yaml`, `soe/trainer/sft/args.py`,
  `tests/trainer/sft/test_args_on_cpu.py`

## Context

M2 freezes one SFT checkpoint from which every DPO and RL run starts. A second SFT run is allowed
once, and only for a reason fixed before run #1's results exist; otherwise the SFT recipe could be
tuned by looking at results. D5 fixed the recipe (a conventional starting point, not tuned) and D6
the four criteria, measured only on the RL validation split and the training logs, never on Levels
A, B or C. M2.8 measured the speed: 14.08M tokens/h, so run #1 takes about 8.8 h in one job. Three
points were open in D6's text: criterion 1 needs three validations but training validated once;
nothing said what happens when several criteria hold; and criterion 3's change shrinks the data
below 25K examples.

## Decision

1. **Recipe of run #1** (`examples/sft/m2_run1_qwen35-4b.yaml`, seed 1): Qwen3.5-4B-Base at
   revision `1001bb4d…`, full BF16 (decision 025) with FSDP2 over 4 A40s on fc098 (decision 028).
   One epoch of the M2.5 SFT set (`train.jsonl`, 25,551 examples, sha256 `0264027c…`). AdamW (torch
   fused; betas 0.9, 0.999; eps 1e-8), learning rate 1e-5, 24 linear warmup steps (3% of 799, rounded
   up) and cosine decay to 0, weight decay 0.0, gradient clip 1.0. 32 sequences per optimizer step
   (4 GPUs × 1 × 8 accumulation steps), 799 steps; `max_length` 18,432 (no example is longer).
   Liger loss on completion tokens only. A save every 290 steps (about 3 h), the last two kept.
2. **The frozen checkpoint is run #1's final export** (step 799). No checkpoint is chosen by
   validation loss or any evaluation.
3. **Validation at every save.** The SFT validation loss (`val.jsonl`, 967 examples of 500 held-out
   pool problems) is computed at each save and by the Trainer at the last step: steps 290, 580 and
   799. Each validation is logged before its save, so a resumed job does not repeat it. This
   replaces decision 028's single validation for runs that save (about 9 extra minutes); runs
   without saves still validate once.
4. **Measurement for the rule (M2.11).** The base model and the SFT export, each on the RL
   validation split (`datasets/soe/v3.1/rl/val.jsonl`, 256 problems, decision 012) × 16 samples at
   the suite's sampling (soe-eval-v4: temperature 0.6, top-p 0.95, 16,384 tokens, request seed 0,
   the same prompt). The SFT pool is disjoint from these problems (decision 027). The base half
   may run during M2.10.
5. **Run #2 happens if and only if at least one criterion holds:**
   1. **Training failure:** a NaN or Inf in the logged training loss or gradient norm; or the mean
      training loss of the last quarter of steps (600–799) is not below that of the first quarter
      (1–200); or the validation loss rises at both later validations: val(580) > val(290) and
      val(799) > val(580).
   2. **Format failure:** fewer than 98% of SFT's stopped samples (finish reason `stop`) contain
      exactly one `</think>` and a `\boxed{}` answer after it.
   3. **Truncation not reduced:** SFT's truncation rate (share of its 4,096 samples that hit 16,384
      tokens) is not below the base model's.
   4. **Collapse:** base pass@1 − SFT pass@1 > 2 SE, or base pass@16 − SFT pass@16 > 2 SE. pass@k
      is the unbiased estimator per problem (n = 16), averaged over the 256 problems. SE is the
      standard deviation of the paired difference over 2,000 bootstrap resamples of the problems
      (seed 0), both models on the same resampled problems.
6. **Run #2's change is fixed now, and every criterion that holds adds its change:**
   - 1 or 4: learning rate 1e-5 / 3 (3.33e-6), everything else unchanged;
   - 2: fix the data rule that let the format through, rebuild with the M2.5 build otherwise
     unchanged;
   - 3: drop the train examples above the 90th percentile of train token lengths (nearest rank,
     counted as main_sft tokenizes); `val.jsonl` unchanged. This leaves about 23K examples, below
     the plan's 25K–50K range, and removes the longest, mostly hardest traces. Both are accepted
     for a repair run and reported with the base→SFT transitions.
   Steps, warmup (3%, rounded up) and the cosine horizon follow the new data size; seed 1, the save
   cadence and everything else stay. If changes combine, run #2 cannot name a single cause; it is a
   repair, not a compared arm, so that is accepted.
7. **At most one run #2.** If run #2 is triggered, its final export is frozen if it meets no
   criterion. If it meets one, no third run: a decision memo chooses between the two exports and
   says why.
8. **The verdict is mechanical.** M2.11 writes `docs/reports/m2_sft_run2_check.json` with one line
   per criterion: the values, the threshold and whether it holds, then the verdict.

## Consequences (including what could confound the algorithm comparison)

- One SFT checkpoint for every algorithm, fixed by this rule; nothing here differs between them.
- No evaluation level is used for any SFT choice. The RL validation split, which is also the G1
  stopping signal for RL, is used once here, as a yes/no on fixed thresholds.
- Criteria 3 and 4 measure what matters for this study: SFT on the base model's own correct traces
  (rejection-sampling fine-tuning) can sharpen pass@1 and narrow pass@16 (risk 2). A narrowed
  coverage that triggers criterion 4 is repaired once; a smaller narrowing stays and is reported as
  base→SFT transitions, so later "expansion" is always read against the base.
- Each criterion-4 test at 2 SE fires by chance about 2.3% of the time when the models are equal,
  so about 4.5% for the pair. Criterion 3 has no margin: if SFT does not cut truncation at all, the
  data or the recipe failed its first purpose.
- Run #1 takes about 9 h on fc098 (8.8 h measured plus two validations), one job with
  `CHAIN_LEFT=1` for an automatic resume. Run #2, if any, costs another run plus 5.6 A40-h for its
  check.
- The optimizer settings and the validation cadence are written out in `trainer_arguments`, so a
  library default cannot change them; `tests/trainer/sft/test_args_on_cpu.py` checks the run #1
  config against items 1 and 3.

## Alternatives considered

- **Level A pass@1 in the rule (D6 (b)):** rejected; no evaluation level is used for selection.
- **A 2B learning-rate sweep or lr 2e-5 (D5 (b), (c)):** rejected; a 2B sweep would not transfer,
  and criteria 1 and 4 cover a bad learning rate.
- **Drop the validation-loss clause** instead of validating at every save: loses the only held-out
  signal in criterion 1, for 9 minutes saved.
- **A priority order when several criteria hold** (only the first one's change): a lower-priority
  failure would stay in the frozen checkpoint.
- **For criterion 3, cut only down to 25,000 examples** (about the longest 2%): stays in the range
  but probably too weak to change truncation. **Learning rate ÷ 3 for criterion 3:** keeps the data
  but is unlikely to teach the model to stop.
- **Choose the best of the saved checkpoints by validation loss:** a selection step the rule does
  not need; the final export is the checkpoint.
