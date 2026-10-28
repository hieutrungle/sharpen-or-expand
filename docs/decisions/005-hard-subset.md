# 005 — Hard subset for K = 32/64: selection rule (pre-registered)

- Date: 2026-10-26
- Status: rule accepted (user, 2026-10-26: chose "4B-Base pass rate, independent seed" over two
  model-free rules); pre-registered: committed before any selection sample existed. The frozen
  file's sha256 is recorded in the suite and the report, so this note stays unedited.
- Milestone: M0 (task M0.6)
- Config: `examples/data_preprocess/m0_hard_subset.yaml`; rule in `soe/utils/dataset/hard_subset.py`

## Context
The plan reports pass@32/64 on a frozen 200–300 problem hard subset, chosen in M0. Large K only
distinguishes sharpening from expansion on problems the models do not already solve at K = 16, so
a subset defined by difficulty for the main model uses the 64-sample budget best. Choosing problems
by a model's own pass rate has one trap: the problems where the base model happened to score low
are, on a re-draw with the same random numbers, still scored low, while any other model is scored
on fresh draws. That biases every comparison toward "the trained model expanded" (regression to the
mean). Selection therefore uses draws that are independent of the evaluation draws.

## Decision
1. **Selector:** Qwen3.5-4B-Base at the revision pinned in `scripts/stage_assets.yaml`, with the
   frozen suite's prompt, chat template and sampling (temperature 0.6, top-p 0.95, max 16,384
   tokens; truncated = incorrect), except the **vLLM seed: 1** (evaluation uses 0).
2. **Candidates:** Level A (MATH-500, AIME 2024, AIME 2025, OlympiadBench-EN) and Level B: 2,228
   problems. Level C is never used for any selection and stays out.
3. **Score:** 16 samples per problem; c = number correct.
4. **Eligible:** c ≤ 4 (at most 25% correct), including c = 0. If fewer than 200 are eligible,
   the threshold becomes c ≤ 8, once. If still fewer than 200: stop and write a new decision.
5. **Size:** at most 250 eligible → take all. More → a stratified draw of 250 by problem set,
   proportional to each set's eligible count (largest remainder, ties to the set name that sorts
   first), `random.Random(20261026)` over sorted ids.
6. **Format halt:** if fewer than 80% of all selection samples contain a `\boxed{}` answer, stop
   before freezing: the subset would select format failures, not hard problems. Then the prompt
   or thinking mode is revisited in a new decision, before any problem list is looked at.
7. **Output:** `datasets/soe/v1/eval/hard_subset.jsonl` (original Problem records, sorted by id)
   and `docs/reports/m0_hard_subset_v1.json` (per-problem c, per-set rates, selected ids,
   sha256 values; no problem text). The suite gets `hard_subset` with that file's sha256.

## Consequences
- The hard subset mixes Level A and Level B problems; results on it are also reported per level.
- Each subset problem's first 16 evaluation samples are its main-tier samples (decision 004).
- The base model's evaluated pass rate on the subset is unbiased (fresh draws), but the subset is
  still defined by Qwen3.5-4B-Base. The Gemma replication (M8) uses the same subset, which is hard
  for Qwen by construction; Gemma results on it are read as "Qwen-hard problems".
- Selection costs one 16-sample pass of 4B-Base over 2,228 problems on Lane B (two A40 shards,
  resumable).
- Any change to the rule after selection samples exist needs a new suite name.

## For the report
"The hard subset (N problems) holds the Level A and Level B problems that Qwen3.5-4B-Base solved
at most 4 times in 16 samples, drawn with a sampling seed independent of evaluation so that its
evaluated pass rates are not biased by selection; the rule was fixed before sampling."

## Alternatives considered
- **Model-free, by source and difficulty** (AIME 2024/2025, MATH-500 level 5, OlympiadBench draw):
  no model dependence and could be frozen at once, but many problems would be solved by every
  model at K = 16 or by none at K = 64, wasting most of the 64-sample budget.
- **Model-free, AIME + OlympiadBench only:** the same weakness, and no in-distribution problems.
- **Selecting with the evaluation seed:** cheaper (reuse the base model's evaluation samples) but
  biased, as above.
- **SFT-checkpoint pass rates:** the plan fixes the subset in M0, before SFT exists.
