# 026 — Suite `soe-eval-v4`: one impossible Level B gold excluded; v3 samples re-scored, not re-sampled

- Date: 2026-11-03
- Status: accepted (user, 2026-11-03: M2 roadmap D7 (a), "accept the plan with your
  recommendations")
- Milestone: M2 (task M2.1; the plan's M2 suite revision)
- Code: `soe/trainer/config/eval_suite.yaml` (v4), `soe/trainer/config/eval_suite_v3.yaml` (v3,
  byte-identical to the file the v3 base reference was sampled with, sha256 `533c2053…`),
  `soe/trainer/main_eval.py` (`--rescore-from`, `--rescore-suite`, `common_suite`),
  `scripts/compare_eval.py`, `scripts/transitions.py`

## Context

The decision-011 review found that Level B problem `dapo/254353b7-e34c-4b96-99f5-0b51ec637c7f` has
an impossible gold (2). It appeared in an identical-text pair with an RL-pool problem whose gold was
329, and that RL-pool copy was dropped in v3. The problem stayed in the frozen v3 suite. The plan
(M2) asks for any suite revision to happen before the first SFT checkpoint is evaluated, with the
base reference re-scored on CPU, without re-sampling. Every later checkpoint uses that suite.

The problem is not in the hard subset: it met the selection rule but was not drawn
(`m0_hard_subset_v1.json`). `main_eval --stage report` could not re-score under a new suite. It
refuses a `plan.json` that names another suite, and it stops on samples of problems outside its
plan.

## Decision

1. **`soe-eval-v4` = v3 + one exclusion.** The problem joins `excluded_problems`, with the reason.
   Nothing else changes: problem files and their hashes, hard subset, sampling, prompt, scoring and
   tiers. A CPU test enforces exactly this difference. Level B's main tier goes from 983 to 982
   problems; the hard tier is unchanged.
2. **v3 is kept** as `eval_suite_v3.yaml`, byte-identical, for the runs that used it (M0's base
   reference, M1's development evaluations), as v1 and v2 were.
3. **Re-score without re-sampling.** `main_eval --stage report --rescore-from <v3 dir>
   --rescore-suite eval_suite_v3.yaml --suite eval_suite.yaml --output <new dir>`. It requires:
   - the given old suite file to be the one the source plan names (name and sha256);
   - the two suites to differ only in `name` and in added exclusions;
   - the same checkpoint;
   - a full (not `--only-problems`) evaluation with every shard finished.
   It drops the samples of newly excluded problems, counts them, and scores the rest with the
   current verifier, as every report does (decision 008). It writes `report.json` and
   `per_problem.jsonl` to the new directory. The new `plan.json` records the source directory, the
   source plan's sha256, the source suite, the sha256 of every shard file, and the dropped problems
   and samples. The new directory never holds samples.
4. **From M2 on, every checkpoint is evaluated on v4** and compared with the v4 base reference
   (`qwen35-4b-base_soe-eval-v4`). The "One yardstick" rule in the plan is revised to name v4 at M2
   close (roadmap D7).

## Consequences (including what could confound the algorithm comparison)

- Base and every later checkpoint are measured on the same 2,451 problems (2,452 in v3, minus one).
  The base is not re-sampled, so its numbers on the other 2,451 problems are unchanged; only Level
  B and the overall aggregates move, by one problem of 983.
- v3 and v4 numbers must never be compared problem by problem. Every report names its suite in
  its plan. `scripts/compare_eval.py` and `scripts/transitions.py` refuse two evaluations scored
  under different suites (`main_eval.common_suite`), and `transitions.py` also refuses a `--suite`
  file other than theirs. Re-running an M1 comparison therefore needs `--suite
  eval_suite_v3.yaml`.
- Build configs of earlier data (`m0_hard_subset.yaml`, `m1_rl_data.yaml`, `m1_sft_dev_data.yaml`)
  keep pointing at `eval_suite.yaml`. Their committed reports record the v3 name and sha256 they
  were built with; the prompt they render is identical in v4.
- RL configs read the protocol from `eval_suite.yaml`, which v4 leaves unchanged, so no RL run
  changes.

## For the report

"One Level B problem whose gold answer is impossible was excluded before the first SFT checkpoint
was evaluated (suite v4; 982 Level B problems). The base model was not re-sampled: its v3 samples
were re-scored under v4, and every later checkpoint is evaluated on v4."

## Alternatives considered

- **Correct the gold to 329 instead of excluding:** the RL-pool copy's gold is unverified, and
  editing a frozen problem file changes its hash. The suite already handles every known bad gold by
  exclusion (`excluded_problems`), never by editing a frozen file.
- **Re-sample the base on v4:** about 74 A40-hours for a one-problem change, and a new seed draw
  would move every number within noise.
- **Keep v3 and ignore the problem:** one wrong gold in 983 is within noise, but it is known, and
  the plan asks for the revision.
