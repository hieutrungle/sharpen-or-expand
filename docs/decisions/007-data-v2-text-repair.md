# 007 — Data v2: repair encoding debris in DAPO-derived problem text, splits frozen

- Date: 2026-10-28
- Status: accepted (2026-10-28). The fix was requested by the user ("do the fix for level B
  problems"); the in-place method was chosen after a full rebuild turned out to reshuffle Level B
- Milestone: M0 (found while drafting the M0.8 audit)
- Built by: `examples/data_preprocess/m0_repair_v2.py` from v1; report `docs/reports/m0_datasets_v2.json`
- Frozen in: `soe/trainer/config/eval_suite.yaml` (`soe-eval-v2`); v1 kept as `eval_suite_v1.yaml`

## Context
The audit sheet could not be written because a Level B problem contained a form feed. Upstream
DAPO text has encoding debris: "\frac" decoded as a form feed plus "rac" (604 of the 14,116 raw pool
rows), stray form feeds inside ordinary words ("fi<FF>ve", "the <FF>first"), and a few other
control bytes around "\le". In the v1 build this affects 42 of 994 Level B and 585 of 12,236 train
questions. Level A, Level C, every gold answer and all 250 hard-subset problems are clean.

Rebuilding v1 from scratch with clean text is not an option: the cleaned text changes which pool
problems are exact duplicates, so Level B's seeded draw runs over a different list. A trial rebuild
replaced 379 of the 994 Level B problems and dropped 27 of the 113 Level B problems in the frozen
hard subset.

## Decision
1. v1 stays exactly as built and reproducible. v2 is derived from v1 by repairing the text in place:
   "\x0crac" becomes "\frac"; every other control character (C0 except tab and newline, and C1)
   is removed. Problem ids, order, answers and levels do not change.
2. Decision 003's checks are re-run on the repaired text: exact duplicates, near-duplicates of
   evaluation and Level B problems (13-grams, 0.5; boilerplate n-grams recounted over train + Level
   B, a slightly stricter set), and the final coverage audit. Only train may lose problems; a leak
   into Level B or an evaluation set stops the build.
3. Result: Level B keeps its 994 problems (42 repaired); train has 12,232 (585 repaired, 4 dropped
   as exact duplicates of other train problems, no new leaks; residual coverage empty). Only the
   Level B and train files change (Level B sha256 `a4833e89b697…` → `de177396a6e9…`; train `f53e1cad352c…` →
   `4c4b4a20acc1…`); the 11 evaluation files and the hard subset are byte-identical.
4. Suite `soe-eval-v2`: the v1 suite with the new Level B hash and v2 paths. Sampling, prompt,
   scoring and tiers are unchanged (a test pins this).

4b. **Excluded problems** (user, 2026-10-28: "exclude, no refill; screen for more"). Six problems the
   M0.8 audit found with a wrong or non-unique gold stay in the frozen files but are never sampled
   or scored (`excluded_problems` in the suite, with reasons): OlympiadBench 2969 (gold is for
   T=10, the question states T=8), 2862 and 2863 (any of several labels is valid), and three Level B
   problems whose question asks for more than the gold holds. Public benchmarks are excluded, not
   re-gold'ed, to stay comparable with published numbers. Two of them were in the hard subset,
   which is not refilled: 248 of 250 remain. The suite is 2,467 problems, 51,376 samples.
   The gold screen then added 13 more (10 with a wrong gold, verified by enumeration or direct
   calculation; 3 ambiguous questions), all with reasons in the suite, and moved 4 verifier
   false negatives out of the hard tier only (decision 008). Format mismatches stay in
   the data and are fixed in the verifier. Decision 008 then excluded two golds that are format debris (2349, 2498) and
   moved 4 more problems out of the hard tier. Totals: 21 excluded, 2,452 problems, hard tier 233.

4c. **Effect on the 4B-Base reference** (`docs/reports/qwen35-4b-base_soe-eval-v2_report.json`).
   Level B pass@1 is 0.665 under v1 (994 problems) and v2 (991). On the 42 repaired problems
   pass@1 is 0.897 on the damaged text and 0.872 on the repaired text (672 samples each, 20
   problems change, none by more than 3 of 16): within seed noise, so the debris did not make
   these problems easier or harder. The hard subset is 0.0629 pass@1 on its 248 problems, the
   same as v1 restricted to them.

## Consequences
- The 4B-Base reference evaluation is carried to v2 by re-evaluating only the 42 repaired Level B
  problems (`main_eval --only-problems`) and reusing every other result (`scripts/merge_eval.py`,
  which refuses if any changed problem lacks a re-run).
- The M0.6 selection scored the 42 affected Level B candidates on corrupted text. None of them is in
  the hard subset; Level B's eligible count used for the stratified draw may differ by a few.
- RL training (M1 on) and the SFT decontamination (M2) use v2 train. Nothing was trained on v1.
- The probes in `scripts/` now read v2 train.

## For the report
"About 5% of the DAPO-derived problems contained encoding artefacts (a `\frac` decoded as a form
feed); we repaired them in place before any training, keeping every split and the hard subset fixed."

## Alternatives considered
- **Full rebuild with clean text:** cleanest pipeline, but reshuffles 38% of Level B and breaks the
  frozen hard subset.
- **Leave the text as is:** comparisons stay consistent, but 4% of Level B prompts (and 5% of RL
  prompts) are garbled, which no reader would accept once noticed.
- **Drop the affected problems:** loses 42 Level B problems and still changes the frozen file.
