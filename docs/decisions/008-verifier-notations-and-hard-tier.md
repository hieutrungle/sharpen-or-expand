# 008 — Verifier accepts audited notations; measurement errors leave the hard tier

- Date: 2026-10-28
- Status: accepted (2026-10-28)
- Milestone: M0.8 (amends 004's scoring and 005's hard subset)

## Context

The M0.8 hand audit (200 base-model answers) found 38 verifier false negatives and no false
positives; the gold screen (`scripts/gold_screen.py`) found 9 more. Most are correct answers in
another notation: a percent sign, a column vector written as a tuple, labels ("w=16, d=3"), words
between answers ("7 goats and 4 helicopters"), ordinals ("12th grade"), a one-variable inequality
for an interval, two sets for a flat list, a family of solutions written with n instead of k.
The hard subset (005) was selected on the verifier's pass rate, so the same errors put some
problems into it.

## Decision

1. **Verifier.** `soe/utils/reward_score/math_verify.py` keeps math-verify's comparison and, when
   it rejects, compares again after rewriting both answers to one notation (`normalize`, with a
   percent sign either dropped or read as /100). A pair is accepted if either comparison accepts
   it, so the rewrites only add acceptances. Each audited case is pinned in
   `tests/utils/reward_score/test_math_verify_on_cpu.py`, with guards that the rewrites keep wrong
   answers wrong. The same verifier gives the RL reward and the evaluation verdicts.
2. **Report re-scores.** `main_eval --stage report` judges every stored answer again with the
   current verifier and records the scoring commit and how many verdicts differ from sampling time.
   A verifier change reaches old samples with a CPU report run, no re-sampling.
3. **Hard tier.** A problem that met 005's rule (c ≤ 4/16 on selection seed 1) only because the
   verifier rejected correct answers leaves the hard tier (`removed_from_hard_subset`) but stays in
   its problem set at the main tier. The test is the rule itself under the fixed verifier: its
   selection-seed count is above 4/16. No refill: a refill would be a second, post-hoc draw.
4. **Exclusions** (with 007 §4b). Wrong or ambiguous gold is excluded from the suite. Two golds are
   format debris that no general rule can read (OlympiadBench 2349 `221,$8`, 2498
   `\frac{1}{2},\sqrt{x}`); they are excluded too rather than special-cased in the verifier.

## Consequences (including what could confound the algorithm comparison)

- Verifier check on all 87,888 stored 4B-Base samples (`scripts/verifier_diff.py`): 703 verdicts
  change, all incorrect to correct, from 83 distinct answers on 22 problems; every one was checked
  by hand and is the gold answer in another notation. No verdict changes from correct to
  incorrect. Of the 200 audit items, 30 of the 38 false negatives are now accepted; the 53 judged
  not equivalent and the 107 correct stay as they were.
- Still rejected: answers to excluded problems, and the four families of 2501 written as two
  with period π/2 (V039, pinned as a known rejection).
- Hard tier: 8 problems leave it (4 found by the gold screen, 4 more by the selection-seed
  re-check). With 9 hard problems excluded, 233 of the frozen 250 remain; the file and its hash
  are unchanged.
- Known limit kept: math-verify compares a bare comma list as a set (order and repeats ignored).
  The audit found no false positive from it; changing it would reject correct unordered answers.
- Every checkpoint is scored by this verifier, including RL rewards from M1 on, so the change
  cannot favour one algorithm. Reports scored before this note (soe-eval-v1 and the first v2
  reference) are not comparable with later ones; `scoring.git_commit` in each report says which
  verifier produced it.

## Alternatives considered

- **Re-gold the problems** to the model's notation: edits public benchmark data and would need a
  per-problem judgement each time; rejected (007 §4b).
- **A general LLM judge for equivalence:** slower, non-deterministic, and itself needs auditing.
- **Keep the hard subset as frozen:** leaves 8 problems the base model solves in 5–16 of 16
  selection samples, which dilutes the K = 32/64 tier the plan relies on.
