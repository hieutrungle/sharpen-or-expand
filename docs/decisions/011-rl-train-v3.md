# 011 — RL train v3: hand-reviewed copies of evaluation problems removed

- Date: 2026-10-28
- Status: accepted (user: option 2, reviewed by hand, "do not drop lots of valuable data")
- Milestone: M0.9 (amends 003's near-duplicate rule for the RL train pool)

## Context

Decision 003 drops a train problem when it holds >= 50% of an evaluation or Level B problem's word
13-grams. The M0.9 report (`scripts/contamination_report.py`,
`docs/reports/m0_contamination_rl.json`) found copies the rule misses: LaTeX formatting breaks the
13-word runs (`$1/6$` vs `\frac{1}{6}`), and DAPO re-encodes answers ("all pairs" becomes "the sum
of all elements"). Of 354 pairs sharing >= 20% of 13- or 8-grams it classifies 52 as identical
after formatting, 33 as the same data reworded or re-encoded, 269 as different numbers.

## Decision

Option 2 with a hand review: every pair that is identical (E), reworded (R), or has the same answer
was read side by side (121 pairs, 120 train problems; `docs/reports/m0_train_v3_review.json`).
A train problem is dropped only when it is the same problem as the reference: reformatted or
reworded (86 pairs), the answer re-encoded (21), or one part of a multi-part reference (1). It is
kept when it is a variant with other data, asks a different question on a similar setup, or only
shares answer-format boilerplate (13 pairs, 12 train problems).

Result: `datasets/soe/v3/rl/train.jsonl`, 12,124 problems (12,232 − 108; built by
`examples/data_preprocess/m0_train_v3.py`, report `docs/reports/m0_datasets_v3.json`). The
evaluation files and Level B are unchanged (still v2), so the suite and every evaluation result
stand. Every RL run and preference set uses v3 train.

## Consequences (including what could confound the algorithm comparison)

- The dropped copies protect 74 Level B and 24 Level A problems; without the fix, RL would have
  trained on the problems these levels measure. Re-check on v3 (`docs/reports/m0_contamination_rl_v3.json`):
  0 residual at the frozen rule, 0 identical-after-formatting pairs, and the option-2 rule finds
  exactly the 12 reviewed keeps.
- Not removed on purpose: variants with different data (option 3 would drop 79 more, option 4
  230 more). They teach the same technique on other numbers, as any training data does; they are
  listed in the v3 report for later analysis of Level B gains.
- The review is a judgement by one reader; the pairs, verdicts and reasons are committed so it
  can be audited. Pairs below 20% n-gram overlap were not read.
- The review also found a wrong Level B gold (dapo/254353b7…: gold 2 is impossible). It stays in
  the frozen v3 suite and is noted for exclusion at the next suite revision (one of 983 Level B
  problems).
- The SFT data (M2) must be checked the same way, against all three levels, before SFT.

## Alternatives considered

- **Option 1, identical text only (52):** misses reworded and answer-re-encoded copies.
- **Options 3–4 (187 or 338):** remove useful variants with different data.
- **Excluding the evaluation problems instead:** changes the frozen suite and public benchmarks.
