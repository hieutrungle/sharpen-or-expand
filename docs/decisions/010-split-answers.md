# 010 — Multi-part answers boxed separately count as one answer

- Date: 2026-10-28
- Status: accepted (user)
- Milestone: M0.8 (amends 004's answer extraction)

## Context

Decision 004 scores the last `\boxed{}` only. Some completions box the parts of a multi-part
answer separately: "`\boxed{64} \quad \text{and} \quad \boxed{\frac{1}{4}}`", two triples in two
boxes, two sequences in two boxes. The last box alone is then judged wrong.

Measured on the 50,416 soe-eval-v3 samples of Qwen3.5-4B-Base (`scripts/split_answers.py`):
28,388 stopped; 1,909 have two or more distinct boxes; 248 of those have a wrong last box. Joining
the last 2 boxes recovers 109, the last 3 recovers 12, the last 4 none. All 121 are OlympiadBench
problems whose gold is multi-part (24 problems). In 11 more samples an earlier single box was
right and the model later changed its answer; those stay wrong.

## Decision

When the last box is wrong, the verifier also compares the last two, then three, boxes of the
final answer (the text after the last `</think>`) joined with commas. The boxes must be distinct
and non-empty. The joined string becomes the sample's extracted answer (also for maj@k). It is the
same function for evaluation and the RL reward. The report stage now re-scores from the stored
completion, not from the stored answer, so extraction changes reach old samples too
(`scripts/verifier_diff.py` likewise).

## Consequences (including what could confound the algorithm comparison)

- v3 reference re-scored at `d72167c`: exactly the 121 measured samples change, all incorrect to
  correct, each checked against its gold. OlympiadBench pass@1 0.626 → 0.633, Level A 0.740 →
  0.744, hard tier 0.0658 → 0.0695 (olympiadbench_en/2427 alone has 50 of the flips). Levels B, C
  are unchanged.
- Only adds acceptances. A joined answer must equal the gold as a whole, so boxing extra
  candidates is not rewarded; a box left in the reasoning cannot complete the final answer.
- The FP8 vs BF16 comparison (009) was made before this change with one scorer on both sides and
  stays valid as a protocol comparison.

## Alternatives considered

- **Last box only** (004 as is): a known underestimate of about 0.5 points on OlympiadBench.
- **Any box anywhere**: would reward a model that changes its mind; rejected.
