# 002 — Evaluation sets: olympiad benchmark (Level A) and Level C

- Date: 2026-10-26
- Status: accepted (user, 2026-10-26: "use your recommendation")
- Milestone: M0 (tasks M0.3)

## Context
The plan fixes Level A as MATH-500, AIME 2024, AIME 2025 and "one olympiad-level benchmark", and
Level C as "a recent, contamination-resistant set, excluded from training and from all
hyperparameter selection". Qwen3.5 open weights were released on 2026-02-16; Alibaba publishes no
training-data cutoff, so only problems published around or after that date are safely unseen.

Evidence: `scripts/candidate_overlap.py` (output in `docs/reports/candidate_overlap.txt`). An
evaluation problem counts as a near-duplicate of a pool when at least 50% of its word 13-grams occur
in that pool. "Any 13-gram hit" is reported too but is inflated by boilerplate (for example AIME's
"m and n are relatively prime positive integers. Find m+n"), so it is not used for decisions.

## Decision

**Olympiad benchmark: OlympiadBench, English text-only competition math (`OE_TO_maths_en_COMP`),
674 problems**, `Hothan/OlympiadBench@91184b52`, Apache-2.0.

| Candidate | Problems | Licence | Released | Near-dups in DAPO / DeepScaleR / Skywork |
| --- | --- | --- | --- | --- |
| **OlympiadBench-EN (chosen)** | 674 | Apache-2.0 | 2024-07 | 46 / 6 / 76 |
| OlymMATH-EN (easy + hard) | 200 | MIT | 2025-03 | 0 / 2 / 6 |
| Omni-MATH | 4,428 | Apache-2.0 | 2024-10 | 355 / 3,531 / 4,289 |

Why: it is the most widely reported olympiad set, so numbers are comparable with published work,
and 674 problems give a pass@1 standard error of about ±1.9 points at 50% accuracy. Its pool
overlap is removed from the RL pool (decision 003), not tolerated. OlymMATH has the lowest overlap
but only 200 problems (100 of them likely near 0% for a 4B model). Omni-MATH is almost entirely
contained in two candidate pools and many of its answers are not machine-checkable.

**Level C: two frozen sub-levels, always reported separately.**

| Sub-level | Contents | Problems | Licence | Near-dups in any pool |
| --- | --- | --- | --- | --- |
| **C1 (post-release)** | AIME 2026 I+II without I-P1, HMMT Feb 2026 | 29 + 33 = 62 | CC-BY-NC-SA-4.0 | 0 after the exclusion |
| **C2 (recent, pre-release)** | HMMT Nov 2025, HMMT Feb 2025, BRUMO 2025, SMT 2025, CMIMC 2025 | 30 + 30 + 30 + 53 + 40 = 183 | CC-BY-NC-SA-4.0 | 0 |

Why: C1 is the only data clearly newer than Qwen3.5, but 62 problems give about ±6 points of
standard error on pass@1, too coarse alone. C2 adds 183 problems that no candidate RL pool contains,
but they predate Qwen3.5's release and may be in its pretraining data; keeping the two apart lets
the report say which conclusions hold on strictly unseen problems.

AIME 2026 I Problem 1 ("Patrick started walking at a constant rate...", answer 277) is excluded
from C1: it appears verbatim, with the same answer, in DAPO-Math-17k and Skywork-OR1, both last
modified in 2025, so it was public before the 2026 exam. Sources agreeing it is AIME 2026 I-P1:
MathArena/aime_2026_I, math-ai/aime26, evalscope aime26.

## Consequences
- Problem files live in `$SOE_DATA`; the public repo holds only their sha256 in
  `soe/trainer/config/eval_suite.yaml` (CC-BY-NC-SA share-alike would otherwise apply to the repo).
- Level C is never used for checkpoint selection, hyperparameters or the G1 stopping rule.
- Answer formats differ by set (AIME integers; HMMT/SMT/CMIMC expressions), so the M0.8 verifier
  audit samples every set.

## For the report
We evaluate on four public Level A sets (MATH-500, AIME 2024, AIME 2025, OlympiadBench-EN with 674
text-only problems) and a held-out Level C split into C1, 62 problems from competitions held after
the base model's release (AIME 2026 minus one problem found verbatim in 2025 training corpora, HMMT
February 2026), and C2, 183 problems from 2025 competitions absent from every candidate RL corpus
but possibly seen in pretraining. Results on C1 and C2 are reported separately.

## Alternatives considered
See the tables above; also waiting for later 2026 competitions (rejected: the suite must freeze in
M0, week 1).
