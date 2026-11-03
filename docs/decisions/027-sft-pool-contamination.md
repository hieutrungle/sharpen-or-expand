# 027 — SFT problem pool: every problem a contamination check lists is dropped, in rounds

- Date: 2026-11-03
- Status: accepted (user, 2026-11-03: "accept the decision"; "delete 92 keeps"). The reading was
  done by the agent; the user chose each rule
- Milestone: M2 (task M2.3; roadmap D2, exit criterion 1)
- Code: `scripts/contamination_report.py` (`--references rl`, `--levels`, `--min-share`),
  `examples/data_preprocess/m2_sft_pool_clean.py` + `.yaml`
- Reports (ids and numbers only): per round, `docs/reports/m2_contamination_sft{,_levelc,_rl}.json`
  (round 1) and `…_r2.json` (round 2); the final check `…_clean.json`; the cleaner's
  `m2_sft_pool_clean.json`; the reading record `m2_sft_pool_review.json`

## Context

The SFT pool (`datasets/soe/v1/sft/pool.jsonl`) holds 81,260 DeepMath-103K problems (sha256
`e255c1f3…`, M2.2). It must not contain any evaluation problem (Levels A, B, C). It must also be
disjoint from RL train v3.1 and its validation split (D2), so that RL's newly solved problems were
never trained in SFT.

The check is decision 011's method:
- word 13-gram and 8-gram shares, with boilerplate removed (n-grams in more than 20 pool
  problems);
- every pair at ≥ 20% of a reference's n-grams is listed;
- each pair is classed E (identical after formatting), R (same numbers, reworded) or V (different
  numbers).

Round 1 lists 670 suite pairs (279 E, 37 R, 354 V), almost all with MATH-500 and Level B. It also
lists 5,625 pairs with the RL pool, although DeepMath's authors say most of its problems are absent
from DAPO-17K.

DeepMath was released on 2025-04-14 and last modified on 2025-05-29, so it could contain the Level
C2 contests held before then:

| Level C set | Held | Relative to DeepMath |
|---|---|---|
| HMMT Feb 2025 (C2) | February 2025 | before: could be in it |
| BRUMO 2025 (C2) | 4–5 April 2025 | before the release: could be in it |
| CMIMC 2025, SMT 2025 (C2) | spring 2025 (MathArena gives CMIMC as both March and May) | around the release: could be in it |
| HMMT Nov 2025 (C2) | November 2025 | after |
| AIME 2026, HMMT Feb 2026 (C1) | February 2026 | after (problems can still be reused) |

The roadmap therefore asks for Level C to be checked below 20% as well.

## Decision

1. **Every pool problem in any listed pair is dropped.** Each round has three reports, all made on
   the pool left after the previous round:
   - Levels A, B, C at 20%;
   - Levels C1 and C2 at 10% (`--levels C1 C2 --min-share 0.1`);
   - RL train v3.1 + validation (`--references rl`).

   A problem listed in any of them is dropped. Reading does not decide.
2. **Rounds until a check lists nothing.** Removing problems shrinks the boilerplate set, so
   n-grams that were filtered come back and a re-check can list new pairs. Round 2 found 53 more
   RL-pool problems this way, including real copies: an infinite series answered as m + n, an
   orthocenter problem, a square probability, an incircle problem.

   | Round | Pool in | Levels A/B/C (20%) | Level C (10%) | RL pool | Dropped (union) | Pool out |
   |---|---|---|---|---|---|---|
   | 1 | 81,260 | 640 | 59 | 4,170 | 4,641 | 76,619 |
   | 2 | 76,619 | 6 | 0 | 53 | 54 | 76,565 |
   | check | 76,565 | 0 pairs | 0 pairs | 0 pairs | — | 76,565 |

   The final check also leaves nothing at the build's frozen rule (13-grams ≥ 0.5, single
   problem or union).
3. **Why reading does not decide (the history).** The agent first read the pairs side by side
   under decision 011's rule: drop the same problem, keep a genuine variant. The record is in
   `m2_sft_pool_review.json`:
   - 435 pairs read: 343 judged the same problem, 92 variants;
   - 260 lower-share V pairs left unread.

   Three successive steps took the rule to "drop everything listed":
   - The V class proved unreliable. Copies hide behind Asymptote code, multiple-choice options,
     or DAPO's "give m + n" re-encoding, and 12 of 49 high-share V pairs read were copies. The
     user dropped the 260 unread pairs.
   - The user then dropped the 15 variants at the frozen rule.
   - A careful re-read of the 92 variants on 91 pool problems found no misread copy. About half
     are template variants of an evaluation problem: one number changed, or another question on
     the same stem or diagram. A trace for one of them teaches the exact solution path of its
     reference. Examples:
     - F₁₀₀ mod 8 next to MATH-500's F₁₀₀ mod 4 (both answers 3);
     - ⌊Σ 1/aₙ⌋ up to 2000 next to Level B's Σ 1/aₙ up to 1980 (both 88);
     - x⁸ + 98x⁴ + 1 next to x⁸ − 98x⁴ + 1.

     The rest share only answer-format wording. The user dropped all 92 (user, 2026-11-03). The cost is 91 problems,
     about 0.1% of the pool.

   Under the new rule the 260 unread pairs and the 92 variants are all dropped. So are the 7 new
   round-2 suite pairs; all of them are earlier-read keeps that share the "m/n relatively prime,
   find m + n" boilerplate. Finally, 27 of the 60 round-1 Level C pairs at 10% were not read.
4. **This applies to the SFT pool only.** RL train v3.1 keeps decision 011's variants: it was
   frozen in M1 and every RL algorithm sees it identically. The SFT pool can afford the stricter
   rule. It has 76,565 problems for 25K trajectories (D3), and every variant it drops is one
   fewer base-to-SFT transition that SFT could have taught directly.

**Result.** `datasets/soe/v1/sft/pool_clean.jsonl`: **76,565 problems**, sha256
`87d55db66a7900aeabdeef3325e13e24f1bc54baca5456a55a1ca991a7949273`. 4,695 problems were dropped:
- 672 matched an evaluation problem, by pairs: 190 to Level A, 454 to Level B, 22 to Level C1,
  38 to Level C2;
- 4,023 matched only the RL pool; another 200 matched both an evaluation problem and the RL pool.

Of the 60 Level C pairs, 33 were read: none is a copy, and the two with the same answer are
coincidences. The other 27 are V pairs at 10–20% with a different answer. Reading found no wrong
Level B gold.

## Consequences (including what could confound the algorithm comparison)

- **What the check cannot catch.** A copy sharing less than 20% of its 8-grams and 13-grams with
  the reference (10% for Level C): a full paraphrase or a translation. The same holds for the RL
  pool (decision 011).
- **Disjoint pools.** The SFT and RL pools are disjoint at this rule. No RL training or validation
  problem is an SFT problem.
- **Asymmetry with the RL pool.** The SFT pool holds no listed variant of an evaluation problem; the
  RL pool v3.1 still holds decision 011's reviewed variants. Every algorithm trains on the same
  RL pool from the same SFT checkpoint, so the comparison is unaffected. A per-problem analysis of
  RL gains on those references should still read them with this in mind; their ids are in
  `m0_contamination_rl_v3.json`.
- **Rebuild check.** The clean pool is rebuilt on FASTER from the committed code, and its sha256
  must equal `87d55db6…` before M2.4 samples from it.
- **M2.5 re-check.** The M2.5 train/val files are re-checked with the same three reports.
  Expected: no pairs.

## For the report

The SFT problem pool comes from DeepMath-103K: 81,260 problems after format filters. The check
pairs each pool problem with an evaluation or RL problem when they share at least 20% of the
reference's word 8-grams or 13-grams (10% for the 2025–2026 contests). We removed every problem
it listed against the evaluation problems or the RL training and validation pool. We repeated the
check on what remained until it listed nothing. In total we removed 4,695 problems, leaving
76,565:
- 672 matched evaluation problems (MATH-500, AIME 2024/2025, OlympiadBench, the in-distribution
  held-out split and, at the lower threshold, 2025–2026 contests);
- 4,023 matched only the RL pool.

Reading the pairs showed that n-gram classes miss copies: they hide behind figure code,
multiple-choice options and re-encoded answers. Close variants were removed as well. None of the
2025–2026 contest pairs we read was a copy.

## Alternatives considered

- **Drop only E pairs (279).** Rejected: R and V pairs include real copies.
- **Drop the pairs reading judged the same problem, keep variants (decision 011's rule).** This
  was the first version. The user rejected it for the SFT pool: about half the 92 variants hand
  the evaluation problem's method to SFT, and the pool can afford to lose them.
- **One round only.** Rejected: round 2 found 53 more RL-pool problems, including real copies.
