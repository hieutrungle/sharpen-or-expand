# 003 — RL problem pool and the Level B held-out split

- Date: 2026-10-26
- Status: accepted (user, 2026-10-26: "use your recommendation")
- Milestone: M0 (task M0.4)

## Context
The plan asks for 10K–20K RL math problems, and a Level B held-out split carved from that pool
before any training touches it. Reward noise from the verifier would confound the sharpen/expand
comparison, so answer checkability matters as much as size. Evidence:
`docs/reports/candidate_overlap.txt` (method in decision 002).

## Decision
**Pool: DAPO-Math-17k**, the de-duplicated English copy `open-r1/DAPO-Math-17k-Processed@31dd3095`
(config `en`), derived from `BytedTsinghua-SIA/DAPO-Math-17k@65877096` (Apache-2.0; the processed
copy states no licence of its own).

| Candidate | Unique problems | Answer types | Licence | Near-dups of MATH-500 / OlympiadBench-EN / C1+C2 |
| --- | --- | --- | --- | --- |
| **DAPO-Math-17k, EN (chosen)** | 14,116 | 100% integer | Apache-2.0 (original) | 18 / 46 / 1 |
| DeepScaleR-40K | 40,315 | 58% integer, 18% rational, 24% expression | MIT | 18 / 6 / 0 |
| Skywork-OR1 (math) | 105,055 | mixed | not stated | 22 / 76 / 1 |

Why:
1. Every answer is an integer, so the reward is close to noise-free: a false positive or negative
   from the verifier would otherwise be mistaken for an algorithm effect.
2. It is the data DAPO was designed and tuned on, which is fair to the DAPO arm and its ablations.
3. Its size fits the plan's 10K–20K range after cleaning without subsampling choices.
4. Its licence is clear. DeepScaleR is larger but mixes answer types and contains AIME 1983–2023
   and Omni-MATH; Skywork-OR1 has useful per-problem difficulty labels but no stated licence.

Preparation (`examples/data_preprocess/dapo_math.py`, M0):
1. Drop exact duplicates after normalising case, punctuation and whitespace (679 found).
2. Drop every near-duplicate (>= 50% 13-gram overlap) of any Level A, B-candidate or C problem.
3. Carve **Level B: 1,000 problems**, uniformly at random with a fixed seed, before any training.
   1,000 gives a pass@1 standard error of about ±1.6 points at 50% accuracy.
4. The rest (about 12.4K) is the RL training pool. Exact counts are written by the script and
   recorded in the M0 contamination report.

## Consequences
- Integer-only answers differ from MATH-500's expression answers; training reward and Level A
  accuracy measure slightly different skills, which the report states.
- DAPO's prompts carry a fixed instruction wrapper; preprocessing stores the bare problem and the
  prompt template lives in the experiment config, identical for every algorithm.
- If M1 shows many always-solved or never-solved problems for the 4B SFT model, difficulty
  filtering is a new decision note, applied to all algorithms alike.

## For the report
RL uses DAPO-Math-17k (English, de-duplicated: 14,116 problems, all with integer answers, which keeps
the verifiable reward close to noise-free). We remove every problem whose word 13-grams overlap at
least 50% with any evaluation problem, then hold out 1,000 problems at random as the in-distribution
Level B split before training; the remainder is the training pool shared by every algorithm.

## Alternatives considered
DeepScaleR-40K and Skywork-OR1-RL (math) — see the table and reasons above.
