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

Preparation (`examples/data_preprocess/m0_datasets.py` + `m0_datasets.yaml`; logic in
`soe/utils/dataset/{decontaminate,splits}.py`; built 2026-10-26, report
`docs/reports/m0_datasets_v1.json`):

| Step | Rule | Problems |
| --- | --- | --- |
| Raw pool | DAPO-Math-17k, English, de-duplicated copy | 14,116 |
| 1 | drop exact duplicates (normalised case, punctuation, whitespace) | −679 |
| 2 | drop near-duplicates of any Level A/C problem (50 OlympiadBench-EN, 20 MATH-500) | −70 |
| 3 | sample **Level B: 1,000** uniformly at random, seed 20261026 | 1,000 |
| 4 | drop train problems that near-duplicate a Level B problem | −131 |
| 5 | remove Level B problems the train pool still covers as a whole (reworded copies) | −6 from B |
| Result | **RL train 12,236 · Level B 994**; final audit: 0 evaluation or Level B problems covered by train | |

Near-duplicate = at least 50% of the reference problem's word 13-grams occur in the candidate.
**Refinement (2026-10-26, during the build):** 13-grams shared by more than 20 pool problems are
ignored (73 n-grams). They are answer-format boilerplate, e.g. DAPO's "where gcd(m, n) = 1. Please
provide the value of m + n" (in 639 problems) and "m and n are relatively prime positive
integers". Without this, short problems matched any problem with the same boilerplate: step 4
removed 822 problems hitting only 124 Level B problems, and just 16% shared the gold answer; with
it, 131 are removed and 91% share the gold answer. Step 2 is unchanged by the refinement (70 either
way) because the evaluation sets carry no DAPO boilerplate. Of the 70, 26 share the gold answer;
the rest are templated variants of an evaluation problem, removed deliberately. A threshold of 0.3
was tried and rejected: more false positives (76% same-answer) and still 4–6 reworded leaks, which
step 5 handles directly. Level B is therefore 994 rather than 1,000 (pass@1 standard error ±1.6
points at 50% accuracy either way).

## Consequences
- Integer-only answers differ from MATH-500's expression answers; training reward and Level A
  accuracy measure slightly different skills, which the report states.
- DAPO's prompts carry a fixed instruction wrapper; preprocessing stores the bare problem and the
  prompt template lives in the experiment config, identical for every algorithm.
- If M1 shows many always-solved or never-solved problems for the 4B SFT model, difficulty
  filtering is a new decision note, applied to all algorithms alike.

## For the report
RL uses DAPO-Math-17k (English, de-duplicated: 14,116 problems, all with integer answers, which keeps
the verifiable reward close to noise-free). After removing 679 exact duplicates, we drop the 70
problems that share at least half of their word 13-grams with an evaluation problem (ignoring
answer-format boilerplate shared by more than 20 problems), hold out 1,000 problems at random as the
in-distribution Level B split, and remove from training the 131 problems that near-duplicate a
Level B problem. Six Level B problems whose reworded copies remain spread across the training pool
are dropped from Level B, leaving 994 Level B and 12,236 training problems; a final audit finds no
evaluation or Level B problem covered by the training pool.

## Alternatives considered
DeepScaleR-40K and Skywork-OR1-RL (math) — see the table and reasons above.
