# 004 — Evaluation protocol: sampling, prompt, scoring and transitions

- Date: 2026-10-26
- Status: accepted (user, 2026-10-26: "do your recommendations", reviewed and committed; literature-standard sampling and
  verifier-equivalence maj@k were the user's choices)
- Milestone: M0 (task M0.5)
- Frozen in: `soe/trainer/config/eval_suite.yaml` (`soe-eval-v1`)

## Context
Every checkpoint (base, SFT, DPO, every RL run, both models) must be evaluated identically, or
differences between algorithms could come from the yardstick. The plan fixes K = 1, 4, 8, 16 on
everything and K = 32, 64 on a 200–300 problem hard subset; the rest was open. Using the settings of
the study this project extends keeps the methods section short and the results comparable.

## Decision

**Sampling: as in Yue et al. (2025)**, "Does Reinforcement Learning Really Incentivize Reasoning
Capacity in LLMs Beyond the Base Model?" (arXiv 2504.13837), which samples base and RLVR models
identically at temperature 0.6, top-p 0.95, with up to 16,384 generated tokens.

| Setting | Value | Note |
| --- | --- | --- |
| temperature / top-p / top-k | 0.6 / 0.95 / off | Yue et al. |
| max response tokens | 16,384 | Yue et al.; also the cap on RL rollout length (checked from M1) |
| seed | 0 for every request | same random numbers for every checkpoint: lower-variance comparisons, reproducible runs |
| samples per problem | 16 (main); 64 (hard subset, first 16 shared) | plan's K values |

**Prompt: zero-shot, identical for every model stage**, as in Yue et al. (no few-shot examples for
base models): the problem, a blank line, then "Please reason step by step, and put your final answer
within \boxed{}.", wrapped in the model's chat template. Qwen3.5-2B/4B-Base ship a chat template;
`google/gemma-4-E4B` does not, so M8 uses the template of `google/gemma-4-E4B-it` (same tokenizer).
Consequence: SFT data (M2) is formatted the same way.

**Scoring.**
- The answer is the content of the last `\boxed{}`; no box means incorrect.
- A response cut off at 16,384 tokens counts as incorrect; the truncation rate is reported per set.
- maj@k groups answers by verifier equivalence (math-verify): each answer joins the first group
  whose first answer it matches; ties go to the group seen first; samples without an answer do not
  vote. "0.5", "\frac12" and "1/2" are one vote.

**Capability transitions.** A problem is solved at k when its unbiased pass@k estimate is at least
0.5. Transitions (unsolved→solved, solved→lost, kept, never) are counted at two k values per tier:
- k = 1: solved in at least half of the samples, i.e. solved reliably (the sharpening view);
- k = 16 (main) and k = 64 (hard subset): solved at least once in all samples (the expansion view).

## Consequences
- One evaluated checkpoint is 51,568 samples (2,473 problems × 16, plus 250 hard-subset problems
  × 48 more): about 103M / 206M / 413M generated tokens at a mean response of 2K / 4K / 8K tokens.
  M1 measures A40 throughput; if Lane B cannot keep up, the knobs are which checkpoints get the full
  suite (decided at G1) or samples per set, never the length cap (truncation hides exactly the
  long reasoning that matters at large k).
- No RL run may use a rollout length above 16,384 tokens.
- maj@k with verifier grouping is not transitive in edge cases (A~B, B~C, A≁C); grouping against
  the group's first answer makes the result deterministic. The M0.8 audit covers equivalence.

## For the report
All models are evaluated with the protocol of Yue et al. (2025): temperature 0.6, top-p 0.95, up to
16,384 generated tokens, and the same zero-shot prompt in the model's chat template for base, SFT
and RL checkpoints. We draw 16 samples per problem (64 on a frozen hard subset) and report pass@k
with the unbiased estimator, and maj@k with answers grouped by symbolic equivalence. Truncated
responses count as incorrect. A problem counts as solved at k when its pass@k estimate is at least
0.5; we report capability transitions at k = 1 (reliably solved) and at the largest k (solved at
least once).

## Alternatives considered
- Temperature 1.0 / top-p 1.0 (the RL sampling distribution): closer to what RL optimises, but not
  the literature standard, and every comparison with published pass@k curves would need a caveat.
- 8,192 tokens: half the evaluation cost, but truncates long reasoning chains unevenly across
  checkpoints.
- maj@k by exact string: splits votes between equivalent forms (user preferred equivalence).
