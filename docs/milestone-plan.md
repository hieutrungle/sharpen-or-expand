# RLVR Capability-Expansion Study — Milestone Plan

Oct 26, 2026 · @Hieu

## Overview

The full study takes about **16 weeks end to end**, because the real budget is wall-clock on a single 2×A100 slot, not GPU-hours. With at most 2×A100 and 2×A40 running at once, every A100 GPU-hour in the original budget costs half an hour of calendar time on the only lane that can train the 4B models.

The A100-bound work (SFT, DPO, screening, full runs, seeds, ablations, Gemma training) sums to roughly 1,900–2,900 A100 GPU-hours. That is 950–1,450 hours of continuous 2×A100 occupancy, or 40–60 days, before queueing and restarts. Add about 25% for those, three weeks of setup before the A100s are fully busy, and a week of write-up, and you land at 15–16 weeks rather than the 8–12 in the original scoping.

The plan therefore runs two lanes in parallel and orders the work so the A100 lane never sits idle. Everything that does not need an A100 (development on 2B, all evaluation, preference generation, small sweeps) moves to the A40 lane. The research content, model choices and protected core experiments are unchanged from the original scoping.

## Hardware lanes

Lane A trains; Lane B evaluates and develops. One wall-clock hour on Lane A equals 2 A100 GPU-hours, so a 120–160 GPU-hour RL run is a 60–80 hour job.

| Lane | Hardware | Runs here | Never runs here |
| --- | --- | --- | --- |
| A | 2×A100 | Qwen3.5-4B SFT, DPO, every RL run (screening, full, seeds, ablations), Gemma 4 E4B SFT and RL | Evaluation sweeps, preference sampling, 2B work |
| B | 2×A40 (48 GB) | Qwen3.5-2B development, RS-GRPO β sweep, ablation pilots on 2B, base/SFT/DPO evaluation, preference generation, pass@k evaluation of every saved checkpoint | Training that feeds final 4B conclusions (unless the M1 exception below applies) |

Evaluation is decoupled from training. RL jobs on Lane A only run a small in-loop pass@1 validation set and write checkpoints at fixed rollout-token intervals; Lane B picks those checkpoints up and runs the full K=1,4,8,16 suite and transition tables. This keeps Lane A training, and it means most of the final high-K evaluation is done by the time the last run finishes.

The one exception: if the M1 memory study forces BF16 LoRA for all algorithms anyway, test whether the identical LoRA config fits on the A40 pair. If it does, Lane B can also run seeds, which saves about three weeks (see Cut order and compression).

Open question: are the A100s 40 GB or 80 GB? This decides how likely full-parameter RL is, and it should be confirmed before M1.

## Schedule at a glance

&#91;embedded content: two-lane schedule · 16 weeks, 4 decision gates\]

Lane A runs back to back from week 2, so any slip there moves the end date; Lane B work can absorb delays. The four gates are where later work can change: G0 fixes the training mode, G1 the full-run budget, G2 the hypothesis, and G3 whether the seeds confirm it.

## Milestones

Ten milestones, M0–M9. Each lists its weeks, lanes, the work, and exit criteria you can tick off; a milestone is done only when every exit box is ticked.

### M0 — Frozen evaluation and verifier (Week 1, Lane B)

The yardstick is fixed before any serious training, so nothing downstream can drift it.

- Level A: MATH-500, AIME 2024, AIME 2025, one olympiad-level benchmark.
- Level B: a held-out split carved from the RL problem pool before any training touches it.
- Level C: a recent, contamination-resistant set, excluded from training and from all hyperparameter selection.
- Hard subset of 200–300 problems for K=32/64, chosen now and frozen.
- Fixed sampling parameters (temperature, top-p, max tokens) and the unbiased pass@k estimator.

Exit criteria:

- [ ] Eval config and problem lists committed with a hash
- [ ] Verifier audited by hand on \~200 items, including equivalent-form edge cases (fractions, radicals, boxed formats)
- [ ] Contamination report: SFT and RL pools checked against all three levels

### M1 — Working pipeline and 4B memory study (Weeks 1–2, Lanes A and B)

Lane B builds and debugs the full pipeline on Qwen3.5-2B; Lane A spends about two days measuring what 4B training costs. Budget: 100–200 GPU-hours, mostly A40. None of these results count toward conclusions.

- Lane B: SFT, rollout generation (vLLM/SGLang), GRPO, checkpoint/resume, evaluation hand-off, logging of every mandatory metric.
- Lane A: Qwen3.5-4B full BF16 vs BF16 LoRA at G=8 — peak memory, KV-cache headroom, rollout tokens/sec, training tokens/sec.

Exit criteria:

- [ ] 2B GRPO run shows training reward rising with sensible evaluation behaviour
- [ ] A deliberately killed job resumes from checkpoint with identical metrics continuity
- [ ] **Gate G0:** full BF16 or BF16 LoRA chosen for all 4B algorithms
- [ ] Measured throughput converts the 30–50M and 100–200M rollout-token budgets into wall-clock hours

### M2 — Frozen SFT checkpoint (Weeks 2–3, Lane A trains, Lane B evaluates)

One SFT checkpoint becomes the shared starting point for every RL run.

- 25K–50K verified reasoning trajectories, decontaminated against M0.
- Base model evaluation on Lane B while SFT data is prepared.
- SFT run #1 for one epoch; run #2 only if the recipe needs adjustment.

Exit criteria:

- [ ] Base and SFT results at K=1,4,8,16 recorded
- [ ] SFT checkpoint frozen, hashed, and referenced by every later config

### M3 — RL screening (Weeks 4–5, Lane A; Lane B in parallel)

GRPO, Dr.GRPO, DAPO and RS-GRPO each run once at 30–50M rollout tokens, roughly 30–50 hours of wall-clock each. The Dr.GRPO screening run doubles as its diagnostic run.

- Lane B meanwhile: preference generation from the SFT policy (15K–30K prompts × 4–8 samples, verifier-labelled pairs), RS-GRPO β ∈ {1, 2, 4} on 2B, evaluation of screening checkpoints.

Exit criteria:

- [ ] All four implementations verified against their published behaviour
- [ ] Pathologies observed and logged (zero-variance groups, length drift, entropy collapse, ratio outliers, truncation)
- [ ] β chosen for 4B from the 2B sweep
- [ ] **Gate G1:** one-page decision memo fixing the full-run rollout budget and the stopping criterion

### M4 — DPO (Week 6, Lane A, 1–2 days)

DPO runs here, not before screening, because its preference data is generated on Lane B during M3 and it is not on the RL critical path.

Exit criteria:

- [ ] DPO trained from the frozen SFT checkpoint and evaluated on the full suite

### M5 — Full seed-1 runs (Weeks 6–8, Lane A)

GRPO, DAPO and RS-GRPO run to the budget fixed at G1, each a 60–80 hour job (chain two jobs if needed). Lane B evaluates every checkpoint and builds capability-transition tables as they arrive.

Exit criteria:

- [ ] Three runs stopped by the predeclared criterion, not by eye
- [ ] pass@1–16, hard-subset pass@32/64 and transition tables for all three
- [ ] **Gate G2:** go/no-go written down — the hypothesis seeds and Gemma will test, stated before running them

### M6 — Seeds 2 and 3 (Weeks 9–11, Lane A)

Six more full runs (GRPO, DAPO, RS-GRPO × 2 seeds), about 15–20 days of Lane A. This is the largest single compute item and the one that makes the headline claim credible.

Exit criteria:

- [ ] Mean and spread over 3 seeds for every headline metric
- [ ] **Gate G3:** does the G2 hypothesis survive the seeds? Record yes, no, or revised

### M7 — Ablations (Weeks 12–13, Lane A)

Leave-one-out ablations and a group-size check, 6–12 days of Lane A. Pilot each ablation on 2B on Lane B during M5–M6 to catch no-op ablations before spending 4B time.

- Full DAPO minus dynamic sampling, minus Clip-Higher, minus token-level loss.
- DAPO minus overlong handling only if truncation showed up in M3/M5.
- GRPO at G = 4, 8, 16 with shorter runs, matched by rollout tokens.

Exit criteria:

- [ ] Each component's effect attributed on pass@1, pass@k, entropy and length
- [ ] Group-size result stated as newly solved problems per million rollout tokens

### M8 — Gemma 4 E4B replication (Weeks 14–15, Lane A)

Repeat the hypothesis, not the project: Gemma SFT with the same data recipe, then only the two methods the G3 finding implicates (for example DAPO and RS-GRPO).

Exit criteria:

- [ ] Gemma SFT frozen and evaluated
- [ ] Two RL runs evaluated; replication or non-replication reported either way

### M9 — Final evaluation and write-up (Week 16, mostly Lane B)

Most evaluation is already done; this week fills gaps, re-checks final checkpoints and writes everything up.

Exit criteria:

- [ ] Full suite at K=1,4,8,16 and hard subset at K=32/64 for every final checkpoint
- [ ] Compute accounting table: rollout tokens, training tokens, GPU-hours, wall-clock per run
- [ ] Newly solved held-out problems per million rollout tokens for each method
- [ ] Report, configs and code public or interview-ready, plus a one-paragraph project story

## Changes from the original ordering

Five changes, all driven by the single A100 lane; the experiments themselves are the same.

1. **DPO moves after screening.** Preference sampling runs on Lane B during M3, so Lane A goes straight from SFT to RL instead of waiting on generation.
2. **Full seed-1 runs come before ablations.** The full-run budget is set by screening curves, not by ablations, and leave-one-out ablations do not change the full DAPO recipe. Moving them later brings the go/no-go (G2) forward by about two weeks.
3. **Seeds come before ablations, and Gemma comes last.** Lane A work is ordered by how protected it is, so a slip late in the project cuts the least important work first.
4. **Evaluation leaves the training jobs.** Checkpoints are evaluated asynchronously on the A40s, which keeps every Lane A hour on training.
5. **Cheap questions go to 2B on Lane B.** The β sweep and ablation pilots run on Qwen3.5-2B in parallel, so 4B time is spent only on runs already known to matter.

## Cut order and compression

The protected core never gets cut: base, SFT, DPO, GRPO/DAPO/RS-GRPO × 3 seeds, one Dr.GRPO diagnostic, 2–3 DAPO ablations, full pass@1–16, hard-subset pass@32/64, transition analysis and compute accounting.

If the schedule slips, cut in this order:

1. Gemma replication (M8), about two weeks of Lane A.
2. The GRPO group-size study.
3. The third DAPO ablation, keeping the two whose 2B pilots showed the largest effect.
4. Full-run length, down to the low end of the 100–200M rollout-token range, decided at G1 rather than mid-run.

Three ways to finish sooner, in order of payoff:

- **Run seeds on Lane B too.** Only possible if G0 picks LoRA and the same config fits on the A40s; seeds stay matched by rollout tokens, and the hardware is logged per run. Saves roughly 2–3 weeks.
- **Automate job chaining.** Each job resubmits itself from its last checkpoint, so queue gaps cost hours rather than overnight idle time.
- **Start SFT data curation in week 1.** If the SFT set is ready when the M1 memory study ends, M2 can start mid-week 2 and shift everything up by several days.

## Standing rules

These hold for every run from M2 onward; breaking one makes the algorithm comparison ambiguous.

- **One starting point.** Every Qwen RL run and DPO starts from the single frozen SFT checkpoint.
- **One training mode.** Whatever G0 picks (full BF16 or BF16 LoRA) applies to all algorithms; QLoRA is for development only.
- **Matched by rollout tokens.** Runs are compared at equal generated-token budgets, not equal steps.
- **Predeclared stopping.** A run stops on the G1 criterion (for example, no meaningful validation gain across N evaluations), never because a curve looks good.
- **Jobs sized for 48–72 hours.** The 96-hour limit is a ceiling, not a target; checkpoint every 2–4 hours and resume automatically.
- **Level C stays untouched.** It is never used for hyperparameter or checkpoint selection.
- **Everything logged.** All capability, transition, training, behaviour and efficiency metrics from the original scoping, plus commit hash, config, seed and hardware per run.
- **Development results stay out.** Nothing from M1 on 2B appears in the conclusions.
