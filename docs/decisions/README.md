# Decision notes

One file per design choice: `NNN-short-title.md` from [TEMPLATE.md](TEMPLATE.md). A note is
never edited after it is accepted; a later note supersedes it and says so.

| # | Title | Status |
| --- | --- | --- |
| [000](000-repo-layout-and-environment.md) | Repo layout, environment and cluster workflow | proposed |
| [001](001-gpu-stack-verl.md) | GPU stack: verl 0.9.1 for RL, TRL 0.27 for SFT/DPO | proposed |
| [002](002-evaluation-sets.md) | Evaluation sets: OlympiadBench-EN; Level C = C1 (2026) + C2 (2025) | accepted |
| [003](003-rl-pool-and-level-b.md) | RL pool DAPO-Math-17k (EN, dedup); Level B 994 held out | accepted |
| [004](004-evaluation-protocol.md) | Evaluation protocol: Yue et al. sampling, zero-shot prompt, scoring, transitions | accepted; KV-cache dtype amended by 009, extraction by 010 |
| [005](005-hard-subset.md) | Hard subset: 4B-Base c ≤ 4/16 on Levels A+B, independent seed 1, 250 | accepted (pre-registered); item 6 superseded by 006; hard tier amended by 008 |
| [006](006-thinking-mode-and-truncation.md) | Thinking mode on (amends 004); 16K kept; selection halts on format only | accepted |
| [007](007-data-v2-text-repair.md) | Data v2: DAPO text debris repaired in place, splits frozen; suite soe-eval-v2 | accepted |
| [008](008-verifier-notations-and-hard-tier.md) | Verifier accepts audited notations; report re-scores; measurement errors leave the hard tier | accepted |
| [009](009-fp8-kv-cache.md) | FP8 KV cache for all sampling and RL rollouts; suite soe-eval-v3 | accepted |
| [010](010-split-answers.md) | Multi-part answers boxed separately: last 2-3 final-answer boxes joined | accepted |
| [011](011-rl-train-v3.md) | RL train v3: hand-reviewed copies of evaluation problems removed (12,124) | accepted; 256 problems moved to validation by 012 |
| [012](012-rl-validation-split.md) | In-loop validation split (256) carved from RL train v3; RL data v3.1 (11,868 train) | accepted |
| [013](013-rl-no-kl-step-checkpoints.md) | RL runs: no KL term; a checkpoint every N steps with its rollout-token count | accepted |
| [014](014-rl-rollout-sampling-and-truncation.md) | RL rollouts sample like evaluation (0.6 / 0.95, no fixed seed); truncated rollout reward 0 | accepted; sampling superseded by 017 |
| [015](015-vllm-fp8-wakeup-backport.md) | Backport vLLM PR #41602 (FP8 KV wake-up on hybrid models) to the pinned vLLM 0.24.0 | proposed |
| [016](016-rl-rollout-failures-stop-the-run.md) | A rollout failure stops the RL run; RL rewards scored in a process pool (math-verify time limits) | proposed |
| [017](017-rl-rollout-sampling-1.0.md) | RL rollouts and DPO preference data sample at 1.0 / 1.0 for every algorithm; in-loop validation stays at evaluation sampling | accepted |
