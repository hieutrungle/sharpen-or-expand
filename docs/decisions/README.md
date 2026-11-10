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
| [014](014-rl-rollout-sampling-and-truncation.md) | RL rollouts (and DPO preference data) at 1.0 / 1.0, validation at evaluation sampling, no fixed seed (via 017); truncated rollout reward 0 | accepted |
| [015](015-vllm-fp8-wakeup-backport.md) | Backport vLLM PR #41602 (FP8 KV wake-up on hybrid models) to the pinned vLLM 0.24.0 | accepted |
| [016](016-rl-rollout-failures-stop-the-run.md) | A rollout failure stops the RL run; RL rewards scored in a process pool (math-verify time limits) | accepted |
| [017](017-rl-rollout-sampling-1.0.md) | RL rollouts and DPO preference data sample at 1.0 / 1.0 for every algorithm; in-loop validation stays at evaluation sampling | accepted |
| [018](018-rl-checkpoint-export-and-token-count.md) | RL checkpoints exported in the training job; soe merges LoRA in FP32; mtp copied from base; rollout tokens = generated tokens of training rollouts | accepted |
| [019](019-freeze-vision-tower.md) | Freeze the vision tower in verl's actor (verl 0.9.1 ignores freeze_vision_tower) | accepted |
| [020](020-rl-token-budget-stop.md) | RL runs stop at their rollout-token budget (or a dev wall-clock / step cap); the last step is re-estimated at each save | accepted |
| [021](021-rl-step-metrics.md) | RL step metrics: soe metrics from per-rollout records at each save; importance ratios from verl's off-policy diagnostics | accepted |
| [022](022-rl-updates-per-step.md) | RL steps make 4 optimizer updates on mini-batches of whole groups, the same for every algorithm | accepted |
| [023](023-sft-wiring.md) | SFT wiring: TRL on pre-tokenized prompt/completion examples (`<|im_end|>` appended, over-long dropped), BF16 LoRA merged into the base layout | accepted |
| [024](024-memory-study-method.md) | M1.10 memory study: descending knob search, per-phase memory marks and GPU/RAM sidecar, fused log-prob as a hardware knob; 4,096 MiB weight-sync buffer | accepted |
| [025](025-g0-shared-4b-config.md) | G0: full BF16 for every 4B run (SFT, DPO, RL) on 4×A40; one shared 4B base config; image and libraries frozen | accepted |
| [026](026-suite-v4.md) | Suite soe-eval-v4: one impossible Level B gold excluded; v3 samples re-scored on CPU, never compared across suites | accepted |
| [027](027-sft-pool-contamination.md) | SFT pool: every problem a contamination check lists (Levels A/B/C at 20%, Level C at 10%, RL pool) is dropped, in rounds until a check lists nothing; 76,565 problems | accepted |
| [028](028-full-bf16-sft.md) | Full-BF16 SFT: FSDP2 through the Trainer (FP32 load, FP32 reduce), sharded checkpoints, resume with an example-id log, full export; supersedes 023 items 3 and 5 | accepted |
| [029](029-sft-recipe-and-run2-rule.md) | Pre-registered SFT run #1 (D5 recipe, final export frozen, validation at every save) and the run-#2 rule: four criteria on the RL validation split and training logs, every triggered change applied, at most one run #2 | accepted |
| [030](030-frozen-sft-export.md) | SFT checkpoint of record = run #2's final export (meets criteria 3 and 4 but beats run #1 on every measure); no third run, SFT's truncation rise reported, not repaired; whether RL starts from it is reopened | accepted |
| [031](031-rl-and-dpo-start-from-the-base.md) | Every RL and DPO run starts from Qwen3.5-4B-Base (pinned snapshot, digest checked at launch); SFT run #2 is an evaluated arm; prompt-identity guard; pass@1 changes split into stopping and accuracy | accepted |
