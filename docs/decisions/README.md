# Decision notes

One file per design choice: `NNN-short-title.md` from [TEMPLATE.md](TEMPLATE.md). A note is
never edited after it is accepted; a later note supersedes it and says so.

| # | Title | Status |
| --- | --- | --- |
| [000](000-repo-layout-and-environment.md) | Repo layout, environment and cluster workflow | proposed |
| [001](001-gpu-stack-verl.md) | GPU stack: verl 0.9.1 for RL, TRL 0.27 for SFT/DPO | proposed |
| [002](002-evaluation-sets.md) | Evaluation sets: OlympiadBench-EN; Level C = C1 (2026) + C2 (2025) | accepted |
| [003](003-rl-pool-and-level-b.md) | RL pool DAPO-Math-17k (EN, dedup); Level B 994 held out | accepted |
| [004](004-evaluation-protocol.md) | Evaluation protocol: Yue et al. sampling, zero-shot prompt, scoring, transitions | accepted |
| [005](005-hard-subset.md) | Hard subset: 4B-Base c ≤ 4/16 on Levels A+B, independent seed 1, 250 | accepted (pre-registered); item 6 superseded by 006 |
| [006](006-thinking-mode-and-truncation.md) | Thinking mode on (amends 004); 16K kept; selection halts on format only | accepted |
| [007](007-data-v2-text-repair.md) | Data v2: DAPO text debris repaired in place, splits frozen; suite soe-eval-v2 | proposed |
