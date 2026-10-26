# sharpen-or-expand

Does reinforcement-learning post-training with verifiable rewards (RLVR) **sharpen** what a
reasoning model already solves (higher pass@1), or **expand** what it can solve at all (higher
pass@k at large k)? An empirical study on Qwen3.5-4B-Base, starting from one frozen SFT checkpoint
and comparing DPO, GRPO, Dr.GRPO, DAPO and RS-GRPO at equal rollout-token budgets.

- Plan (source of truth): [docs/milestone-plan.md](docs/milestone-plan.md)
- Progress: [ROADMAP.md](ROADMAP.md) and [docs/roadmap/](docs/roadmap/)
- Design decisions: [docs/decisions/](docs/decisions/); gate memos: [docs/gates/](docs/gates/)
- Running on TAMU FASTER (offline compute nodes): [docs/cluster.md](docs/cluster.md)

## Layout

The package follows [verl](https://github.com/verl-project/verl)'s conventions, so our extensions
sit where verl's extension points are. Entries marked *(Mx)* arrive in that milestone.

```
soe/
  trainer/                        entry points: python -m soe.trainer.main_*
    main_rl.py                    verl RL runs (GRPO, Dr.GRPO, DAPO, RS-GRPO)       stub until M1
    main_eval.py                  Lane B evaluation of one checkpoint               stub until M0
    main_sft.py  main_dpo.py      TRL SFT / DPO                                      (M2, M4)
    config/
      experiment.py               ExperimentConfig; validators enforce the standing rules
      eval_suite.py               frozen eval suite schema, problem files pinned by sha256
      eval_suite.yaml             the frozen suite                                   (M0)
    rl/                           core_algos.py (RS-GRPO estimator), verl_overrides.py (M1)
  workers/
    reward_manager/math_reward.py the reward function verl calls (same verifier as eval)
    rollout/                      vLLM sampler for evaluation                        (M0)
  utils/
    reward_score/math_verify.py   last \boxed{} extraction + math-verify equivalence
    metric/                       pass_at_k.py (unbiased pass@k, maj@k), transitions.py
    dataset/                      problems.py (Problem, Level A/B/C), records.py (SampleRecord)
    run.py  run_card.py  jsonl.py run directories, run cards, JSONL I/O
  analysis/                       cross-run aggregation (seeds, equal-token comparisons)
  viz/                            one fig_<ID>.py per figure, rebuilt from runs/
examples/                         experiment YAMLs grouped by algorithm; data_preprocess/ (M0)
tests/                            mirrors soe/; *_on_cpu.py run in CI; special_sanity/, special_e2e/
scripts/                          stage_offline.py + stage_assets.yaml, check_container.py
slurm/                            sbatch templates: common.sh, check_container, train, eval
container/soe.def                 the one Apptainer/Singularity image (built locally)
docs/                             plan, roadmap, decisions, gate memos, cluster guide
runs/ figures/ logs/              generated, gitignored
```

Big files (models, datasets, the .sif, checkpoints) live outside git in a sibling folder
`../soe-data/` (`$SOE_DATA`).

## Quick start (local, CPU)

```bash
make install   # uv sync --locked (CPU deps only; the GPU stack is the `train` extra, in the image)
make hooks     # pre-commit: ruff + mypy on every commit
make check     # ruff + ruff format --check + mypy --strict + CPU tests
```
