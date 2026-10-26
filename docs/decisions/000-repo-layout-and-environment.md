# 000 — Repo layout, environment and cluster workflow

- Date: 2026-10-26
- Status: proposed
- Milestone: M0

## Context
Training runs on TAMU FASTER, whose compute nodes have no internet and whose home directory is
capped at 10k files. Code must be auditable per run (commit, config, seed, hardware), and the repo
must stay small enough for a newcomer to read in an afternoon.

## Decision
- Repo `sharpen-or-expand/`, package `soe/`, laid out like verl (user decision, 2026-10-26):
  `trainer/` holds the `main_*.py` entry points and `config/` schemas, `workers/` the pieces verl
  calls into (reward function, eval sampler), `utils/` shared building blocks (`reward_score/`,
  `metric/`, `dataset/`, run directories). `analysis/` and `viz/` are research layers verl has no
  equivalent for. This replaces the earlier topic split (data, verifier, rollout, train, eval).
- Experiments are YAML files in `examples/<algorithm>_trainer/`, validated by
  `soe.trainer.config.experiment.ExperimentConfig`; the validators encode the standing rules
  (frozen SFT start, QLoRA only for development, DAPO/RS-GRPO-only switches, equal-token budgets).
- Tests mirror the package; `*_on_cpu.py` files run in CI (`cpu_unit_tests.yml`), GPU tests run
  on FASTER. `pre-commit.yml` runs ruff and mypy through the repo's own pinned tools.
- Python 3.12 (the verl wheelhouse ships cp312 wheels), uv + `uv.lock`. Base dependencies are
  CPU-only so CI and laptops never install the GPU stack; the GPU stack is the `train` extra and
  lives only in the container.
- One image, `container/soe.def`, built locally and rsynced to `$SOE_DATA/sif/soe.sif`. The
  package is bind-mounted at run time (`PYTHONPATH=$PWD`), so code changes need no rebuild.
- Big files live in a sibling folder `soe-data/` (`$SOE_DATA`): `hf/` (HF cache), `datasets/`,
  `sif/`, `checkpoints/`, `manifest.json`. On FASTER: `/scratch/user/hieult/research/soe-data`.
- Run directories `runs/<UTC yyyymmdd-HHMMSS>-<config name>-s<seed>/` hold a frozen config copy,
  `run.json` (provenance) and `efficiency.json` (token and time accounting).
- Slurm templates refuse to run from a dirty tree, and chained jobs refuse to run if HEAD moved
  since the run started. Code moves only through GitHub; FASTER only pulls.
- CLIs use `pydantic_settings.CliApp` with kebab-case flags.

## Consequences
- Every number can be traced to a commit, a config hash and a GPU model.
- Changing the GPU stack means rebuilding the image; that happens only between milestones.

## Alternatives considered
- A venv on scratch instead of an image: tens of thousands of files per environment against a
  2M-file quota, and drift between nodes.
