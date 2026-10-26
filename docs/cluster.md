# Running on TAMU FASTER

Compute nodes have **no internet**; home is capped at 10 GB / 10k files. Everything heavy lives on
scratch in `$SOE_DATA=/scratch/user/hieult/research/soe-data`, and code moves only through GitHub.

## Lanes
| Lane | GPUs | Use | Template |
| --- | --- | --- | --- |
| A | 2×A100 (PCIe 40 GB) | every 4B / Gemma training run | `slurm/train.sbatch` (default `gpu:a100:2`) |
| B | 2×A40 (48 GB) | 2B development, evaluation, preferences, sweeps | `slurm/eval.sbatch` (`gpu:a40:1`); dev training with `SBATCH_GRES=gpu:a40:2` |

## One-time setup
1. Locally: `make container` (needs sudo + apptainer) builds `../soe-data/sif/soe.sif`.
2. From your terminal: `rsync -av ../soe-data/sif/soe.sif faster-claude:/scratch/user/hieult/research/soe-data/sif/`.
3. On the FASTER login node (downloads are light work):
   `export SOE_DATA=/scratch/user/hieult/research/soe-data && cd /scratch/user/hieult/research/sharpen-or-expand && HF_HOME=$SOE_DATA/hf uv run scripts/stage_offline.py --assets scripts/stage_assets.yaml`
4. `mkdir -p logs/slurm && sbatch slurm/check_container.sbatch`, then again with
   `SBATCH_GRES=gpu:a40:1`.

## Everyday flow
1. Edit locally on a work branch `m<k>/<topic>`; `make check`; commit; push.
2. On FASTER: `git fetch && git switch m<k>/<topic>` (or `git pull --ff-only`). Never edit there; a
   fix made on FASTER goes on `faster/<topic>` and is merged locally.
3. Submit from the repo root:
   - new run, chained up to 2 more times: `CHAIN_LEFT=2 sbatch slurm/train.sbatch examples/dapo_trainer/m3_screen_qwen35-4b.yaml 1`
   - resume: `RUN_DIR=runs/<run_id> sbatch slurm/train.sbatch examples/dapo_trainer/m3_screen_qwen35-4b.yaml`
   - evaluate a checkpoint: `sbatch slurm/eval.sbatch soe/trainer/config/eval_suite.yaml <checkpoint> <out dir>`
4. Poll with `squeue -u hieult` and `sacct -j <id> --format=JobID,State,Elapsed,ExitCode`; read
   `logs/slurm/<name>-<jobid>.out`.
5. Bring results home with rsync from your own terminal, excluding checkpoints:
   `rsync -av --exclude 'checkpoints/' faster-claude:/scratch/user/hieult/research/sharpen-or-expand/runs/ runs/`

## Rules the templates enforce
- Assets are staged as complete Hub snapshots (no file filters): offline, vLLM and
  huggingface_hub refuse a snapshot with any repo file missing.
- Every `sbatch` has `--nodes=1 --ntasks=1` (FASTER's submit filter rejects jobs without them).
- Jobs refuse to start from a dirty git tree; chained jobs refuse to start if HEAD moved.
- `HF_HUB_OFFLINE=1`, `WANDB_MODE=offline`; Ray's temp dir is short and node-local.
- Override GPU type and time limit with `SBATCH_GRES` / `SBATCH_TIMELIMIT` environment
  variables, not `--gres` / `--time`: chained successors inherit the environment, not the
  command line.

## Status
Untested on FASTER (the cluster is down as of 2026-10-26): the image has not been built, and no
job from these templates has run. M0.1–M0.2 exercise them.
