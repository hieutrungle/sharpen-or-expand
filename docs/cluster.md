# Running on TAMU FASTER

Compute nodes have **no internet**; home is capped at 10 GB / 10k files. Everything heavy lives on
scratch in `$SOE_DATA=/scratch/user/hieult/research/soe-data`, and code moves only through GitHub.

## Lanes
| Lane | GPUs | Use | Template |
| --- | --- | --- | --- |
| A | 2×A100 (PCIe 40 GB) | every 4B / Gemma training run | `slurm/train.sbatch` (default `gpu:a100:2`) |
| B | 2×A40 (48 GB) | 2B development, evaluation, preferences, sweeps | `slurm/eval.sbatch` (`gpu:a40:1`); dev training with `SBATCH_GRES=gpu:a40:2` |

Sampling-only jobs (evaluation, selection, probes) may also use idle A100, A30 (24 GB, sm_80) and
A10 (24 GB, sm_86) GPUs; `check_container` passes on all four types, while T4s (no BF16) are out.
Slurm's `normal` QOS allows 16 GPUs per user; in practice we run at most six one-GPU jobs at once.
A100/A40/A30/A10 all cost 128 SUs per GPU-hour. Sampling parameters never depend on the GPU type;
each sampling job records the GPU it ran on.

Sampling throughput is bound by KV-cache memory, not by vLLM's request limit: with 16K-token
reasoning, vLLM starts about 180 sequences, the cache fills as they grow, and steady state holds about
50 on an A10 (about 900 tokens/s; about 1,300 on an A40). `max_num_seqs` therefore does not help.
Keep `gpu_memory_utilization` at 0.90: at 0.95 the Qwen3.5 linear-attention prefill kernel runs out
of memory on 24 GB cards (A30 probe, job 3221501). Since soe-eval-v3 the KV cache is FP8 (decision
009): 1.75× the cache capacity on an A40 (1.69M tokens), about 1.25× the throughput (about 1,650
tokens/s), FlashInfer attention on every GPU type. The full 4B-Base suite then costs about 0.94
worker-hours per shard of 96 (A40/A30/A10 mix; 14 workers finished in about 6.5 hours).

## One-time setup
1. Locally: `make container` (needs sudo + apptainer) builds `../soe-data/sif/soe.sif`.
2. From your terminal: `rsync -av ../soe-data/sif/soe.sif faster-claude:/scratch/user/hieult/research/soe-data/sif/`.
3. On the FASTER login node (the system Python is 3.6, so uv brings its own; keep uv's Python
   and cache on scratch, never home: `export UV_CACHE_DIR=$SOE_DATA/cache/uv
   UV_PYTHON_INSTALL_DIR=$SOE_DATA/cache/uv-python`). Models are ~30 GB: run it with
   `setsid nohup ... > $SOE_DATA/logs/stage_models.log 2>&1 &`.
   `export SOE_DATA=/scratch/user/hieult/research/soe-data && cd /scratch/user/hieult/research/sharpen-or-expand && HF_HOME=$SOE_DATA/hf uv run scripts/stage_offline.py --assets scripts/stage_assets.yaml`
4. Datasets: `sbatch slurm/build_datasets.sbatch` builds them on a CPU node inside the container,
   offline from the staged snapshots, and fails unless the result is byte-identical to the
   committed `docs/reports/m0_datasets_v1.json` and every `eval_suite.yaml` hash matches.
5. `mkdir -p logs/slurm && sbatch slurm/check_container.sbatch`, then again with
   `SBATCH_GRES=gpu:a40:1`.

## Everyday flow
1. Edit locally on a work branch `m<k>/<topic>`; `make check`; commit; push.
2. On FASTER: `git fetch && git switch m<k>/<topic>` (or `git pull --ff-only`). Never edit there; a
   fix made on FASTER goes on `faster/<topic>` and is merged locally.
3. Submit from the repo root:
   - new run, chained up to 2 more times: `CHAIN_LEFT=2 sbatch slurm/train.sbatch examples/dapo_trainer/m3_screen_qwen35-4b.yaml 1`
   - resume: `RUN_DIR=runs/<run_id> sbatch slurm/train.sbatch examples/dapo_trainer/m3_screen_qwen35-4b.yaml`
   - evaluate a checkpoint: start several workers, any sampling GPU type, e.g.
     `SBATCH_GRES=gpu:a30:1 sbatch slurm/eval.sbatch soe/trainer/config/eval_suite.yaml <checkpoint> $SOE_DATA/eval/<name>`;
     each claims shards until none are left (`--shard k` finishes an interrupted one). Then
     `sbatch slurm/cpu.sbatch python -m soe.trainer.main_eval --suite ... --checkpoint ... --output ... --stage report`
     writes `report.json` and `per_problem.jsonl`. Smoke test: add `--limit-per-set 5`.
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
- Every compile/kernel cache (vLLM, torch.compile, Triton, FlashInfer, CUDA) lives in
  `$SOE_DATA/cache/`, never under `$HOME` (10k-file quota); `check_container` fails otherwise.
- Override GPU type and time limit with `SBATCH_GRES` / `SBATCH_TIMELIMIT` environment
  variables, not `--gres` / `--time`: chained successors inherit the environment, not the
  command line.

## Status
Untested on FASTER (the cluster is down as of 2026-10-26): the image has not been built, and no
job from these templates has run. M0.1–M0.2 exercise them.
