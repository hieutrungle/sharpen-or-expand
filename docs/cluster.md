# Running on TAMU FASTER

Compute nodes have **no internet**; home is capped at 10 GB / 10k files. Everything heavy lives on
scratch in `$SOE_DATA=/scratch/user/hieult/research/soe-data`, and code moves only through GitHub.

## Lanes
| Lane | GPUs | Use | Template |
| --- | --- | --- | --- |
| A | 4×A40 (48 GB) on fc098; target, confirmed at G0 | every 4B / Gemma training run | `slurm/train.sbatch` (default `gpu:a40:4`, `--nodelist=fc098`) |
| A fallback | 2×A100 (PCIe 40 GB), only on the user's request, whole comparison blocks | 4B runs when fc098 is unavailable, or a parallel block | `SBATCH_GRES=gpu:a100:2 sbatch --nodelist=<A100 node> slurm/train.sbatch …` |
| B, development | A40 pairs on fc096 and fc097 | 2B training (2-GPU jobs) | `SBATCH_GRES=gpu:a40:2 sbatch --nodelist=fc096 slurm/train.sbatch …` |
| B, sampling | A30s (fc123, fc124) first, A10s for overflow, spare A40s | evaluation, preference sampling (1-GPU workers) | `SBATCH_GRES=gpu:a30:1 sbatch slurm/eval.sbatch …` |

While no Lane A run is in flight, Lane B jobs use whatever GPUs of their type are free, fc098 included
(user, 2026-10-28), e.g. `SBATCH_GRES=gpu:a40:2 sbatch --nodelist=fc098 slurm/train.sbatch …`. A run stays
on the node it started on (chained jobs are pinned), and runs compared with each other keep one GPU type.

### Training job sizing (`slurm/train.sbatch`)
- FASTER compute nodes are identical: 64 cores, 256 GB RAM and 3.84 TB local NVMe (`/tmp`). GPUs are
  composed onto nodes through the Liqid fabric, up to 10 per node (HPRC hardware page). Checked
  2026-10-28 with `scontrol show node fc098`: CPUTot 64, RealMemory 256000 MB (`CfgTRES mem=250G`),
  `Gres=gpu:a40:4`, partition gpu. fc096 and fc097: 64 CPUs, 256000 MB, `gpu:a40:2` each. For
  current use: `sinfo -N -n fc096,fc097,fc098 -o '%N %c %m %G %T'`, `squeue -w fc098`, `gpuavail`,
  `pestat -p gpu -G`.
- CPUs scale with the GPU count and RAM with the CPUs (`--cpus-per-gpu=15`, `--mem-per-cpu=4G`).
  A 4×A40 job takes 60 CPUs and 240 G of fc098's 250 G. A 2-GPU job takes 30 CPUs and 120 G, so an
  A100 job leaves the rest of a shared A100 node to other users. Raise the RAM with
  `SBATCH_MEM_PER_CPU` if CPU offload (G0) needs more. FASTER's submit filter rejects
  `--mem-per-gpu` ("either --mem or --mem-per-cpu should be specified"). `--cpus-per-gpu` cannot be
  combined with `--cpus-per-task`. All of this was checked with `sbatch --test-only`, which submits
  nothing.
- Pinning: `--nodelist` is a `#SBATCH` directive (fc098) or a command-line option (fallback, dev).
  sbatch has no nodelist input variable (FASTER's Slurm 25.05.6 man page lists none), so
  `chain_next_job` passes `--nodelist=$SLURM_JOB_NODELIST`. Every successor runs on the node where
  its run started, and `train.sbatch` refuses to continue a run whose GPUs differ from those in
  `runs/<id>/hardware.txt`.
- Start-time estimates from `--test-only` were three days out even for idle nodes (2026-10-28).
  In M0, A40 estimates were hours too pessimistic, so queue anyway.
- A pinned job waits when its node is busy, drained or recomposed; it never moves. Check the reason
  with `squeue -u hieult -o '%i %j %T %R'`.

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
     (a run that started off fc098 also needs its `SBATCH_GRES` and `--nodelist`; `hardware.txt` refuses anything else)
   - evaluate a checkpoint: start several workers, any sampling GPU type, e.g.
     `SBATCH_GRES=gpu:a30:1 sbatch slurm/eval.sbatch soe/trainer/config/eval_suite.yaml <checkpoint> $SOE_DATA/eval/<name>`;
     each claims shards until none are left (`--shard k` finishes an interrupted one). Then
     `sbatch slurm/cpu.sbatch python -m soe.trainer.main_eval --suite ... --checkpoint ... --output ... --stage report`
     writes `report.json` and `per_problem.jsonl`. Smoke test: add `--limit-per-set 5`.
4. Poll with `squeue -u hieult` and `sacct -j <id> --format=JobID,State,Elapsed,ExitCode`; read
   `logs/slurm/<name>-<jobid>.out`.
5. Bring results home with rsync from your own terminal, excluding checkpoints:
   `rsync -av --exclude 'checkpoints/' faster-claude:/scratch/user/hieult/research/sharpen-or-expand/runs/ runs/`

## Submission pitfalls on FASTER (seen 2026-10-28)
- FASTER's `sbatch` wrapper (it prints the SU charge notice) re-splits the script's arguments, so an
  argument containing spaces arrives in pieces. `sbatch slurm/cpu.sbatch bash -c 'a && b'` ran only
  `bash -c python`, which exited 0 at once with no output (job 3222582). Pass plain commands only,
  and submit separate jobs instead of chaining commands in one string.
- "Socket timed out on send/recv operation" does not mean the job was rejected: it may already be
  queued. Check `squeue -u hieult` before resubmitting.
- `--parsable` output is mixed with the wrapper's notice; read job ids from `squeue`/`sacct`.

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

## Status (2026-10-28)
- Image `soe.sif` (sha256 `15824634…7196`) is on FASTER; `check_container` passes on A100 (sm_80),
  A40 (sm_86), A30 (sm_80) and A10 (sm_86). T4s are not usable (no BF16).
- In production since M0: `eval.sbatch`, `lane_b.sbatch`, `cpu.sbatch`, `build_datasets.sbatch`,
  `check_container.sbatch` (selection, evaluation, probes, dataset builds).
- `train.sbatch` has not run a training job yet: its directives pass `sbatch --test-only` on fc098 and
  fc097, and the first real run, with chaining, is M1.3/M1.5.
