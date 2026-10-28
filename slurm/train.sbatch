#!/bin/bash
# Lane A training: 4xA40 on node fc098 (plan revision 2026-10-28; confirmed at G0). verl starts its
# own single-node Ray cluster in the job.
# New run:     CHAIN_LEFT=2 sbatch slurm/train.sbatch <config.yaml> <seed>
# Resume:      RUN_DIR=runs/<run_id> sbatch slurm/train.sbatch <config.yaml>
# Fallback:    SBATCH_GRES=gpu:a100:2 CHAIN_LEFT=2 sbatch --nodelist=<A100 node> slurm/train.sbatch <config.yaml> <seed>
#              (only on the user's request, for a whole comparison block; plan revision 2026-10-28)
# Lane B dev:  SBATCH_GRES=gpu:a40:2 CHAIN_LEFT=1 sbatch --nodelist=fc096 slurm/train.sbatch <config.yaml> <seed>
# GPU type and count: set SBATCH_GRES, not --gres. sbatch reads it, and chained jobs inherit it.
# Node: Slurm has no nodelist input variable, so a --nodelist given on the command line is not
# inherited. chain_next_job therefore pins every successor to the node its predecessor ran on.
# CPUs scale with the GPU count and RAM with the CPUs, so that a 2-GPU job on a shared node does
# not take the whole node: 4 GPUs -> 60 CPUs and 240G (fc098 has 250G), 2 GPUs -> 30 CPUs and 120G.
# FASTER's submit filter accepts only --mem or --mem-per-cpu, not --mem-per-gpu. Override with
# SBATCH_CPUS_PER_GPU / SBATCH_MEM_PER_CPU, which chained jobs inherit too.
#SBATCH --job-name=soe-train
#SBATCH --account=142788516569
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:4
#SBATCH --nodelist=fc098
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=15
#SBATCH --mem-per-cpu=4G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
CONFIG="${1:?usage: sbatch slurm/train.sbatch <config.yaml> [seed]}"
SEED="${2:-1}"
source slurm/common.sh

if [[ -z "${RUN_DIR:-}" ]]; then
    RUN_DIR="$(run_in_container python -m soe.utils.run --config "$CONFIG" --seed "$SEED")"
fi
if [[ -f "$RUN_DIR/DONE" ]]; then
    echo "run already finished: $RUN_DIR"
    exit 0
fi
echo "run dir: $RUN_DIR"
# A run never switches hardware: its first job records the GPUs, and every later job must match.
# Checked before chaining, so a job on the wrong hardware does not queue successors.
GPUS_NOW="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c | xargs)"
if [[ ! -f "$RUN_DIR/hardware.txt" ]]; then
    echo "$GPUS_NOW" > "$RUN_DIR/hardware.txt"
elif [[ "$(cat "$RUN_DIR/hardware.txt")" != "$GPUS_NOW" ]]; then
    echo "refusing to run: $RUN_DIR started on '$(cat "$RUN_DIR/hardware.txt")', this job has '$GPUS_NOW'" >&2
    exit 1
fi
chain_next_job slurm/train.sbatch "$CONFIG"
run_in_container python -m soe.trainer.main_rl --run-dir "$RUN_DIR"
