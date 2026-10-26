#!/bin/bash
# Lane A training: 2xA100 on one node; verl starts its own single-node Ray cluster in the job.
# New run:     CHAIN_LEFT=2 sbatch slurm/train.sbatch <config.yaml> <seed>
# Resume:      RUN_DIR=runs/<run_id> sbatch slurm/train.sbatch <config.yaml>
# Lane B dev:  SBATCH_GRES=gpu:a40:2 sbatch slurm/train.sbatch <config.yaml> <seed>
# (Use SBATCH_GRES / SBATCH_TIMELIMIT rather than --gres / --time: chained jobs inherit
#  environment variables but not command-line options.)
#SBATCH --job-name=soe-train
#SBATCH --account=142788516569
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
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
chain_next_job slurm/train.sbatch "$CONFIG"
run_in_container python -m soe.trainer.main_rl --run-dir "$RUN_DIR"
