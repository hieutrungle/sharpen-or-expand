#!/bin/bash
# Evaluation worker: claims shards of the frozen suite for one checkpoint until none are left.
# Submit several at once on any sampling GPU type (A40, A30, A10, A100; see docs/cluster.md):
#   SBATCH_GRES=gpu:a30:1 sbatch slurm/eval.sbatch <suite.yaml> <checkpoint> <output dir> [extra args]
# Extra args go to soe.trainer.main_eval (e.g. --shard 7, --limit-per-set 5 for a smoke test).
# Tables afterwards: sbatch slurm/cpu.sbatch python -m soe.trainer.main_eval --suite ... --stage report
#SBATCH --job-name=soe-eval
#SBATCH --account=142788516569
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
SUITE="${1:?usage: sbatch slurm/eval.sbatch <suite.yaml> <checkpoint> <output dir> [extra args]}"
CHECKPOINT="${2:?missing checkpoint}"
OUTPUT="${3:?missing output dir}"
shift 3
source slurm/common.sh
run_in_container python -m soe.trainer.main_eval --suite "$SUITE" --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" --stage sample "$@"
