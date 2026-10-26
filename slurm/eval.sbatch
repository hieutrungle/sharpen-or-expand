#!/bin/bash
# Lane B evaluation: sample the frozen suite for one checkpoint on one A40.
# Usage: sbatch slurm/eval.sbatch <suite.yaml> <checkpoint dir or hf_id> <output dir>
#SBATCH --job-name=soe-eval
#SBATCH --account=142788516569
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
SUITE="${1:?usage: sbatch slurm/eval.sbatch <suite.yaml> <checkpoint> <output dir>}"
CHECKPOINT="${2:?missing checkpoint}"
OUTPUT="${3:?missing output dir}"
source slurm/common.sh
run_in_container python -m soe.trainer.main_eval --suite "$SUITE" --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT"
