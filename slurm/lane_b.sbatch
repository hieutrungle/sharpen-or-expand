#!/bin/bash
# Lane B: run one command in the container on one A40 (development runs, probes, selection sampling).
# Usage: sbatch [--time=HH:MM:SS] [--job-name=...] slurm/lane_b.sbatch <command> [args...]
#SBATCH --job-name=soe-lane-b
#SBATCH --account=142788516569
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
(( $# )) || { echo "usage: sbatch slurm/lane_b.sbatch <command> [args...]" >&2; exit 1; }
source slurm/common.sh
run_in_container "$@"
