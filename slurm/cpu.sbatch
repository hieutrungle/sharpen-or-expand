#!/bin/bash
# Run one command in the container on a CPU node (evaluation reports, selection, data checks).
# Usage: sbatch [--time=HH:MM:SS] slurm/cpu.sbatch <command> [args...]
#SBATCH --job-name=soe-cpu
#SBATCH --account=142788516569
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
(( $# )) || { echo "usage: sbatch slurm/cpu.sbatch <command> [args...]" >&2; exit 1; }
source slurm/common.sh
run_in_container "$@"
