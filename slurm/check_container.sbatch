#!/bin/bash
# GPU stack check plus offline load of every staged model, then a few vLLM tokens on 2B.
# Usage: sbatch slurm/check_container.sbatch                  (A100; every pinned model must be staged)
#        bash slurm/check_container.sbatch --allow-missing    (local; skips models not staged)
#        SBATCH_GRES=gpu:a40:1 sbatch slurm/check_container.sbatch
#SBATCH --job-name=soe-check
#SBATCH --account=142788516569
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
source slurm/common.sh
run_in_container python scripts/check_container.py --vllm Qwen/Qwen3.5-2B-Base "$@"
