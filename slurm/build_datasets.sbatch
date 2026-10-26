#!/bin/bash
# Build the M0 problem files (examples/data_preprocess/) on a CPU node inside the container,
# offline from the staged snapshots, then prove they are byte-identical to the committed build:
# the rebuilt report must leave git clean, and the eval suite's sha256 values must all match.
# Usage: sbatch slurm/build_datasets.sbatch
#SBATCH --job-name=soe-build-data
#SBATCH --account=142788516569
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
source slurm/common.sh

run_in_container python examples/data_preprocess/m0_datasets.py \
    --config examples/data_preprocess/m0_datasets.yaml
if git diff --quiet -- docs/reports/; then
    echo "ok: rebuilt report is identical to the committed docs/reports/m0_datasets_v1.json"
else
    echo "MISMATCH: rebuilt report differs from the committed one" >&2
    git diff --stat -- docs/reports/ >&2
    exit 1
fi
run_in_container python scripts/check_eval_suite.py
echo "build_datasets: all passed"
