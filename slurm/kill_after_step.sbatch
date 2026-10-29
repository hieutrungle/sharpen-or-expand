#!/bin/bash
# M1.5 kill-and-resume test: cancel a training job as soon as its run has logged a given step.
# Usage: sbatch slurm/kill_after_step.sbatch <training job id> <step>
# The run directory is read from the job's own log ("run dir: ..."), since train.sbatch creates it.
# Plain bash on a CPU node: no container, only squeue / scontrol / scancel.
#SBATCH --job-name=soe-kill
#SBATCH --account=142788516569
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
JOB="${1:?usage: sbatch slurm/kill_after_step.sbatch <job id> <step>}"
STEP="${2:?missing step}"

alive() { [[ -n "$(squeue -h -j "$JOB" -o %T 2>/dev/null)" ]]; }

LOG="$(scontrol show job "$JOB" | grep -oP 'StdOut=\K\S+')"
RUN_DIR=""
while [[ -z "$RUN_DIR" ]]; do
    alive || { echo "job $JOB ended before naming its run dir" >&2; exit 1; }
    RUN_DIR="$(grep -oP '^run dir: \K\S+' "$LOG" 2>/dev/null | head -1 || true)"
    sleep 30
done
METRICS="$RUN_DIR/metrics/segment-$JOB.jsonl"
echo "watching $METRICS for step $STEP"
until [[ -f "$METRICS" ]] && grep -q "^{\"step\":$STEP," "$METRICS"; do
    alive || { echo "job $JOB ended before logging step $STEP" >&2; exit 1; }
    sleep 30
done
echo "step $STEP logged at $(date -Is); cancelling job $JOB"
scancel "$JOB"
