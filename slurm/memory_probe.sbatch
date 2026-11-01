#!/bin/bash
# M1.10 knob probes for one memory-study cell on 4xA40 fc098 (user, 2026-10-31: descending search).
# For each rollout TP, try the gpu_memory_utilization values in the order given, highest first, and
# stop at the first two-step probe that finishes every phase: a lower value cannot generate faster,
# because more KV cache never slows vLLM. Values after the first fit are "not run, dominated".
# Each probe is a normal run directory made by slurm/train.sbatch; its FSDP checkpoint and HF export
# are deleted afterwards, its records stay. One line per probe goes to logs/slurm/memory-probe-<job>.tsv.
# Usage: sbatch slurm/memory_probe.sbatch <cell.yaml> <tp,tp> <gmu,gmu,...>
#        (comma lists: FASTER's sbatch splits script arguments that contain spaces)
# Only fc098 has 4 A40s, so the 4-GPU request needs no --nodelist. A 2-GPU smoke may use any A40
# pair (user, 2026-11-01): SBATCH_GRES=gpu:a40:2 sbatch slurm/memory_probe.sbatch ...
#SBATCH --job-name=soe-m1.10-probe
#SBATCH --account=142788516569
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=15
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
CELL="${1:?usage: sbatch slurm/memory_probe.sbatch <cell.yaml> <tp,tp> <gmu,gmu,...>}"
TPS="${2:?tp list, e.g. 1,2}"
GMUS="${3:?gpu_memory_utilization list, highest first, e.g. 0.7,0.6,0.5,0.4}"
SEED=20261029  # every M1.10 cell trains on the same seeded prompts (docs/roadmap/M1.md)
# Two 4B steps are not timed yet (2-2.7 h estimated with start-up, validation and the save); a
# vLLM engine death now stops a probe at once (main_rl), so this only catches other hangs.
PROBE_TIMEOUT=240m
source slurm/common.sh

summary="logs/slurm/memory-probe-$SLURM_JOB_ID.tsv"
echo -e "tp\tgmu\texit\tseconds\trun_dir" > "$summary"
fits=0
for tp in ${TPS//,/ }; do
    for gmu in ${GMUS//,/ }; do
        mkdir -p "$SOE_JOB_TMP"  # each train.sbatch removes it when it exits
        config="$(run_in_container python scripts/memory_probe_config.py --cell "$CELL" \
            --tp "$tp" --gmu "$gmu" --out-dir runs/memory-probes | tail -n 1)"
        log="logs/slurm/memory-probe-$SLURM_JOB_ID-tp$tp-gmu$gmu.out"
        started=$(date +%s)
        set +e
        timeout --kill-after=5m "$PROBE_TIMEOUT" bash slurm/train.sbatch "$config" "$SEED" > "$log" 2>&1
        status=$?
        set -e
        run_dir="$(grep -m1 '^run dir: ' "$log" | cut -d' ' -f3 || true)"
        echo -e "$tp\t$gmu\t$status\t$(( $(date +%s) - started ))\t$run_dir" | tee -a "$summary"
        # Free the GPUs from anything a failed probe left behind before the next one starts.
        mkdir -p "$SOE_JOB_TMP"
        run_in_container ray stop --force > /dev/null 2>&1 || true
        sleep 15
        if [[ -n "$run_dir" ]]; then
            rm -rf -- "$run_dir/checkpoints" "$SOE_DATA/checkpoints/$(basename "$run_dir")"
        fi
        if (( status == 0 )); then
            fits=$((fits + 1))
            break
        fi
    done
done
# Non-zero when no probe fitted, so a job chained with afterok does not start (the 4B check).
echo "probes that fit: $fits"
(( fits > 0 ))
