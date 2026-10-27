# Sourced by every sbatch template (submit from the repo root). Paths follow docs/cluster.md.
# shellcheck shell=bash

SOE_DATA="${SOE_DATA:-${SCRATCH:-$HOME}/research/soe-data}"
SOE_SIF="${SOE_SIF:-$SOE_DATA/sif/soe.sif}"
export SOE_DATA

# Compute nodes have no internet: everything comes from the staged cache.
export HF_HOME="$SOE_DATA/hf"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline WANDB_DIR="$PWD/runs"
# Kernel/compile caches (vLLM, torch.compile, Triton, FlashInfer, CUDA PTX) default to $HOME, which
# Singularity mounts and which FASTER caps at 10k files. Keep them with the data; they persist across
# jobs, so only the first job per GPU type and model pays the ~2 min compile.
SOE_CACHE="$SOE_DATA/cache"
export XDG_CACHE_HOME="$SOE_CACHE/xdg"
export VLLM_CACHE_ROOT="$SOE_CACHE/vllm"
export TORCHINDUCTOR_CACHE_DIR="$SOE_CACHE/torchinductor"
export TRITON_HOME="$SOE_CACHE/triton"
export FLASHINFER_WORKSPACE_BASE="$SOE_CACHE/flashinfer"
export CUDA_CACHE_PATH="$SOE_CACHE/nv"
# Ray's unix sockets must stay under 107 characters, so keep its temp dir short and node-local.
export RAY_TMPDIR="/tmp/ray-${SLURM_JOB_ID:-local}"

# Run cards record the commit, so the commit must describe the code that actually runs.
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "refusing to run: uncommitted changes in $(pwd)" >&2
    exit 1
fi
# A chained job must continue with the same code its run started with.
if [[ -n "${SOE_EXPECTED_COMMIT:-}" && "$(git rev-parse HEAD)" != "$SOE_EXPECTED_COMMIT" ]]; then
    echo "refusing to run: HEAD moved from $SOE_EXPECTED_COMMIT since this run started" >&2
    exit 1
fi
echo "soe git $(git rev-parse HEAD) on $(hostname), job ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# FASTER compute nodes have SingularityCE (not on login nodes); the local machine has Apptainer.
CONTAINER_CMD="$(command -v apptainer || command -v singularity)" || {
    echo "no apptainer or singularity on PATH (container commands only work on compute nodes)" >&2
    exit 1
}
run_in_container() {
    "$CONTAINER_CMD" exec --nv \
        --bind "$SOE_DATA" --bind "$PWD" --pwd "$PWD" \
        --env PYTHONPATH="$PWD" --env PYTHONUNBUFFERED=1 \
        "$SOE_SIF" "$@"
}

# Job chaining: queue a successor that starts when this job ends in any state, so the 96-hour
# limit or a node failure costs a queue wait instead of an idle night. The successor resumes
# RUN_DIR from its latest checkpoint and exits at once if the run wrote DONE.
# Usage: chain_next_job <sbatch script> [script args...]; set CHAIN_LEFT=<n> at first submit.
chain_next_job() {
    local left="${CHAIN_LEFT:-0}"
    if (( left > 0 )); then
        sbatch --dependency="afterany:$SLURM_JOB_ID" \
            --export="ALL,RUN_DIR=$RUN_DIR,CHAIN_LEFT=$((left - 1)),SOE_EXPECTED_COMMIT=$(git rev-parse HEAD)" \
            "$@"
    fi
}
