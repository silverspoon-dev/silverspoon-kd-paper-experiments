#!/bin/bash
# Shared SLURM helpers. Source from SLURM scripts.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Multi-seed support: SEEDS=1 (default) uses seed 42; SEEDS=N runs seeds 1..N.
SEEDS=${SEEDS:-1}

seed_list() {
    if [ "$SEEDS" -le 1 ]; then
        echo "42"
    else
        seq 1 "$SEEDS"
    fi
}

setup_env() {
    # SLURM cluster environment setup.
    # The original experiments used Lmod with Miniforge3 + CUDA modules and a
    # shared conda env on cluster scratch.  Adapt to your local module system
    # by setting SLURM_CONDA_ENV (path to a conda env containing the deps from
    # requirements.txt) and the module names below.
    module load "${MOD_MINIFORGE:-Miniforge3/24.1.2-0}"
    module load "${MOD_CUDA:-CUDA/13.0.0}"
    source activate "${SLURM_CONDA_ENV:?set SLURM_CONDA_ENV to your conda env path}"
    export TOKENIZERS_PARALLELISM=true
    export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
    cd "$PROJECT_DIR"

    # Background GPU memory monitor — samples every 30s, writes peak to
    # gpu_monitor_<jobid>.csv.  Automatically killed when the job exits.
    _start_gpu_monitor
}

_start_gpu_monitor() {
    local logfile="${PROJECT_DIR}/runs/gpu_monitor_${SLURM_JOB_ID:-$$}.csv"
    (
        echo "timestamp,memory_used_MiB,memory_total_MiB,utilization_pct" > "$logfile"
        while true; do
            nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
                --format=csv,noheader,nounits >> "$logfile" 2>/dev/null
            sleep 30
        done
    ) &
    _GPU_MONITOR_PID=$!
    trap "kill $_GPU_MONITOR_PID 2>/dev/null; wait $_GPU_MONITOR_PID 2>/dev/null" EXIT
    echo "GPU monitor started (PID=$_GPU_MONITOR_PID, log=$logfile)"
}

submit_experiment() {
    # Usage: submit_experiment <script_path> [sbatch_args...]
    local script="$1"; shift
    sbatch "$@" "$script"
}
