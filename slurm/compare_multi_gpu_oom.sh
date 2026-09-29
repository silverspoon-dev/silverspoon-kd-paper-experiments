#!/bin/bash
#SBATCH --job-name=cmp-mgpu-oom
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A40:2
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --output=slurm-%j.out

# Part 5: OOM Boundary — max batch size probing per placement strategy.
#
# Uses 2 GPUs only to ensure 100% GPU utilization (all probes use 2 ranks).
# Split-GPU OOM is skipped (needs 4 GPUs); the split-GPU demo is in
# compare_multi_gpu_placement.sh (Part 4) instead.
#
# Probes DistillKit (replicated) vs silverspoon-kd (replicated, sharded)
# at seq_len 1024 and 2048 to find each toolkit's max batch size.

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

PROJECT_DIR="$SLURM_SUBMIT_DIR"
OUTPUT="$PROJECT_DIR/results/multi_gpu_comparison.json"
mkdir -p "$PROJECT_DIR/results"

echo "=== Multi-GPU: OOM boundary ==="
CUDA_VISIBLE_DEVICES=0,1 python "$PROJECT_DIR/scripts/toolkit_comparison/compare_multi_gpu.py" \
    --part oom \
    --output "$OUTPUT"

echo "=== Done ==="
