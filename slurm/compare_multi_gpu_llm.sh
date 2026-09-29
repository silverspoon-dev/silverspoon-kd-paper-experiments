#!/bin/bash
#SBATCH --job-name=cmp-mgpu-llm
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A40:2
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%j.out

# Part 3: LLM DDP — Qwen3-4B→0.6B / Dolma (DistillKit vs silverspoon-kd).
# 2 GPUs for DDP with 2 ranks — 100% GPU utilization.
#
# Parts 4 (placement) and 5 (OOM) are separate jobs:
#   compare_multi_gpu_placement.sh — 4 GPUs for split-GPU demo
#   compare_multi_gpu_oom.sh       — 2 GPUs for OOM boundary probing

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

PROJECT_DIR="$SLURM_SUBMIT_DIR"
mkdir -p "$PROJECT_DIR/results"

echo "=== Multi-GPU: LLM DDP ==="
CUDA_VISIBLE_DEVICES=0,1 python "$PROJECT_DIR/scripts/toolkit_comparison/compare_multi_gpu.py" \
    --part llm \
    --output "$PROJECT_DIR/results/multi_gpu_comparison.json"

echo "=== Done ==="
