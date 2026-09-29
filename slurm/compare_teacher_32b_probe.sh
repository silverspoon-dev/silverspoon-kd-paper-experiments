#!/bin/bash
#SBATCH --job-name=cmp-32b-probe
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=8
#SBATCH --time=1:30:00
#SBATCH --output=slurm-%j.out

# Probe Qwen3-32B (64 GB bf16 weights) feasibility on 2x A100 (40 GB ea):
#  - DK/TB replicated: definitive OOM (teacher alone exceeds per-GPU mem)
#  - SK replicated: definitive OOM (same reason)
#  - SK sharded: 32 GB/GPU teacher leaves ~8 GB for student/grads/acts.
#                Bs=1 should fit; bs>=2 likely OOM.
#
# Time budget includes ~10-15 min of HF model download (~64 GB).
# Merges with existing teacher_scaling section (4B,8B,14B) — does NOT
# overwrite, thanks to the per-section merge in _save_partial.

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

PROJECT_DIR="$SLURM_SUBMIT_DIR"
OUTPUT="$PROJECT_DIR/results/multi_gpu_comparison.json"
mkdir -p "$PROJECT_DIR/results"

echo "=== Qwen3-32B probe: 64 GB teacher on 2x A100 (40 GB), seq=1024 ==="
CUDA_VISIBLE_DEVICES=0,1 python "$PROJECT_DIR/scripts/toolkit_comparison/compare_multi_gpu.py" \
    --part teacher_scaling \
    --teacher_scaling_models "Qwen/Qwen3-32B" \
    --teacher_scaling_seq 1024 \
    --student "Qwen/Qwen3-0.6B" \
    --output "$OUTPUT"

echo "=== Done ==="
