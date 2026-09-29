#!/bin/bash
#SBATCH --job-name=cmp-torchdistill
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%j.out

# Toolkit comparison: silverspoon-kd vs torchdistill (VGG16 / CIFAR-10)
# VGG16 teacher → depthwise-separable student, 5000 steps, 8 runs (sequential).
# Includes: torchdistill baseline/KD/AT, silverspoon-kd baseline/ReSKD/fused/bf16/BKD.

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

PROJECT_DIR="$SLURM_SUBMIT_DIR"
mkdir -p "$PROJECT_DIR/results"

echo "=== Toolkit comparison: torchdistill vs silverspoon-kd (VGG16/CIFAR-10) ==="
CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/toolkit_comparison/compare_torchdistill.py" \
    --num_steps 5000 \
    --output "$PROJECT_DIR/results/torchdistill_comparison.json"

echo "=== Done ==="
