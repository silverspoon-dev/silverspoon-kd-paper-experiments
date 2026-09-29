#!/bin/bash
#SBATCH --job-name=cmp-distillkit
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j.out

# Toolkit comparison: silverspoon-kd vs DistillKit (single GPU)
# Qwen3-4B → Qwen3-0.6B on Dolma, 500 steps, bf16, gradient checkpointing.
# Each run takes ~15-30 min on A40.

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

PROJECT_DIR="$SLURM_SUBMIT_DIR"
mkdir -p "$PROJECT_DIR/results"

echo "=== Toolkit comparison: DistillKit vs silverspoon-kd (single GPU) ==="
CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/toolkit_comparison/compare_distillkit.py" \
    --num_steps 500 \
    --output "$PROJECT_DIR/results/distillkit_comparison.json"

echo "=== Done ==="
