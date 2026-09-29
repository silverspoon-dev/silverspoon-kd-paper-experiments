#!/bin/bash
#SBATCH --job-name=cmp-torchtune
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=slurm-%j.out

# Toolkit comparison: silverspoon-kd vs torchtune
# Qwen3-4B → Qwen3-0.6B on Dolma, 1000 steps, 3 runs (sequential).
# Each run: ~30-60 min on A40 (teacher + student in bf16).

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

PROJECT_DIR="$SLURM_SUBMIT_DIR"
mkdir -p "$PROJECT_DIR/results"

echo "=== Toolkit comparison: torchtune vs silverspoon-kd ==="
CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/toolkit_comparison/compare_torchtune.py" \
    --num_steps 1000 \
    --output "$PROJECT_DIR/results/torchtune_comparison.json"

echo "=== Done ==="
