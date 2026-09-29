#!/bin/bash
#SBATCH --job-name=cmp-textbrewer
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j.out

# Toolkit comparison: silverspoon-kd vs TextBrewer
# BERT-base → BERT-T6 on MNLI, 1000 steps, 4 runs (sequential).
# BERT models are small — each run takes ~5-15 min.

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

PROJECT_DIR="$SLURM_SUBMIT_DIR"
mkdir -p "$PROJECT_DIR/results"

echo "=== Toolkit comparison: TextBrewer vs silverspoon-kd ==="
CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/toolkit_comparison/compare_textbrewer.py" \
    --num_steps 1000 \
    --output "$PROJECT_DIR/results/textbrewer_comparison.json"

echo "=== Done ==="
