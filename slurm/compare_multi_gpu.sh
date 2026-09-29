#!/bin/bash
#SBATCH --job-name=cmp-multi-gpu
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A40:2
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j.out

# Multi-GPU DDP toolkit comparisons (2 GPUs).
#
# Part 1: Vision DDP — ResNet-50→18 / ImageNet (torchdistill vs silverspoon-kd)
# Part 2: NLP DDP   — BERT-base→T6 / MNLI   (TextBrewer vs silverspoon-kd)
# Part 3: LLM DDP   — Qwen3-4B→0.6B / Dolma (DistillKit vs silverspoon-kd)
#
# Uses 2 GPUs — the standard DDP config for all toolkits.
# Placement + OOM boundary (needs 4 GPUs) are in compare_multi_gpu_placement.sh.

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

PROJECT_DIR="$SLURM_SUBMIT_DIR"
OUTPUT="$PROJECT_DIR/results/multi_gpu_comparison.json"
mkdir -p "$PROJECT_DIR/results"

for PART in vision nlp llm; do
    echo "=== Multi-GPU DDP: $PART ==="
    CUDA_VISIBLE_DEVICES=0,1 python "$PROJECT_DIR/scripts/toolkit_comparison/compare_multi_gpu.py" \
        --part "$PART" \
        --output "$OUTPUT"
done

echo "=== Done ==="
