#!/bin/bash
#SBATCH --job-name=cmp-teacher-scale
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=8
#SBATCH --time=2:30:00
#SBATCH --output=slurm-%j.out

# Teacher Scaling Frontier: vary teacher size with student + seq fixed.
# Demonstrates SK's FSDP-sharded teacher placement — DK/TB can only
# replicate the teacher across GPUs, so they OOM on teachers that
# exceed per-GPU memory once a student + activations are added.
#
# Setup: 2x A100 (40 GB), Qwen3-0.6B student, seq=1024.
# Teachers: Qwen3-4B (8 GB bf16) -> 8B (~16 GB) -> 14B (~28 GB).
# Expectation: DK/TB OOM at 14B (replicated 28 GB teacher leaves
# nothing for student + acts on a 40 GB card); SK keeps fitting via
# FSDP sharding (~14 GB per GPU teacher).
#
# Incremental JSON saves after each (toolkit, teacher) pair, so a
# preempted job retains whatever measurements completed.

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

PROJECT_DIR="$SLURM_SUBMIT_DIR"
OUTPUT="$PROJECT_DIR/results/multi_gpu_comparison.json"
mkdir -p "$PROJECT_DIR/results"

echo "=== Teacher Scaling Frontier: Qwen3-{4B,8B,14B} -> 0.6B, seq=1024, 2 GPUs ==="
CUDA_VISIBLE_DEVICES=0,1 python "$PROJECT_DIR/scripts/toolkit_comparison/compare_multi_gpu.py" \
    --part teacher_scaling \
    --teacher_scaling_models "Qwen/Qwen3-4B,Qwen/Qwen3-8B,Qwen/Qwen3-14B" \
    --teacher_scaling_seq 1024 \
    --student "Qwen/Qwen3-0.6B" \
    --output "$OUTPUT"

echo "=== Done ==="
