#!/bin/bash
#SBATCH --job-name=cmp-mgpu-place
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A40:4
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%j.out

# Part 4: Teacher Placement — replicated vs FSDP-sharded vs split-GPU.
#
# Needs 4 GPUs for split-GPU (teacher on GPUs 2-3, student on GPUs 0-1).
# Uses --llm_steps 200 to keep runs short and avoid efficiency-monitor kill
# (2/4 GPUs are idle during replicated/sharded runs).
#
# OOM boundary probing (Part 5) runs separately in compare_multi_gpu_oom.sh
# with only 2 GPUs for 100% utilization.

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

PROJECT_DIR="$SLURM_SUBMIT_DIR"
OUTPUT="$PROJECT_DIR/results/multi_gpu_comparison.json"
mkdir -p "$PROJECT_DIR/results"

echo "=== Multi-GPU: placement ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 python "$PROJECT_DIR/scripts/toolkit_comparison/compare_multi_gpu.py" \
    --part placement \
    --llm_steps 200 \
    --output "$OUTPUT"

echo "=== Done ==="
