#!/bin/bash
#SBATCH --job-name=cmp-split-tp
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A40:4
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%j.out

# SK split-placement comparison: FSDP-shard vs TP for inference-only teacher.
# Both modes use 4 GPUs: teacher on [2, 3], student DDP on [0, 1].
#
# Hypothesis: TP avoids FSDP's per-forward all-gather of teacher weights,
# so it should be faster for inference-only teachers (where the gather
# can't be amortized over backward + optimizer).
#
# Sweep: Qwen3-{4B, 8B, 14B} -> Qwen3-0.6B at seq=1024.
#
# Writes to a NEW JSON section "split_comparison" — does NOT touch the
# existing oom_boundary, teacher_scaling, llm, or placement sections.
#
# Utilization note: TP uses all 4 GPUs continuously (teacher TP across
# 2 + student DDP across 2), so this avoids the "half-idle GPUs" pattern
# that got our earlier 4-GPU OOM job admin-killed.

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

PROJECT_DIR="$SLURM_SUBMIT_DIR"
OUTPUT="$PROJECT_DIR/results/multi_gpu_comparison.json"
mkdir -p "$PROJECT_DIR/results"

echo "=== Split-placement comparison: FSDP-shard vs TP, 4x A40 ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 python "$PROJECT_DIR/scripts/toolkit_comparison/compare_multi_gpu.py" \
    --part split_comparison \
    --split_comparison_models "Qwen/Qwen3-4B,Qwen/Qwen3-8B,Qwen/Qwen3-14B" \
    --split_comparison_seq 1024 \
    --student "Qwen/Qwen3-0.6B" \
    --output "$OUTPUT"

echo "=== Done ==="
