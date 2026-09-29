#!/bin/bash
#SBATCH --job-name=cmp-oom-ext
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j.out

# Extended OOM Frontier: adds seq=4096 (DK likely OOMs entirely).
# Uses 2 GPUs (replicated + sharded only); split-GPU placement is
# auto-skipped by run_oom_boundary when ngpu<4. Reason: a previous
# 4-GPU run was killed by cluster admins for "severely inefficient
# resource usage" — the OOM probes are bursty, mostly idle, and the
# split-only modes left 2/4 GPUs unused for >50% of wall time.
#
# A100 (40 GB) matches the paper's stated "~40 GB memory budget"
# exactly, and the A100's ~2x bf16 compute (312 vs 150 TFLOPS on A40)
# makes framework-overhead differences more visible (less compute
# saturation masking the comparison).

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

PROJECT_DIR="$SLURM_SUBMIT_DIR"
OUTPUT="$PROJECT_DIR/results/multi_gpu_comparison.json"
mkdir -p "$PROJECT_DIR/results"

echo "=== Extended OOM Frontier: seq=1024,2048,4096, 2 GPUs ==="
CUDA_VISIBLE_DEVICES=0,1 python "$PROJECT_DIR/scripts/toolkit_comparison/compare_multi_gpu.py" \
    --part oom \
    --oom_seq_lengths 1024,2048,4096 \
    --output "$OUTPUT"

echo "=== Done ==="
