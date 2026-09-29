#!/bin/bash
#SBATCH --job-name=qwen3-qat
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm-%j.out

# Qwen3-1.7B quantization-aware training with distillation (Exp 8).
# For every bit width in BITS (default: the INT4/INT5/INT6 sweep reported in
# the paper) this trains QAT + BKD, QAT + HKD, QAT + ReSKD and the QAT-only
# baseline, then evaluates them next to the PTQ floor and the bf16 ceiling.
#
# Usage:
#   sbatch slurm/qat.sh              # INT4, INT5, INT6
#   BITS="4 8" sbatch slurm/qat.sh   # other widths (recipes exist for 4-8)

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

BITS="${BITS:-4 5 6}"

for seed in $(seed_list); do
    [ "$SEEDS" -gt 1 ] && echo "======== Seed $seed / $SEEDS ========"
    for bits in $BITS; do
        for method in bkd hkd reskd baseline; do
            echo "=== INT${bits}: ${method} ==="
            bash "scripts/qwen3_quantization/int${bits}_${method}.sh" run.seed=$seed "$@"
        done
    done
done

echo "=== bf16 reference ==="
bash scripts/qwen3_quantization/bf16_pretrained_eval.sh
echo "=== Evaluation (PTQ floor + all QAT runs) ==="
bash scripts/qwen3_quantization/eval_qat.sh $BITS
echo "=== Done ==="
