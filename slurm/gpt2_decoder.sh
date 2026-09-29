#!/bin/bash
#SBATCH --job-name=gpt2-decoder
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%j.out

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

SCRIPTS=(
    # KD experiments (core results)
    scripts/gpt2_distillation/reskd.sh
    scripts/gpt2_distillation/hkd.sh
    scripts/gpt2_distillation/bkd.sh
    # Stage 2 (depends on bkd.sh)
    scripts/gpt2_distillation/bkd_hkd.sh
    # Stage 3 (depends on bkd_hkd.sh)
    scripts/gpt2_distillation/bkd_hkd_lora.sh
    # Baseline last (reference only)
    scripts/gpt2_distillation/baseline.sh
)

for seed in $(seed_list); do
    [ "$SEEDS" -gt 1 ] && echo "======== Seed $seed / $SEEDS ========"
    for script in "${SCRIPTS[@]}"; do
        echo "=== Running $(basename "$script") ==="
        bash "$script" run.seed=$seed "$@"
    done
done

# Evaluation (runs once, evaluates all trained models)
echo "=== Evaluating all GPT-2 distillation runs ==="
bash scripts/gpt2_distillation/eval_all.sh "$@"
