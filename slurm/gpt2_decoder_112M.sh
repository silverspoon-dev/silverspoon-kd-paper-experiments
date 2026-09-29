#!/bin/bash
#SBATCH --job-name=gpt2-112M
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm-%j.out

# GPT-2 112M distillation: same-arch, larger student (vs 96M in gpt2_decoder.sh)

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

SCRIPTS=(
    # KD experiments
    scripts/gpt2_distillation_112M/reskd.sh
    scripts/gpt2_distillation_112M/hkd.sh
    scripts/gpt2_distillation_112M/bkd.sh
    # Stage 2 (depends on bkd.sh)
    scripts/gpt2_distillation_112M/bkd_hkd.sh
    # Baseline last (reference only)
    scripts/gpt2_distillation_112M/baseline.sh
)

for seed in $(seed_list); do
    [ "$SEEDS" -gt 1 ] && echo "======== Seed $seed / $SEEDS ========"
    for script in "${SCRIPTS[@]}"; do
        echo "=== Running $(basename "$script") ==="
        bash "$script" run.seed=$seed "$@"
    done
done

echo "=== Evaluating all GPT-2 112M distillation runs ==="
bash scripts/gpt2_distillation_112M/eval_all.sh "$@"
