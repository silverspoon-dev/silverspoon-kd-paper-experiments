#!/bin/bash
#SBATCH --job-name=cross-arch
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%j.out

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

SCRIPTS=(
    # KD experiments (HKD/ResKD only — BKD is incompatible cross-architecture
    # because teacher layer inputs can't be fed to student layers when the
    # layer interfaces differ, e.g. GPT-2 MHA kwargs vs DeepSeek-V3 MLA kwargs)
    scripts/gpt2_cross_arch/reskd.sh
    scripts/gpt2_cross_arch/hkd.sh
    # Stage 2 — plain CE fine-tune from each distilled checkpoint.
    # HolisticDistiller forces feature-level alignment that can
    # actively hurt cross-arch (GPT-2 MHA → DeepSeek-V3 MLA); this
    # stage lets the student optimise freely for the LM objective.
    scripts/gpt2_cross_arch/hkd_ft.sh
    scripts/gpt2_cross_arch/reskd_ft.sh
    # Baseline last (reference only)
    scripts/gpt2_cross_arch/baseline.sh
)

for seed in $(seed_list); do
    [ "$SEEDS" -gt 1 ] && echo "======== Seed $seed / $SEEDS ========"
    for script in "${SCRIPTS[@]}"; do
        echo "=== Running $(basename "$script") ==="
        bash "$script" run.seed=$seed "$@"
    done
done

# Evaluation (runs once, evaluates all trained models)
echo "=== Evaluating all cross-architecture runs ==="
bash scripts/gpt2_cross_arch/eval_all.sh "$@"
