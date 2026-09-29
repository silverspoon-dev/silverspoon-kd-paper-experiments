#!/bin/bash
#SBATCH --job-name=linear-attn
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm-%j.out

# GPT-2 linearization: LoLCATs hybrid (sliding-window + linear attention)
# 6 experiments: 3 standalone + 3 two-stage BKD pipelines, plus the no-KD baseline

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

SCRIPTS=(
    # Standalone methods (from scratch)
    scripts/gpt2_linearization/lolcats_hkd.sh
    scripts/gpt2_linearization/lolcats_reskd.sh
    # BKD stage-1 (attention alignment — required before stage-2 scripts)
    scripts/gpt2_linearization/lolcats_bkd.sh
    # BKD stage-2 pipelines (depend on BKD above)
    scripts/gpt2_linearization/lolcats_bkd_hkd.sh
    scripts/gpt2_linearization/lolcats_bkd_reskd.sh
    scripts/gpt2_linearization/lolcats_bkd_ft.sh
    # Baseline last (reference only)
    scripts/gpt2_linearization/lolcats_baseline.sh
)

for seed in $(seed_list); do
    [ "$SEEDS" -gt 1 ] && echo "======== Seed $seed / $SEEDS ========"
    for script in "${SCRIPTS[@]}"; do
        echo "=== Running $(basename "$script") ==="
        bash "$script" run.seed=$seed "$@"
    done
done

echo "=== Evaluating all LoLCATs runs ==="
bash scripts/gpt2_linearization/eval_all.sh
