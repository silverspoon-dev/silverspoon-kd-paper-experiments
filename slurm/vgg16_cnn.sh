#!/bin/bash
#SBATCH --job-name=vgg16-cnn
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

SCRIPTS=(
    # Teacher must be trained first
    scripts/vision_bkd/teacher.sh
    # KD experiments (core results)
    scripts/vision_bkd/reskd.sh
    scripts/vision_bkd/bkd.sh
    # Stage 2 (depends on bkd.sh)
    scripts/vision_bkd/bkd_finetune.sh
    scripts/vision_bkd/bkd_hkd.sh
    scripts/vision_bkd/bkd_reskd.sh
    # Baseline last (reference only)
    scripts/vision_bkd/baseline.sh
)

for seed in $(seed_list); do
    [ "$SEEDS" -gt 1 ] && echo "======== Seed $seed / $SEEDS ========"
    for script in "${SCRIPTS[@]}"; do
        echo "=== Running $(basename "$script") ==="
        bash "$script" run.seed=$seed "$@"
    done
done

# Post-training eval: harvest best eval_accuracy from each CIFAR-10 run's
# trainer_state.json and write results/vision/<run>.json.  Reuses the
# per-epoch eval_accuracy that compute_metrics already produced during
# training, so no extra forward pass is needed.
echo "=== Collecting CIFAR-10 accuracy ==="
PATTERN='*vgg16*' bash scripts/eval/collect_vision_accuracy.sh
