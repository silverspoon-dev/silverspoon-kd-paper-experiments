#!/bin/bash
#SBATCH --job-name=vision-relkd
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

# Two-stage pipeline per HKD variant: stage 1 = HolisticDistiller on features,
# stage 2 = standard CE fine-tune on CIFAR-100 (trains the fc classifier head
# that HolisticDistiller cannot train since it discards student.output.loss).
# ReSKD is single-stage because it already combines soft KL + hard CE.
SCRIPTS=(
    # Teacher must be trained first
    scripts/vision_relkd/teacher.sh
    # ReSKD — Hinton KD (single stage; soft+hard combined)
    scripts/vision_relkd/reskd.sh
    # Feature-based HKD variants — each as (stage 1 distill → stage 2 fine-tune)
    scripts/vision_relkd/hkd_mse.sh
    scripts/vision_relkd/hkd_mse_ft.sh
    scripts/vision_relkd/hkd_relkd_angle.sh
    scripts/vision_relkd/hkd_relkd_angle_ft.sh
    scripts/vision_relkd/hkd_relkd_da.sh
    scripts/vision_relkd/hkd_relkd_da_ft.sh
    scripts/vision_relkd/hkd_relkd_distance.sh
    scripts/vision_relkd/hkd_relkd_distance_ft.sh
    # Baseline last (reference only)
    scripts/vision_relkd/baseline.sh
)

for seed in $(seed_list); do
    [ "$SEEDS" -gt 1 ] && echo "======== Seed $seed / $SEEDS ========"
    for script in "${SCRIPTS[@]}"; do
        echo "=== Running $(basename "$script") ==="
        bash "$script" run.seed=$seed "$@"
    done
done

# Post-training eval: harvest best eval_accuracy from each CIFAR-100 run's
# trainer_state.json into results/vision/<run>.json.  Covers both the
# stage-1 HKD runs (will generally have chance-level accuracy because the
# fc head is random) and the stage-2 CE fine-tune runs (the "real" numbers
# for the paper).
echo "=== Collecting CIFAR-100 accuracy ==="
PATTERN='*cifar100*' bash scripts/eval/collect_vision_accuracy.sh
