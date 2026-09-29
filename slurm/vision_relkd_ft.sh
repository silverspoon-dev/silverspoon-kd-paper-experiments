#!/bin/bash
#SBATCH --job-name=relkd-ft
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

# Stage 2 CE fine-tuning for RelKD stage-1 runs (angle, DA, distance).
# Stage 1 trains features only (response_loss_weight=0), so the classifier
# head is random (1% accuracy). This stage trains the classifier via CE.

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

echo "=== Running hkd_relkd_angle_ft.sh ==="
bash scripts/vision_relkd/hkd_relkd_angle_ft.sh "$@"

echo "=== Running hkd_relkd_da_ft.sh ==="
bash scripts/vision_relkd/hkd_relkd_da_ft.sh "$@"

echo "=== Running hkd_relkd_distance_ft.sh ==="
bash scripts/vision_relkd/hkd_relkd_distance_ft.sh "$@"

echo "=== Collecting CIFAR-100 accuracy ==="
PATTERN='*cifar100*' bash scripts/eval/collect_vision_accuracy.sh

echo "=== Done ==="
