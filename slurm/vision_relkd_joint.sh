#!/bin/bash
#SBATCH --job-name=relkd-joint
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out

# RelKD with joint CE + relational training in a single stage
# (L_CE + lambda * L_RKD, following Park et al.), for the distance, angle,
# and distance+angle variants.  The two-stage alternative (relational
# stage 1, CE fine-tune stage 2) is driven by vision_relkd.sh.

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

echo "=== Running joint RelKD distance ==="
bash scripts/vision_relkd/hkd_relkd_distance_joint.sh

echo "=== Running joint RelKD angle ==="
bash scripts/vision_relkd/hkd_relkd_angle_joint.sh

echo "=== Running joint RelKD DA ==="
bash scripts/vision_relkd/hkd_relkd_da_joint.sh

echo "=== Collecting CIFAR-100 accuracy ==="
PATTERN='*cifar100*' bash scripts/eval/collect_vision_accuracy.sh

echo "=== Done ==="
