#!/bin/bash
#SBATCH --job-name=bert-downstream
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%j.out

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

# Teacher fine-tuning is seed-independent (only needs to run once).
echo "=== Stage 1: Fine-tuning teachers ==="
bash scripts/bert_downstream/finetune_teachers.sh "$@"

# Distillation and evaluation run per seed.
for seed in $(seed_list); do
    [ "$SEEDS" -gt 1 ] && echo "======== Seed $seed / $SEEDS ========"

    echo "=== Stage 2: Running downstream distillation ==="
    bash scripts/bert_downstream/run_downstream.sh run.seed=$seed "$@"

    echo "=== Stage 3: Evaluating all models ==="
    bash scripts/bert_downstream/evaluate_downstream.sh run.seed=$seed "$@"
done
