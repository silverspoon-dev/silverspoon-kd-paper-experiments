#!/bin/bash
#SBATCH --job-name=bert-glue
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

# Run GLUE MNLI fine-tuning for all BERT pre-training checkpoints.
# Idempotent: skips runs whose stage-2 directory already exists.
# Currently needed for: T6 from-bkd, T6 from-hkd (missing stage-2 evals).

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

echo "=== GLUE evaluation of BERT distillation checkpoints ==="
bash scripts/bert_distillation/eval_glue.sh

echo "=== Done ==="
