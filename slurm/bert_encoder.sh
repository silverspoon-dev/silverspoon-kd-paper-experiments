#!/bin/bash
#SBATCH --job-name=bert-enc
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%j.out

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

# Stage 1 — pre-training distillation: 2 students × 4 distillers = 8 runs.
for seed in $(seed_list); do
    [ "$SEEDS" -gt 1 ] && echo "======== Seed $seed / $SEEDS ========"
    echo "=== Running all BERT distillation experiments ==="
    bash scripts/bert_distillation/run_all.sh run.seed=$seed "$@"
done

# Stage 2 — downstream GLUE eval: fine-tune each distilled checkpoint on
# MNLI with a sequence-classification head.  Idempotent: skips runs whose
# stage-2 directory already exists.  Produces the actual numbers that
# prove distilled encoders transfer to downstream tasks.
echo "=== GLUE evaluation of all BERT distillation checkpoints ==="
bash scripts/bert_distillation/eval_glue.sh
