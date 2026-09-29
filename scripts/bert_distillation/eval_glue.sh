#!/bin/bash
# Evaluate BERT distillation checkpoints by fine-tuning on GLUE downstream tasks.
#
# For each pretraining run (2 students × 4 methods = 8 checkpoints), fine-tunes
# on MNLI to measure downstream utility of the distilled encoder.  The
# resulting fine-tune runs are saved under ``runs/hf__standard__<student>__<task>__from-*``.
#
# Only the glue_mnli data config exists in configs/data/ today (SST-2 support
# can be added by creating configs/data/glue_sst2.yaml).
#
# Usage:
#   DEVICES=0 bash scripts/bert_distillation/eval_glue.sh
#   GLUE_TASKS="glue_mnli" bash scripts/bert_distillation/eval_glue.sh
#   DRY_RUN=true bash scripts/bert_distillation/eval_glue.sh       # preview
source "$(dirname "$0")/../_common.sh"

GLUE_TASKS="${GLUE_TASKS:-glue_mnli}"
STUDENTS=(bert_T6 bert_T4_tiny)

find_model_path() {
    local run_dir="$1"
    if [ -d "$run_dir/student_model" ]; then
        echo "$run_dir/student_model"
    elif [ -d "$run_dir/model" ]; then
        echo "$run_dir/model"
    fi
}

eval_checkpoint() {
    local run_name="$1"
    local student_config="$2"
    local run_dir="$RUNS_DIR/$run_name"

    if [ ! -d "$run_dir" ]; then
        echo ">>> SKIP (not found): $run_name"
        return 0
    fi

    local model_path
    model_path=$(find_model_path "$run_dir")
    if [ -z "$model_path" ]; then
        echo ">>> SKIP (no model): $run_name"
        return 0
    fi

    # "$@" inside the function includes run_name and student_config as
    # $1 / $2 — shift them off so only the outer script's passthrough
    # overrides remain for the Hydra command.
    shift 2
    for task in $GLUE_TASKS; do
        echo "  → Fine-tuning on $task (init from $run_name)"
        # Matches bert_downstream stage-2 pattern:
        #   - student config carries the pruned architecture (T6 / T4-tiny)
        #   - from_pretrained reads the distilled checkpoint's MLM weights
        #   - model_type=sequence_classification auto-wraps with a classifier
        #   - init.method=from_experiment threads the source run into the
        #     experiment name so the stage-2 directory is unique per source
        run_train student=$student_config distiller=standard \
            data=$task training=downstream \
            student.base_model=google-bert/bert-base-uncased \
            student.model_type=sequence_classification \
            student.architecture=from_pretrained \
            student.pretrained_path="$model_path" \
            init.method=from_experiment \
            init.from_experiment="$run_name" \
            init.checkpoint_step=latest \
            run.dtype=float32 training.bf16=false \
            "$@"
    done
}

echo "================================================================"
echo "  GLUE Evaluation of BERT Distillation Checkpoints"
echo "================================================================"

for student in "${STUDENTS[@]}"; do
    echo ""
    echo "--- Student: $student ---"

    # Baseline (no distillation)
    echo "  [baseline]"
    eval_checkpoint "hf__standard__${student}__scratch" "$student" "$@"

    # BKD
    echo "  [bkd]"
    eval_checkpoint "silverspoon-kd__bkd__bert_base_uncased__${student}__scratch" "$student" "$@"

    # HKD
    echo "  [hkd]"
    eval_checkpoint "silverspoon-kd__hkd__bert_base_uncased__${student}__scratch" "$student" "$@"

    # Response-based KD
    echo "  [reskd]"
    eval_checkpoint "silverspoon-kd__reskd__bert_base_uncased__${student}__scratch" "$student" "$@"
done

echo ""
echo "=== Running MNLI matched+mismatched evaluation ==="

# evaluate.py produces JSON with both accuracy_matched and accuracy_mismatched.
# The summary script reads these from results/downstream/.
RESULTS_DIR="$(dirname "$RUNS_DIR")/results/downstream"
mkdir -p "$RESULTS_DIR"

for pattern in "hf__standard__bert_T6__mnli__from-*" "hf__standard__bert_T4_tiny__mnli__from-*"; do
    for d in "$RUNS_DIR"/$pattern; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        model_path=""
        [ -d "$d/model" ] && model_path="$d/model"
        [ -d "$d/student_model" ] && model_path="$d/student_model"
        [ -z "$model_path" ] && continue

        out="$RESULTS_DIR/${name}.json"
        if [ -f "$out" ]; then
            echo ">>> SKIP (already evaluated): $name"
            continue
        fi
        echo ">>> Evaluating MNLI (matched+mismatched): $name"
        CUDA_VISIBLE_DEVICES=$DEVICES python "$PROJECT_DIR/evaluate.py" \
            --model_path "$model_path" --task mnli --output_json "$out"
    done
done

echo ""
echo "=== GLUE evaluation complete ==="
