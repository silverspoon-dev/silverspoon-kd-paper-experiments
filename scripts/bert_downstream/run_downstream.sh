#!/bin/bash
# Focused MNLI head-to-head: silverspoon-kd vs TextBrewer.
#
# Runs 2 students (T6, T4-tiny) × 4 methods:
#   1. Baseline (no teacher)
#   2. TextBrewer GeneralDistiller (hidden_mse + NST, T=8)
#   3. silverspoon-kd HKD (apples-to-apples with TextBrewer)
#   4. silverspoon-kd BKD → fine-tune (the differentiator)
#
# Requires a fine-tuned MNLI teacher (run finetune_teachers.sh first).
#
# Usage:
#   DEVICES=0 bash scripts/bert_downstream/run_downstream.sh
#   STUDENT=bert_T6 bash scripts/bert_downstream/run_downstream.sh   # single student
#   DRY_RUN=true bash scripts/bert_downstream/run_downstream.sh      # preview commands
source "$(dirname "$0")/../_common.sh"

TEACHER=bert_base_cased_mnli
DATA=glue_mnli
MODEL_TYPE=sequence_classification
DTYPE_OVERRIDE="run.dtype=float32"

if [ -n "${STUDENT:-}" ]; then
    STUDENTS=("$STUDENT")
else
    STUDENTS=(bert_T6 bert_T4_tiny)
fi

get_experiment_name() {
    CUDA_VISIBLE_DEVICES="" python "$PROJECT_DIR/train.py" "$@" run.name_only=true 2>/dev/null
}

# Common overrides: use bert-base-cased as init base, set model type
COMMON="student.base_model=google-bert/bert-base-cased student.model_type=$MODEL_TYPE"

echo "================================================================"
echo "  MNLI Head-to-Head: silverspoon-kd vs TextBrewer"
echo "================================================================"

# ── 1. TextBrewer runs ────────────────────────────────────────────────────
echo ""
echo "--- TextBrewer (GeneralDistiller + hidden_mse + NST) ---"

for student in "${STUDENTS[@]}"; do
    # Map config names to TextBrewer student names
    case "$student" in
        bert_T6)     TB_STUDENT="T6" ;;
        bert_T4_tiny) TB_STUDENT="T4-tiny" ;;
        *) echo "Unknown student: $student"; continue ;;
    esac

    echo "  [$student] TextBrewer distillation on MNLI"
    local_cmd="CUDA_VISIBLE_DEVICES=$DEVICES python $PROJECT_DIR/scripts/bert_downstream/run_textbrewer.py \
        --student $TB_STUDENT --batch_size 128 --num_epochs 30 --lr 1e-4 --patience 5"
    echo ">>> $local_cmd"
    [ "$DRY_RUN" = "true" ] || eval "$local_cmd"
done

# ── 2. silverspoon-kd runs ────────────────────────────────────────────────
for student in "${STUDENTS[@]}"; do
    echo ""
    echo "--- Student: $student (silverspoon-kd) ---"

    # ── Baseline (no distillation) ─────────────────────────────────────
    echo "  [baseline] Fine-tuning $student on MNLI"
    run_train student=$student distiller=standard data=$DATA training=downstream \
        $COMMON student.reconfig.freeze_copied_weights=false \
        $DTYPE_OVERRIDE "$@"

    # ── HKD (single-stage) — apples-to-apples with TextBrewer ───────
    # Match TextBrewer's setup exactly: T=8, KD+feature matching, no CE
    # (alpha=0), 30 epochs, LR=1e-4. Single stage — no stage-2 fine-tune,
    # since TextBrewer doesn't have one.
    # response_loss_weight=1/T²=1/64: SK's KL loss multiplies by T²
    # internally, but TB's kd_ce_loss does not. 1/T² cancels this out
    # so both toolkits apply the same effective KD weight.
    echo "  [hkd] Holistic KD on MNLI (single-stage, matching TextBrewer)"
    run_train teacher=$TEACHER student=$student distiller=hkd data=$DATA training=downstream \
        student.base_model=google-bert/bert-base-cased \
        student.reconfig.freeze_copied_weights=false \
        +distiller.temperature=8 +distiller.response_loss_weight=0.015625 \
        $DTYPE_OVERRIDE "$@"

    # ── BKD (2 stages) — silverspoon-kd's differentiator ──────────────
    echo "  [bkd] Stage 1: blockwise alignment on MNLI"
    run_train teacher=$TEACHER student=$student distiller=bkd data=$DATA training=downstream \
        student.base_model=google-bert/bert-base-cased \
        student.reconfig.freeze_copied_weights=false \
        $DTYPE_OVERRIDE "$@"

    BKD_NAME=$(get_experiment_name teacher=$TEACHER student=$student distiller=bkd data=$DATA \
        training=downstream student.base_model=google-bert/bert-base-cased $DTYPE_OVERRIDE)
    echo "  [bkd] Stage 2: fine-tune on MNLI (from $BKD_NAME)"
    run_train student=$student distiller=standard data=$DATA training=downstream \
        $COMMON student.architecture=from_pretrained \
        student.pretrained_path="$RUNS_DIR/$BKD_NAME/student_model" \
        training.learning_rate=2e-5 \
        $(from_run "$BKD_NAME") $DTYPE_OVERRIDE "$@"

done

echo ""
echo "=== All downstream experiments completed ==="
echo "Run evaluate_downstream.sh to produce the comparison table."
