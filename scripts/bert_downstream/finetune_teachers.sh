#!/bin/bash
# Fine-tune BERT-base-cased teacher on MNLI for the downstream head-to-head.
#
# Usage: DEVICES=0 bash scripts/bert_downstream/finetune_teachers.sh
source "$(dirname "$0")/../_common.sh"

echo "=== Fine-tuning BERT-base-cased on MNLI ==="
run_train student=bert_base_cased_ft distiller=standard data=glue_mnli training=teacher_ft \
    student.model_type=sequence_classification "$@"

echo "=== Teacher fine-tuned ==="
