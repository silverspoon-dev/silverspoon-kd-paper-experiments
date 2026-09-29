#!/bin/bash
# Baseline: Qwen3-1.7B pretrained, QAT at int6 (no teacher)
source "$(dirname "$0")/../_common.sh"
run_train student=qwen3_1.7B_pretrained distiller=standard data=dolma \
    run.student_dtype=int6 student.short_name=qwen3_1.7B_int6 \
    training.learning_rate=2e-5 training.lr_scheduler_type=cosine \
    training.batch_size=4 training.optim=adamw_8bit \
    +training.per_device_eval_batch_size=2 training.max_steps=20000 "$@"
