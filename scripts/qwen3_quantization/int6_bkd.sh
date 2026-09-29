#!/bin/bash
# BKD: Qwen3-1.7B (bf16) → Qwen3-1.7B (int6), same architecture QAT + distillation
source "$(dirname "$0")/../_common.sh"
run_train teacher=qwen3_1.7B student=qwen3_1.7B_pretrained distiller=bkd loss=mse data=dolma \
    run.student_dtype=int6 student.short_name=qwen3_1.7B_int6 \
    training.learning_rate=2e-5 training.lr_scheduler_type=cosine \
    training.batch_size=4 training.optim=adamw_8bit \
    +training.per_device_eval_batch_size=2 training.max_steps=20000 "$@"
