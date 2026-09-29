#!/bin/bash
# HKD: Qwen3-1.7B (bf16) → Qwen3-1.7B (int8), same architecture QAT + distillation
source "$(dirname "$0")/../_common.sh"
run_train teacher=qwen3_1.7B student=qwen3_1.7B_pretrained distiller=hkd loss=mse data=dolma \
    run.student_dtype=int8 student.short_name=qwen3_1.7B_int8 \
    +distiller.response_loss_weight=30 \
    training.learning_rate=2e-5 training.lr_scheduler_type=cosine \
    training.batch_size=2 +training.gradient_accumulation_steps=4 \
    training.gradient_checkpointing=true training.optim=adamw_8bit \
    +training.per_device_eval_batch_size=1 training.max_steps=20000 "$@"
