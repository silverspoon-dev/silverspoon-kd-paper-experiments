#!/bin/bash
# Response-based distillation: Qwen3-1.7B (bf16) → Qwen3-1.7B (int7), same architecture QAT + distillation
source "$(dirname "$0")/../_common.sh"
run_train teacher=qwen3_1.7B student=qwen3_1.7B_pretrained distiller=reskd data=dolma \
    run.student_dtype=int7 student.short_name=qwen3_1.7B_int7 \
    distiller.alpha=0 distiller.temperature=1.0 \
    +distiller.use_liger_kernel=true +distiller.output_head_layer=lm_head \
    training.learning_rate=2e-5 training.lr_scheduler_type=cosine \
    training.batch_size=2 training.gradient_checkpointing=true training.optim=adamw_8bit \
    +training.per_device_eval_batch_size=2 training.max_steps=20000 "$@"
