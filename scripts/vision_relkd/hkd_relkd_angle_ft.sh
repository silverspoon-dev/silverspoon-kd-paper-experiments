#!/bin/bash
# Stage 2 — Standard CE fine-tune from the HKD-RelKD-angle stage-1 checkpoint.
# See hkd_mse_ft.sh for the 2-stage rationale.
source "$(dirname "$0")/../_common.sh"
PARENT="silverspoon-kd__hkd__resnet50_cifar100__vgg11bn_cifar100__relkd_angle__scratch"
run_train student=vgg11bn_cifar100 distiller=standard data=cifar100 \
    $(from_run "$PARENT") \
    training.num_epochs=100 training.batch_size=128 \
    training.learning_rate=0.01 training.lr_scheduler_type=constant \
    training.optim=sgd '+training.optim_args="momentum=0.9"' \
    +training.weight_decay=5e-4 \
    +training.lr_milestones=[30,60,80] +training.lr_milestone_gamma=0.2 \
    training.bf16=false training.fp16=false \
    training.torch_compile=false run.dtype=float32 \
    training.eval_strategy=epoch training.save_strategy=epoch \
    training.eval_steps=1 training.save_steps=1 \
    training.logging_steps=50 training.warmup_ratio=0.0 \
    training.early_stopping_patience=0 \
    training.max_grad_norm=0 \
    "$@"
