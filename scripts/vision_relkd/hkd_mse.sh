#!/bin/bash
# Stage 1 — HKD with MSE loss on the global-pool features: ResNet50 → VGG11-BN on CIFAR-100
# Paired with hkd_mse_ft.sh (stage 2 fine-tune) which trains the classifier head.
# HKD alone only trains feature layers; the fc head stays random — the
# stage-2 CE fine-tune is what makes the model usable for classification.
# NOTE: Not from the paper; silverspoon-kd comparison using auto_projector for dim matching.
source "$(dirname "$0")/../_common.sh"
TEACHER_MODEL="$RUNS_DIR/hf__standard__resnet50_cifar100__scratch/model"
run_train teacher=resnet50_cifar100 student=vgg11bn_cifar100 \
    teacher.name="$TEACHER_MODEL" \
    distiller=hkd loss=mse data=cifar100 \
    training.num_epochs=200 training.batch_size=128 \
    training.learning_rate=0.1 training.lr_scheduler_type=constant \
    training.optim=sgd '+training.optim_args="momentum=0.9"' \
    +training.weight_decay=5e-4 \
    +training.lr_milestones=[60,120,160] +training.lr_milestone_gamma=0.2 \
    training.bf16=false training.fp16=false \
    training.torch_compile=false run.dtype=float32 \
    training.eval_strategy=epoch training.save_strategy=epoch \
    training.eval_steps=1 training.save_steps=1 \
    training.logging_steps=50 training.warmup_ratio=0.0 \
    training.early_stopping_patience=0 \
    training.max_grad_norm=0 \
    "$@"
