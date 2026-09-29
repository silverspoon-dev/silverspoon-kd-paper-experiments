#!/bin/bash
# Response-based distillation (Hinton KD): ResNet50 → VGG11-BN on CIFAR-100
# Paper reference: Park et al. "Relational Knowledge Distillation" CVPR 2019, Table 4
# Paper reports HKD (Hinton) accuracy: 74.26% (with T=16, lambda_HKD=16).
#
# Hinton's formula: L = L_CE + lambda * L_soft_KL, i.e. the weights are
# [1 / (1+lambda), lambda / (1+lambda)] on [hard, soft].  In silverspoon-kd,
# ``distiller.alpha`` is the **hard** weight and ``1 - alpha`` is the soft
# weight, so lambda_HKD=16 → alpha = 1/17 ≈ 0.0588.
# (The previous value 0.9 was wrong-way-around and effectively ran the
# student with 90% CE on labels and only 10% KL on teacher soft targets.)
source "$(dirname "$0")/../_common.sh"
TEACHER_MODEL="$RUNS_DIR/hf__standard__resnet50_cifar100__scratch/model"
run_train teacher=resnet50_cifar100 student=vgg11bn_cifar100 \
    teacher.name="$TEACHER_MODEL" \
    distiller=reskd data=cifar100 \
    distiller.temperature=16.0 distiller.alpha=0.0588 \
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
