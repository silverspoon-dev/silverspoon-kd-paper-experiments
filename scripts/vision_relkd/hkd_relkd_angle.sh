#!/bin/bash
# Stage 1 — HKD with RelKD angle loss on global-pool features: ResNet50 → VGG11-BN on CIFAR-100
# Paper reference: Park et al. "Relational Knowledge Distillation" CVPR 2019, Table 4
# Paired with hkd_relkd_angle_ft.sh for stage-2 CE fine-tuning of the classifier.
source "$(dirname "$0")/../_common.sh"
TEACHER_MODEL="$RUNS_DIR/hf__standard__resnet50_cifar100__scratch/model"
run_train teacher=resnet50_cifar100 student=vgg11bn_cifar100 \
    teacher.name="$TEACHER_MODEL" \
    distiller=hkd loss=relkd_angle data=cifar100 \
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
    +distiller.alignment.auto_projector=false \
    +distiller.response_loss_weight=0 \
    "$@"
