#!/bin/bash
# Joint CE + RelKD DA training: ResNet50 → VGG11-BN on CIFAR-100
# Paper reference: Park et al. "Relational Knowledge Distillation" CVPR 2019, Table 4
# Paper reports RKD-DA accuracy: 72.97% using L_CE + 25*L_D + 50*L_A.
#
# See hkd_relkd_distance_joint.sh for hyperparameter rationale.
# The relkd_da loss internally combines: L_DA = 1*L_D + 2*L_A
# With loss_weight=25: 25*(1*L_D + 2*L_A) = 25*L_D + 50*L_A — matches paper.
# alpha=0.5: L = 0.5*(L_CE + 25*L_D + 50*L_A), LR=0.2.
source "$(dirname "$0")/../_common.sh"
TEACHER_MODEL="$RUNS_DIR/hf__standard__resnet50_cifar100__scratch/model"
run_train teacher=resnet50_cifar100 student=vgg11bn_cifar100 \
    teacher.name="$TEACHER_MODEL" \
    distiller=hkd loss=relkd_da data=cifar100 \
    training.num_epochs=200 training.batch_size=128 \
    training.learning_rate=0.2 training.lr_scheduler_type=constant \
    training.optim=sgd '+training.optim_args="momentum=0.9"' \
    +training.weight_decay=2.5e-4 \
    +training.lr_milestones=[60,120,160] +training.lr_milestone_gamma=0.2 \
    training.bf16=false training.fp16=false \
    training.torch_compile=false run.dtype=float32 \
    training.eval_strategy=epoch training.save_strategy=epoch \
    training.eval_steps=1 training.save_steps=1 \
    training.logging_steps=50 training.warmup_ratio=0.0 \
    training.early_stopping_patience=0 \
    training.max_grad_norm=0 \
    training.metric_for_best_model=eval_accuracy training.greater_is_better=true \
    +distiller.alpha=0.5 \
    +distiller.alignment.auto_projector=false \
    +distiller.alignment.loss_weight=25 \
    +distiller.response_loss_weight=0 \
    distiller.magnitude_aware_weighting=false \
    "$@"
