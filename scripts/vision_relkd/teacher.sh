#!/bin/bash
# Train ResNet50 teacher on CIFAR-100 (standard supervised training)
# Paper reference: Park et al. "Relational Knowledge Distillation" CVPR 2019, Table 4
# Paper reports teacher accuracy: 77.76%
#
# IMPORTANT: early stopping disabled.  The default patience=5 trips after a
# few epochs (while CIFAR eval accuracy is still climbing rapidly from a
# random init) and leaves the run stuck at ~44%, well below Park's 77.76%.
source "$(dirname "$0")/../_common.sh"
run_train student=resnet50_cifar100 distiller=standard data=cifar100 \
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
