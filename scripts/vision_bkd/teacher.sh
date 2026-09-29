#!/bin/bash
# Teacher: Fine-tune VGG16 on CIFAR-10 (needed before distillation)
source "$(dirname "$0")/../_common.sh"
run_train student=vgg16_cifar10_teacher distiller=standard \
    data=cifar10 \
    training.num_epochs=100 training.batch_size=128 \
    training.learning_rate=1e-3 training.lr_scheduler_type=cosine \
    training.warmup_ratio=0.05 \
    training.eval_strategy=epoch training.save_strategy=epoch \
    training.eval_steps=1 training.save_steps=1 \
    training.logging_steps=50 "$@"
