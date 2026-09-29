#!/bin/bash
# Response-based distillation: VGG16 → VGG16-DS on CIFAR-10
source "$(dirname "$0")/../_common.sh"
run_train teacher=vgg16_cifar10 student=vgg16_cifar10_depthwise_separable \
    distiller=reskd data=cifar10 \
    training.num_epochs=100 training.batch_size=128 \
    training.learning_rate=1e-3 training.lr_scheduler_type=cosine \
    training.warmup_ratio=0.05 training.torch_compile=false \
    training.eval_strategy=epoch training.save_strategy=epoch \
    training.eval_steps=1 training.save_steps=1 \
    training.logging_steps=50 training.early_stopping_patience=0 "$@"
