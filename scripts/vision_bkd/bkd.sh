#!/bin/bash
# Blockwise distillation: VGG16 → VGG16-DS on CIFAR-10
# Uses copy_matching to copy teacher's trained head + first conv to student (frozen)
source "$(dirname "$0")/../_common.sh"
run_train teacher=vgg16_cifar10 student=vgg16_cifar10_depthwise_separable \
    distiller=bkd data=cifar10 \
    init.method=copy_matching \
    training.num_epochs=100 training.batch_size=128 \
    training.learning_rate=1e-3 training.lr_scheduler_type=cosine \
    training.warmup_ratio=0.05 \
    training.eval_strategy=epoch training.save_strategy=epoch \
    training.eval_steps=1 training.save_steps=1 \
    training.logging_steps=50 training.early_stopping_patience=0 \
    training.bf16=false training.fp16=false run.dtype=float32 "$@"
