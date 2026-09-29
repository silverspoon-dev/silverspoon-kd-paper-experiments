#!/bin/bash
# BKD → HKD: VGG16 → VGG16-DS on CIFAR-10 (stage 2: holistic refinement)
source "$(dirname "$0")/../_common.sh"
PARENT="silverspoon-kd__bkd__vgg16_cifar10__vgg16_cifar10_depthwise_separable__scratch"
run_train teacher=vgg16_cifar10 student=vgg16_cifar10_depthwise_separable \
    distiller=hkd data=cifar10 $(from_run "$PARENT") \
    training.num_epochs=100 training.batch_size=128 \
    training.learning_rate=1e-3 training.lr_scheduler_type=cosine \
    training.warmup_ratio=0.05 training.torch_compile=false \
    training.eval_strategy=epoch training.save_strategy=epoch \
    training.eval_steps=1 training.save_steps=1 \
    training.logging_steps=50 training.early_stopping_patience=0 \
    training.bf16=false training.fp16=false run.dtype=float32 "$@"
