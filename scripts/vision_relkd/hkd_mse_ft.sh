#!/bin/bash
# Stage 2 — Standard CE fine-tune from the HKD-MSE stage-1 checkpoint.
# HolisticDistiller does not include the student's own output.loss in the
# training objective, so stage 1 trains features but leaves the classifier
# head random.  Stage 2 runs a plain supervised training pass on the same
# CIFAR-100 data to train the fc head and fine-tune the features.  Same
# 2-stage pattern used by vision_bkd and bert_downstream.
source "$(dirname "$0")/../_common.sh"
# Stage-1 experiment name is the bare HKD run (loss=mse is the default
# name component and collapses to nothing in the experiment name).
PARENT="silverspoon-kd__hkd__resnet50_cifar100__vgg11bn_cifar100__scratch"
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
