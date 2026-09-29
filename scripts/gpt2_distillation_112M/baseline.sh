#!/bin/bash
# Baseline: GPT-2 112M trained from scratch on Dolma (no distillation)
# LR=1e-3, same as the 96M baseline. Needs higher max_grad_norm because
# n_inner=2400 with default init (std=0.02, no residual scaling) causes
# gradient norms of ~140k vs ~5 for the 96M (n_inner=1536). With the
# default max_grad_norm=1.0, clipping reduces the effective step by
# 140,000x, preventing any learning.
source "$(dirname "$0")/../_common.sh"
run_train student=gpt2_112M distiller=standard data=dolma \
    training.learning_rate=1e-3 training.batch_size=32 \
    training.max_steps=70000 training.max_grad_norm=0 "$@"
