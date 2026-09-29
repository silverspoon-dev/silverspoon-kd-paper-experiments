#!/bin/bash
# ReSKD: GPT-2 → LoLCATs hybrid on Dolma (logit-level distillation)
#
# No alignment pattern needed — ReSKD operates on output logits.
# batch_size=16 + grad_accum=2: the full 50K-token logits for both models
# OOM at bs=32.
source "$(dirname "$0")/../_common.sh"
run_train teacher=gpt2_small student=gpt2_small_lolcats distiller=reskd data=dolma \
    training.learning_rate=1e-2 training.batch_size=16 +training.gradient_accumulation_steps=2 \
    training.max_steps=76000 "$@"
