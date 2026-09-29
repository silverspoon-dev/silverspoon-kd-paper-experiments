#!/bin/bash
# HKD: GPT-2 → LoLCATs hybrid on Dolma (attention-level alignment, holistic)
#
# Aligns linearized attention outputs with teacher's softmax attention
# across all layers simultaneously.  batch_size=16 + grad_accum=2 to fit
# in 44 GiB (HKD stores per-layer hidden states for all 12 layers of both
# teacher and student).
source "$(dirname "$0")/../_common.sh"
run_train teacher=gpt2_small student=gpt2_small_lolcats distiller=hkd loss=mse data=dolma \
    'distiller.alignment.module_pattern="transformer\.h\.(\d+)\.attn$"' \
    training.learning_rate=1e-2 training.batch_size=16 +training.gradient_accumulation_steps=2 \
    training.max_steps=76000 "$@"
