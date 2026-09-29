#!/bin/bash
# BKD: GPT-2 → LoLCATs hybrid on Dolma (attention-level alignment)
#
# Aligns linearized attention outputs with teacher's softmax attention,
# block-by-block.  This is stage-1 only — follow with lolcats_bkd_hkd.sh,
# lolcats_bkd_reskd.sh, or lolcats_bkd_ft.sh for a usable model.
source "$(dirname "$0")/../_common.sh"
run_train teacher=gpt2_small student=gpt2_small_lolcats distiller=bkd loss=mse data=dolma \
    'distiller.alignment.module_pattern="transformer\.h\.(\d+)\.attn$"' \
    training.learning_rate=1e-2 training.batch_size=32 training.max_steps=76000 "$@"
