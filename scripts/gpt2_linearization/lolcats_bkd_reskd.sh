#!/bin/bash
# BKD → ReSKD: GPT-2 → LoLCATs hybrid on Dolma (stage 2: logit refinement)
#
# Stage-2 after BKD attention alignment.  ReSKD refines the full model
# via logit-level distillation (no alignment pattern needed).
source "$(dirname "$0")/../_common.sh"
PARENT="silverspoon-kd__bkd__gpt2_small__gpt2_small_lolcats__scratch"
run_train teacher=gpt2_small student=gpt2_small_lolcats distiller=reskd data=dolma \
    $(from_run "$PARENT") training.learning_rate=1e-3 training.batch_size=16 \
    +training.gradient_accumulation_steps=2 training.max_steps=76000 "$@"
