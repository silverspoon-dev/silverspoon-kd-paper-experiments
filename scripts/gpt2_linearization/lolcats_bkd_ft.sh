#!/bin/bash
# BKD → FT: GPT-2 → LoLCATs hybrid on Dolma (stage 2: CE fine-tune only)
#
# Stage-2 after BKD attention alignment.  Tests whether BKD alignment
# alone is sufficient — no teacher needed, just CE language modeling.
source "$(dirname "$0")/../_common.sh"
PARENT="silverspoon-kd__bkd__gpt2_small__gpt2_small_lolcats__scratch"
run_train student=gpt2_small_lolcats distiller=standard data=dolma \
    $(from_run "$PARENT") training.learning_rate=1e-3 training.batch_size=32 \
    training.max_steps=76000 "$@"
