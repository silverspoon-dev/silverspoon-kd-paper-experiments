#!/bin/bash
# BKD → HKD: GPT-2 Small → GPT-2 112M on Dolma (stage 2: holistic refinement)
source "$(dirname "$0")/../_common.sh"
PARENT="silverspoon-kd__bkd__gpt2_small__gpt2_112M__scratch"
run_train teacher=gpt2_small student=gpt2_112M distiller=hkd loss=mse data=dolma \
    $(from_run "$PARENT") \
    training.learning_rate=1e-3 training.batch_size=16 \
    training.max_steps=117000 "$@"
