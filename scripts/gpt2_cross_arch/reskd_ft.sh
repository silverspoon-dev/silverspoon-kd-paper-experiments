#!/bin/bash
# Stage 2 — Plain CE fine-tune from the ReSKD stage-1 checkpoint.
# Symmetric to hkd_ft.sh — lets the ReSKD-pretrained student continue
# with pure CE training so we can compare the 2-stage pipelines on
# equal footing (same total step budget, same final-stage objective).
source "$(dirname "$0")/../_common.sh"
PARENT="silverspoon-kd__reskd__gpt2_small__deepseek_v3_96M__scratch"
run_train student=deepseek_v3_96M distiller=standard data=dolma \
    $(from_run "$PARENT") \
    training.learning_rate=5e-4 training.batch_size=32 \
    training.max_steps=60000 "$@"
