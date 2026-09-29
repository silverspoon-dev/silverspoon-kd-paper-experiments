#!/bin/bash
# Cross-architecture BKD → HKD: GPT-2 Small → DeepSeek-V3-arch 96M on Dolma (stage 2)
source "$(dirname "$0")/../_common.sh"
PARENT="silverspoon-kd__bkd__gpt2_small__deepseek_v3_96M__mse__scratch"
run_train teacher=gpt2_small student=deepseek_v3_96M distiller=hkd loss=mse data=dolma \
    $(from_run "$PARENT") \
    training.learning_rate=1e-3 training.batch_size=16 \
    training.max_steps=20000 "$@"
