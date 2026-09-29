#!/bin/bash
# Cross-architecture BKD: GPT-2 Small (MHA) → DeepSeek-V3-arch 96M (MLA) on Dolma
source "$(dirname "$0")/../_common.sh"
run_train teacher=gpt2_small student=deepseek_v3_96M distiller=bkd loss=mse data=dolma \
    training.learning_rate=1e-3 training.batch_size=16 \
    training.max_steps=20000 "$@"
