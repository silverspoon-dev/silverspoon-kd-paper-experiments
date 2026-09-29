#!/bin/bash
# Blockwise distillation: GPT-2 Small → GPT-2 96M on Dolma
source "$(dirname "$0")/../_common.sh"
run_train teacher=gpt2_small student=gpt2_96M distiller=bkd loss=mse data=dolma \
    training.learning_rate=1e-3 training.batch_size=32 \
    training.max_steps=60000 "$@"
