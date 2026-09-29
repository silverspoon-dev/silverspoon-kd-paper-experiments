#!/bin/bash
# Holistic distillation (from scratch): GPT-2 Small → GPT-2 112M on Dolma
source "$(dirname "$0")/../_common.sh"
run_train teacher=gpt2_small student=gpt2_112M distiller=hkd loss=mse data=dolma \
    training.learning_rate=1e-3 training.batch_size=16 \
    training.max_steps=117000 "$@"
