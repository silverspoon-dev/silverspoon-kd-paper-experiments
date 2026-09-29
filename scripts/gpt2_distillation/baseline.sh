#!/bin/bash
# Baseline: GPT-2 96M trained from scratch on Dolma (no distillation)
source "$(dirname "$0")/../_common.sh"
run_train student=gpt2_96M distiller=standard data=dolma \
    training.learning_rate=1e-3 training.batch_size=32 \
    training.max_steps=60000 "$@"
