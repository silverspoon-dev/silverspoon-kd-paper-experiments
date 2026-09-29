#!/bin/bash
# Baseline: LoLCATs hybrid GPT-2, teacher-init + CE fine-tune (no distillation)
source "$(dirname "$0")/../_common.sh"
run_train student=gpt2_small_lolcats distiller=standard data=dolma \
    training.learning_rate=1e-3 training.batch_size=32 training.max_steps=76000 "$@"
