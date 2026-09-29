#!/bin/bash
# Baseline: BERT student trained from scratch on Dolma (no distillation)
source "$(dirname "$0")/../_common.sh"
STUDENT="${STUDENT:-bert_small_uncased}"
run_train student=$STUDENT distiller=standard data=dolma data.max_length=512 \
    run.dtype=float32 training.bf16=false training.max_steps=20000 \
    '++student.reconfig.freeze_copied_weights=false' "$@"
