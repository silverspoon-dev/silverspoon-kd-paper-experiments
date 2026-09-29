#!/bin/bash
# HKD: BERT-base → BERT student on Dolma
source "$(dirname "$0")/../_common.sh"
STUDENT="${STUDENT:-bert_small_uncased}"
run_train teacher=bert_base_uncased student=$STUDENT \
    distiller=hkd data=dolma data.max_length=512 \
    '++student.reconfig.freeze_copied_weights=false' \
    run.dtype=float32 training.bf16=false training.max_steps=20000 "$@"
