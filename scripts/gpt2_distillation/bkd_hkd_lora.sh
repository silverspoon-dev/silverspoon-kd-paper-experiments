#!/bin/bash
# BKD → HKD → LoRA: GPT-2 Small → GPT-2 96M on Dolma (3-stage pipeline)
# Requires bkd_hkd.sh to have been run first.
source "$(dirname "$0")/../_common.sh"
# Find the HKD run name (includes hash of parent)
HKD_PARENT=$(ls -d "$RUNS_DIR"/silverspoon-kd__hkd__gpt2_small__gpt2_96M__from-bkd.* 2>/dev/null | head -1 | xargs -r basename || true)
[ -z "$HKD_PARENT" ] && { echo "ERROR: Run bkd_hkd.sh first"; exit 1; }
run_train student=gpt2_96M distiller=lora_gpt2 data=dolma \
    $(from_run "$HKD_PARENT") \
    training.learning_rate=1e-3 training.batch_size=32 \
    training.max_steps=60000 "$@"
