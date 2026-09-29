#!/bin/bash
# Stage 2 — Plain CE fine-tune from the HKD stage-1 checkpoint.
# Cross-arch HKD forces student hidden states to mimic GPT-2's intermediate
# representations, which the DeepSeek-V3 architecture cannot naturally
# produce.  Stage 2 runs standard causal-LM training (no teacher) on the
# same Dolma data so the student can optimise freely for the actual task
# objective, starting from the HKD-initialised weights.
#
# Evaluation compares hf__standard__deepseek_v3_96M__from-hkd.* (this run)
# against hf__standard__deepseek_v3_96M__scratch (plain baseline).  If HKD
# provides useful initialisation, this run's wikitext ppl should beat
# baseline's; if HKD was actively harmful, this run should land in the
# same ballpark as baseline (since the CE objective can undo the damage).
source "$(dirname "$0")/../_common.sh"
PARENT="silverspoon-kd__hkd__gpt2_small__deepseek_v3_96M__scratch"
run_train student=deepseek_v3_96M distiller=standard data=dolma \
    $(from_run "$PARENT") \
    training.learning_rate=5e-4 training.batch_size=32 \
    training.max_steps=60000 "$@"
