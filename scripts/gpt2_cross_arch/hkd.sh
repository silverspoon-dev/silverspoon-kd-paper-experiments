#!/bin/bash
# Cross-architecture HKD: GPT-2 Small (MHA) → DeepSeek-V3-arch 96M (MLA) on Dolma
#
# ── Why response_loss_weight is bumped to 30 ─────────────────────────────
# The default in trainers.py is ``len(alignments) / 3`` which for 6 student
# layers works out to 2.0, so the 6 feature-level MSE alignments
# (sum of weights = 6) dominate the single response-KL alignment 3:1.
# Cross-arch distillation into a fundamentally different attention kernel
# (MHA → MLA) makes that feature-MSE constraint *actively harmful*: it
# forces the student's hidden states into a GPT-2-shaped representation
# that the DeepSeek-V3 architecture cannot naturally produce.  The
# earlier same-setup run hit wikitext ppl 277 vs a plain-training baseline
# of 175 — HKD made the model 1.58× worse.
#
# Setting ``response_loss_weight=30`` inverts the balance: the response
# KL on logits has 5× the weight of the sum of feature alignments, so the
# student optimizes primarily for output-distribution matching (which
# *is* architecture-agnostic) while feature alignment becomes a
# secondary regulariser.  Pair with hkd_ft.sh for a stage-2 CE fine-tune
# that lets the student shake off any remaining over-constraints.
source "$(dirname "$0")/../_common.sh"
run_train teacher=gpt2_small student=deepseek_v3_96M distiller=hkd loss=mse data=dolma \
    +distiller.response_loss_weight=30 \
    +distiller.temperature=4.0 \
    training.learning_rate=1e-3 training.batch_size=32 \
    training.max_steps=60000 "$@"
