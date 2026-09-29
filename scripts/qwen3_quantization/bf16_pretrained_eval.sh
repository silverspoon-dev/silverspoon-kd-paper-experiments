#!/bin/bash
# Eval-only: Qwen3-1.7B pretrained at bfloat16 (reference ceiling for QAT experiments)
source "$(dirname "$0")/../_common.sh"

TASKS="${1:-hellaswag,arc_easy,arc_challenge,winogrande,mmlu,truthfulqa_mc2,wikitext}"
EXTRA="${2:-}"
MODEL="Qwen/Qwen3-1.7B"

cmd="CUDA_VISIBLE_DEVICES=$DEVICES lm_eval --model hf --model_args pretrained=$MODEL,dtype=bfloat16 --tasks $TASKS --batch_size auto $EXTRA"
echo ">>> $cmd"
[ "$DRY_RUN" = "true" ] && exit 0
eval "$cmd"
