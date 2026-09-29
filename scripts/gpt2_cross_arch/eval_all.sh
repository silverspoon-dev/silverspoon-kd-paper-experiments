#!/bin/bash
# Evaluate all GPT-2 → DeepSeek-V3 cross-architecture distillation checkpoints.
# Same benchmarks as gpt2_distillation/eval_all.sh for direct comparison.
source "$(dirname "$0")/../_common.sh"

TASKS="${EVAL_TASKS:-wikitext,lambada_openai,hellaswag}"

eval_model() {
    local run_dir="$1"
    local run_name
    run_name=$(basename "$run_dir")

    local model_path=""
    if [ -d "$run_dir/student_model" ]; then
        model_path="$run_dir/student_model"
    elif [ -d "$run_dir/model" ]; then
        model_path="$run_dir/model"
    else
        echo ">>> SKIP (no model): $run_name"; return 0
    fi

    local output_dir="$RUNS_DIR/eval_results/${run_name}"
    if [ -d "$output_dir" ]; then
        echo ">>> SKIP (already evaluated): $run_name"; return 0
    fi

    # DeepSeek student uses GPT-2 tokenizer (set in config); specify explicitly
    # to avoid tokenizer instantiation issues in lm_eval.
    #
    # --batch_size 16 (fixed) rather than auto: auto probes with 0-length
    # batches that crash GPT-2-family models at the input_ids.view() call.
    echo ">>> CUDA_VISIBLE_DEVICES=$DEVICES lm_eval --model hf --model_args pretrained=$model_path,tokenizer=openai-community/gpt2 --tasks $TASKS --batch_size 16 --output_path $output_dir"
    [ "$DRY_RUN" = "true" ] && return 0
    CUDA_VISIBLE_DEVICES=$DEVICES lm_eval --model hf \
        --model_args "pretrained=$model_path,tokenizer=openai-community/gpt2" \
        --tasks "$TASKS" --batch_size 16 \
        --output_path "$output_dir"
}

for dir in "$RUNS_DIR"/*__deepseek_v3_96M__*; do
    [ -d "$dir" ] || continue
    echo "=== Evaluating: $(basename "$dir") ==="
    eval_model "$dir"
done
