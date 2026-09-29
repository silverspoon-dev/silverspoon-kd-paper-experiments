#!/bin/bash
# Evaluate all GPT-2 distillation checkpoints with scale-appropriate benchmarks.
#
# GPT-2-class models (~100M params) score near-random on MMLU/ARC-Challenge,
# so we use perplexity-centric metrics following the original GPT-2 evaluation:
#   - wikitext (perplexity)
#   - lambada_openai (accuracy + perplexity)
#   - hellaswag (weak but above-random signal at this scale)
#
# Usage:
#   DEVICES=0 bash scripts/gpt2_distillation/eval_all.sh
#   EVAL_TASKS=wikitext bash scripts/gpt2_distillation/eval_all.sh   # override tasks
#   DRY_RUN=true bash scripts/gpt2_distillation/eval_all.sh          # preview
source "$(dirname "$0")/../_common.sh"

TASKS="${EVAL_TASKS:-wikitext,lambada_openai,hellaswag}"

eval_model() {
    local run_dir="$1"
    local run_name
    run_name=$(basename "$run_dir")

    # Three model save layouts:
    #   - silverspoon-kd distillers (BKD/HKD): student_model/
    #   - HF Trainer baseline:                 model/
    #   - LoRA training:                       lora_model/  (PEFT adapter weights only)
    local model_args=""
    if [ -d "$run_dir/student_model" ]; then
        model_args="pretrained=$run_dir/student_model,tokenizer=openai-community/gpt2"
    elif [ -d "$run_dir/model" ]; then
        model_args="pretrained=$run_dir/model,tokenizer=openai-community/gpt2"
    elif [ -d "$run_dir/lora_model" ]; then
        # LoRA: PEFT adapter on top of the parent experiment's student model.
        # The adapter_config.json may point to the original HF model name
        # (e.g. openai-community/gpt2) rather than the reconfigured student,
        # so we resolve the actual base model from the parent run directory.
        local parent_run
        parent_run=$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('$run_dir/config.yaml'))
print(cfg.get('init', {}).get('from_experiment', ''))
" 2>/dev/null)
        local base_model=""
        if [ -n "$parent_run" ] && [ -d "$RUNS_DIR/$parent_run/student_model" ]; then
            base_model="$RUNS_DIR/$parent_run/student_model"
        elif [ -n "$parent_run" ] && [ -d "$RUNS_DIR/$parent_run/model" ]; then
            base_model="$RUNS_DIR/$parent_run/model"
        else
            echo ">>> SKIP (no base model for LoRA): $run_name"; return 0
        fi
        model_args="pretrained=$base_model,peft=$run_dir/lora_model,tokenizer=openai-community/gpt2"
    else
        echo ">>> SKIP (no model): $run_name"; return 0
    fi

    local output_dir="$RUNS_DIR/eval_results/${run_name}"
    if [ -d "$output_dir" ]; then
        echo ">>> SKIP (already evaluated): $run_name"; return 0
    fi

    # Use explicit tokenizer to avoid batch_size=auto crash when student_model
    # directory lacks tokenizer files (GPT-2 pad token issue).
    #
    # --batch_size 16 (fixed) rather than auto: auto batch-size detection
    # probes progressively larger batches and has been observed to feed a
    # 0-length input, which crashes GPT-2's forward at
    # `input_ids.view(-1, input_shape[-1])` with shape `[-1, 0]`.
    echo ">>> CUDA_VISIBLE_DEVICES=$DEVICES lm_eval --model hf --model_args $model_args --tasks $TASKS --batch_size 16 --output_path $output_dir"
    [ "$DRY_RUN" = "true" ] && return 0
    CUDA_VISIBLE_DEVICES=$DEVICES lm_eval --model hf \
        --model_args "$model_args" \
        --tasks "$TASKS" --batch_size 16 \
        --output_path "$output_dir"
}

# Evaluate all runs with gpt2_112M student
for dir in "$RUNS_DIR"/*__gpt2_112M__*; do
    [ -d "$dir" ] || continue
    echo "=== Evaluating: $(basename "$dir") ==="
    eval_model "$dir"
done
