#!/bin/bash
# Evaluate all GPT-2 linearization checkpoints with scale-appropriate benchmarks.
#
# Same metric rationale as gpt2_distillation/eval_all.sh: wikitext perplexity,
# LAMBADA, and HellaSwag are meaningful at ~100M scale.
#
# Usage:
#   DEVICES=0 bash scripts/gpt2_linearization/eval_all.sh
#   DRY_RUN=true bash scripts/gpt2_linearization/eval_all.sh
source "$(dirname "$0")/../_common.sh"

TASKS="${EVAL_TASKS:-wikitext,lambada_openai,hellaswag}"

eval_model() {
    local run_dir="$1"
    local run_name
    run_name=$(basename "$run_dir")

    # Three model save layouts (same as gpt2_distillation/eval_all.sh):
    #   - silverspoon-kd distillers (BKD/HKD/ReSKD): student_model/
    #   - HF Trainer baseline:                       model/
    #   - LoRA training:                             lora_model/ (PEFT adapter)
    local model_args=""
    if [ -d "$run_dir/student_model" ]; then
        model_args="pretrained=$run_dir/student_model,tokenizer=openai-community/gpt2"
    elif [ -d "$run_dir/model" ]; then
        model_args="pretrained=$run_dir/model,tokenizer=openai-community/gpt2"
    elif [ -d "$run_dir/lora_model" ]; then
        # LoRA: PEFT adapter on top of the parent experiment's student model.
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

    # --batch_size 16 (fixed) rather than auto: auto-detection probes with
    # 0-length inputs that crash GPT-2-family models at input_ids.view().
    # Explicit tokenizer because the saved student_model directories do
    # not always carry tokenizer files.
    echo ">>> CUDA_VISIBLE_DEVICES=$DEVICES lm_eval --model hf --model_args $model_args --tasks $TASKS --batch_size 16 --output_path $output_dir"
    [ "$DRY_RUN" = "true" ] && return 0
    # Use Python wrapper to register LoLCATs model class before lm_eval
    # loads the model. Without this, AutoModelForCausalLM falls back to
    # GPT2LMHeadModel and silently discards feature map weights.
    CUDA_VISIBLE_DEVICES=$DEVICES python3 -c "
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath('$0')) + '/../..')
from compiled.lolcats_gpt2 import LoLCATsGPT2LMHeadModel
from transformers import AutoModelForCausalLM, GPT2Config
AutoModelForCausalLM.register(GPT2Config, LoLCATsGPT2LMHeadModel, exist_ok=True)
import lm_eval
results = lm_eval.simple_evaluate(
    model='hf',
    model_args='$model_args',
    tasks='$TASKS'.split(','),
    batch_size=16,
    log_samples=False,
)
from lm_eval.loggers import EvaluationTracker
tracker = EvaluationTracker(output_path='$output_dir')
tracker.save_results_aggregated(results=results['results'], samples=results.get('samples',{}))
"
}

# Evaluate all LoLCATs runs
for dir in "$RUNS_DIR"/*__gpt2_small_lolcats__*; do
    [ -d "$dir" ] || continue
    echo "=== Evaluating: $(basename "$dir") ==="
    eval_model "$dir"
done
