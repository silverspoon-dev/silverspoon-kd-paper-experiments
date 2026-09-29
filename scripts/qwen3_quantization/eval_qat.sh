#!/bin/bash
# QAT evaluation: the PTQ floor and every QAT run for the given bit widths.
#
# Usage:
#   bash eval_qat.sh [BITS...]           # eval all methods for given bit widths (default: 4 6 8)
#   METHODS="ptq baseline" bash eval_qat.sh 4 8   # only PTQ + baseline for INT4/INT8
source "$(dirname "$0")/../_common.sh"

BITS="${@:-4 6 8}"
METHODS="${METHODS:-ptq baseline bkd hkd reskd}"

for bits in $BITS; do
    tag="int${bits}"
    for method in $METHODS; do
        case "$method" in
            ptq)
                name="ptq_${tag}"
                model="Qwen/Qwen3-1.7B"
                tokenizer=""
                ;;
            baseline)
                name="qat_${tag}_baseline"
                model="$RUNS_DIR/hf__standard__qwen3_1.7B_${tag}__scratch/model"
                tokenizer="--tokenizer Qwen/Qwen3-1.7B"
                ;;
            bkd)
                name="qat_${tag}_bkd"
                model="$RUNS_DIR/silverspoon-kd__bkd__qwen3_1.7B__qwen3_1.7B_${tag}__scratch/student_model"
                tokenizer="--tokenizer Qwen/Qwen3-1.7B"
                ;;
            hkd)
                name="qat_${tag}_hkd"
                model="$RUNS_DIR/silverspoon-kd__hkd__qwen3_1.7B__qwen3_1.7B_${tag}__scratch/student_model"
                tokenizer="--tokenizer Qwen/Qwen3-1.7B"
                ;;
            reskd)
                name="qat_${tag}_reskd"
                model="$RUNS_DIR/silverspoon-kd__reskd__qwen3_1.7B__qwen3_1.7B_${tag}__scratch/student_model"
                tokenizer="--tokenizer Qwen/Qwen3-1.7B"
                ;;
            *) echo "Unknown method: $method"; continue ;;
        esac

        output="$RUNS_DIR/eval_results/${name}.json"
        if [ -f "$output" ]; then
            echo "SKIP (exists): $name"
            continue
        fi
        if [ ! -d "$model" ] && [ "$method" != "ptq" ]; then
            echo "SKIP (no model): $name"
            continue
        fi

        echo "=== Eval: $name ==="
        cmd="CUDA_VISIBLE_DEVICES=$DEVICES python $PROJECT_DIR/evaluate_quantized.py \
            --model $model $tokenizer \
            --quant $tag \
            --output $output"
        echo ">>> $cmd"
        [ "$DRY_RUN" = "true" ] && continue
        eval "$cmd"
    done
done
