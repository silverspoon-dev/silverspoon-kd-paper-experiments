#!/bin/bash
# Shared helpers for all experiment scripts.
# Source this at the top of every script:
#   source "$(dirname "$0")/../_common.sh"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUNS_DIR="$PROJECT_DIR/runs"

DEVICES="${DEVICES:-0}"
DRY_RUN="${DRY_RUN:-false}"
export WANDB_PROJECT="${WANDB_PROJECT:-silverspoon-kd-paper-experiments}"

run_train() {
    # Usage: run_train teacher=X student=Y distiller=Z [overrides...] "$@"
    # Skip if run already completed (model directory exists).
    #
    # IMPORTANT: we use "$@" rather than `eval "$cmd"` with `$*` so that
    # override values containing shell metacharacters (e.g. regex patterns with
    # parentheses like `transformer\.h\.(\d+)\.attn$`) pass through unmodified.
    # `eval` would interpret the `(...)` as a subshell and crash with a syntax
    # error.  The echo'd command is only for logging and does not round-trip.
    local name
    name=$(CUDA_VISIBLE_DEVICES="" python "$PROJECT_DIR/train.py" "$@" run.name_only=true 2>/dev/null || true)
    if [ -n "$name" ] && [ -d "$RUNS_DIR/$name/model" -o -d "$RUNS_DIR/$name/student_model" ]; then
        echo ">>> SKIP (done): $name"
        return 0
    fi
    echo ">>> CUDA_VISIBLE_DEVICES=$DEVICES python $PROJECT_DIR/train.py $*"
    [ "$DRY_RUN" = "true" ] && return 0
    CUDA_VISIBLE_DEVICES=$DEVICES python "$PROJECT_DIR/train.py" "$@"
}

from_run() {
    # Usage: from_run "silverspoon__bkd__qwen3_0.6B__qwen3_420M__scratch"
    # Returns: init.method=from_experiment init.from_experiment=<name> init.checkpoint_step=latest
    local parent="$1"
    echo "init.method=from_experiment init.from_experiment=$parent init.checkpoint_step=latest"
}

run_eval() {
    # Usage: run_eval <run_name> [tasks] [extra_args]
    # --batch_size 16 fixed rather than auto (see eval_all.sh notes).
    local run_name="$1"
    local tasks="${2:-hellaswag,arc_easy,arc_challenge,winogrande,mmlu,truthfulqa_mc2,wikitext}"
    local extra="${3:-}"
    local model_path="$RUNS_DIR/$run_name/student_model"
    echo ">>> CUDA_VISIBLE_DEVICES=$DEVICES lm_eval --model hf --model_args pretrained=$model_path --tasks $tasks --batch_size 16 $extra"
    [ "$DRY_RUN" = "true" ] && return 0
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$DEVICES lm_eval --model hf \
        --model_args "pretrained=$model_path" \
        --tasks "$tasks" --batch_size 16 $extra
}
