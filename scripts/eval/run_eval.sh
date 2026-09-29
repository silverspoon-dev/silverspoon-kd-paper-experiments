#!/bin/bash
# Evaluate a trained model with lm-evaluation-harness.
# Usage: ./run_eval.sh <run_name> [tasks] [extra_args]
# Example: DEVICES=0 ./run_eval.sh silverspoon-kd__bkd__qwen3_0.6B__qwen3_420M__scratch
source "$(dirname "$0")/../_common.sh"
[ $# -lt 1 ] && { echo "Usage: $0 <run_name> [tasks] [extra_args]"; exit 1; }
run_eval "$@"
