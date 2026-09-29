#!/bin/bash
# Run all BERT encoder experiments: 2 students × 4 distillers = 8 experiments
source "$(dirname "$0")/../_common.sh"

STUDENTS=(bert_T6 bert_T4_tiny)
SCRIPTS=(baseline bkd reskd hkd)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for student in "${STUDENTS[@]}"; do
    for script in "${SCRIPTS[@]}"; do
        echo "=== STUDENT=$student  SCRIPT=$script ==="
        STUDENT="$student" bash "$SCRIPT_DIR/$script.sh" "$@"
    done
done
