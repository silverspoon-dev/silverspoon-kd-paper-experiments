#!/bin/bash
# Evaluate all MNLI head-to-head models and produce a comparison table.
#
# Usage:
#   DEVICES=0 bash scripts/bert_downstream/evaluate_downstream.sh
#   DRY_RUN=true bash scripts/bert_downstream/evaluate_downstream.sh
source "$(dirname "$0")/../_common.sh"

RESULTS_DIR="$PROJECT_DIR/results/downstream"
mkdir -p "$RESULTS_DIR"

_ensure_tokenizer() {
    # silverspoon-kd saves student_model without tokenizer files.
    # Copy from the nearest checkpoint in the same run directory.
    local model_dir="$1"
    if [ -f "$model_dir/tokenizer.json" ] || [ -f "$model_dir/vocab.txt" ]; then
        return 0
    fi
    local run_dir
    run_dir="$(dirname "$model_dir")"
    for ckpt in "$run_dir"/checkpoint-*/tokenizer.json "$run_dir"/checkpoint-*/vocab.txt; do
        if [ -f "$ckpt" ]; then
            local ckpt_dir
            ckpt_dir="$(dirname "$ckpt")"
            echo "  Copying tokenizer from $ckpt_dir → $model_dir"
            cp "$ckpt_dir"/tokenizer*.json "$model_dir/" 2>/dev/null || true
            cp "$ckpt_dir"/vocab.txt "$model_dir/" 2>/dev/null || true
            cp "$ckpt_dir"/special_tokens_map.json "$model_dir/" 2>/dev/null || true
            return 0
        fi
    done
    echo "  WARNING: no tokenizer found for $model_dir"
}

_write_hkd_result_from_trainer_state() {
    # silverspoon-kd's reconfig saves attention weights with teacher dims
    # ([768, 312]) under a config that says hidden_size=312. Both the saved
    # student_model/ AND the Trainer checkpoint-* dirs share this issue:
    # from_pretrained(..., ignore_mismatched_sizes=True) re-initializes the
    # mismatched layers and reports ~33% accuracy on a model that actually
    # trained to 80%+. The in-training eval ran on the live model with
    # correct shapes, so its best_metric is the ground truth.
    #
    # Read it from any checkpoint's trainer_state.json and write a JSON in
    # the format evaluate.py would have produced. Only matched accuracy is
    # available from training-time eval; mismatched is reported as None.
    local run_dir="$1"
    local output_json="$2"
    local any_state
    any_state="$(ls -t "$run_dir"/checkpoint-*/trainer_state.json 2>/dev/null | head -n1 || true)"
    if [ -z "$any_state" ] || [ ! -f "$any_state" ]; then
        echo "  SKIP: no trainer_state.json under $run_dir"
        return 0
    fi
    echo "  Writing HKD result from $any_state"
    python3 - "$any_state" "$output_json" <<'PY' || true
import json, sys
state_path, out_path = sys.argv[1], sys.argv[2]
d = json.load(open(state_path))
acc = d.get("best_metric")
if acc is None:
    # Fall back to the most recent eval_accuracy in log_history.
    for entry in reversed(d.get("log_history", [])):
        if "eval_accuracy" in entry:
            acc = entry["eval_accuracy"]
            break
if acc is None:
    print(f"  ERROR: no eval_accuracy in {state_path}")
    sys.exit(0)
result = {
    "accuracy": acc,
    "split": "validation_matched",
    "n_samples": 9815,
    "accuracy_matched": acc,
    "accuracy_mismatched": None,
    "model_path": d.get("best_model_checkpoint", state_path.rsplit("/", 1)[0]),
    "task": "mnli",
    "_source": "trainer_state.best_metric",
}
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"  Wrote {acc*100:.2f}% (matched) to {out_path}")
PY
}

_resolve_sk_hkd_model_dir() {
    # silverspoon-kd's saved student_model/ has shape-mismatched attention
    # weights for reconfig'd students: weights are written with teacher dims
    # ([768, 312]) while config says hidden_size=312, so from_pretrained
    # re-initializes them to random → ~33% accuracy.
    # The Trainer's checkpoint-* dirs contain the correct full state, so
    # prefer best_model_checkpoint; fall back to the latest checkpoint if
    # the best one was pruned by save_total_limit.
    local run_dir="$1"
    local latest_ckpt
    latest_ckpt="$(ls -dt "$run_dir"/checkpoint-* 2>/dev/null | head -n1 || true)"
    if [ -z "$latest_ckpt" ] || [ ! -d "$latest_ckpt" ]; then
        # No checkpoints at all → student_model is our only option.
        echo "$run_dir/student_model"
        return 0
    fi
    local best_ckpt=""
    if [ -f "$latest_ckpt/trainer_state.json" ]; then
        best_ckpt="$(python3 - "$latest_ckpt/trainer_state.json" "$run_dir" <<'PY' || true
import json, os, sys
state_path, run_dir = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(state_path))
    p = d.get("best_model_checkpoint")
    if p:
        candidate = os.path.join(run_dir, os.path.basename(p))
        if os.path.isdir(candidate):
            print(candidate)
except Exception:
    pass
PY
)"
    fi
    if [ -n "$best_ckpt" ] && [ -d "$best_ckpt" ]; then
        echo "$best_ckpt"
    else
        # Best was pruned (or not yet recorded) — use latest available checkpoint.
        echo "$latest_ckpt"
    fi
}

run_eval_downstream() {
    local model_path="$1"
    local task="$2"
    local output_json="$3"

    if [ ! -d "$model_path" ]; then
        echo "  SKIP: $model_path not found"
        return 0
    fi

    _ensure_tokenizer "$model_path"

    local cmd="CUDA_VISIBLE_DEVICES=$DEVICES python $PROJECT_DIR/evaluate.py \
        --model_path $model_path --task $task --output_json $output_json"
    echo ">>> $cmd"
    [ "$DRY_RUN" = "true" ] && return 0
    eval "$cmd"
}

STUDENTS=(bert_T6 bert_T4_tiny)

# ── Evaluate teacher ──────────────────────────────────────────────────────
echo "=== Evaluating Teacher ==="
run_eval_downstream "$RUNS_DIR/hf__standard__bert_base_cased__mnli__scratch/model" \
    mnli "$RESULTS_DIR/teacher_mnli.json"

# ── Evaluate silverspoon-kd models ────────────────────────────────────────
echo ""
echo "=== Evaluating silverspoon-kd models ==="
DATA_SHORT="mnli"

for student in "${STUDENTS[@]}"; do
    echo "  --- Student: $student ---"

    # Baseline
    baseline_run="hf__standard__${student}__${DATA_SHORT}__scratch"
    run_eval_downstream "$RUNS_DIR/$baseline_run/model" \
        mnli "$RESULTS_DIR/${student}_${DATA_SHORT}_baseline.json"

    # HKD — single-stage (silverspoon-kd__hkd__*__${student}__mnli__scratch)
    # or two-stage (hf__standard__${student}__mnli__from-hkd.*).
    # For single-stage, the saved checkpoint can't be reloaded standalone
    # for reconfig'd students (see _write_hkd_result_from_trainer_state),
    # so read the in-training best eval directly.
    hkd_found=false
    for dir in "$RUNS_DIR"/silverspoon-kd__hkd__*__${student}__${DATA_SHORT}__scratch; do
        if [ -d "$dir" ] && ls -d "$dir"/checkpoint-* &>/dev/null; then
            _write_hkd_result_from_trainer_state "$dir" \
                "$RESULTS_DIR/${student}_${DATA_SHORT}_hkd.json"
            hkd_found=true
            break
        fi
    done
    if [ "$hkd_found" = false ]; then
        for dir in "$RUNS_DIR"/hf__standard__${student}__${DATA_SHORT}__from-hkd.*; do
            if [ -d "$dir/model" ]; then
                run_eval_downstream "$dir/model" \
                    mnli "$RESULTS_DIR/${student}_${DATA_SHORT}_hkd.json"
                break
            fi
        done
    fi

    # BKD (stage 2 model)
    for dir in "$RUNS_DIR"/hf__standard__${student}__${DATA_SHORT}__from-bkd.*; do
        if [ -d "$dir/model" ]; then
            run_eval_downstream "$dir/model" \
                mnli "$RESULTS_DIR/${student}_${DATA_SHORT}_bkd.json"
            break
        fi
    done
done

# ── Evaluate TextBrewer models ────────────────────────────────────────────
echo ""
echo "=== Evaluating TextBrewer models ==="
for student in T6 T4-tiny; do
    student_slug="${student//-/_}"
    tb_model="$RUNS_DIR/textbrewer__bert_${student}__mnli/model"
    run_eval_downstream "$tb_model" \
        mnli "$RESULTS_DIR/textbrewer_bert_${student_slug}_mnli.json"
done

# ── Print comparison table ────────────────────────────────────────────────
echo ""
echo "=== Results Summary ==="
echo ""

if [ "$DRY_RUN" != "true" ]; then
    python3 -c "
import json, glob, os

results_dir = '$RESULTS_DIR'

def load(name):
    path = os.path.join(results_dir, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

# Teacher
t = load('teacher_mnli.json')
if t:
    m = t.get('accuracy_matched', t.get('accuracy', 0))
    mm = t.get('accuracy_mismatched', 0)
    print(f'Teacher (BERT-base-cased):  MNLI-m/mm = {m*100:.1f}/{mm*100:.1f}')
    print()

# Comparison table
print(f'{\"Student\":<12} {\"Method\":<25} {\"MNLI-m\":<10} {\"MNLI-mm\":<10}')
print('-' * 57)

students = [('bert_T6', 'T6'), ('bert_T4_tiny', 'T4-tiny')]
for cfg_name, display in students:
    slug = display.replace('-', '_')

    # Baseline
    r = load(f'{cfg_name}_mnli_baseline.json')
    if r:
        m = r.get('accuracy_matched', r.get('accuracy', 0)) * 100
        mm = r.get('accuracy_mismatched', 0) * 100
        print(f'{display:<12} {\"Baseline (no teacher)\":<25} {m:<10.1f} {mm:<10.1f}')
    else:
        print(f'{display:<12} {\"Baseline (no teacher)\":<25} {\"---\":<10} {\"---\":<10}')

    # TextBrewer
    r = load(f'textbrewer_bert_{slug}_mnli.json')
    if r:
        m = r.get('accuracy_matched', r.get('accuracy', 0)) * 100
        mm = r.get('accuracy_mismatched', 0) * 100
        print(f'{\"\":<12} {\"TextBrewer (HKD+NST)\":<25} {m:<10.1f} {mm:<10.1f}')
    else:
        print(f'{\"\":<12} {\"TextBrewer (HKD+NST)\":<25} {\"---\":<10} {\"---\":<10}')

    # silverspoon-kd HKD
    r = load(f'{cfg_name}_mnli_hkd.json')
    if r:
        m = r.get('accuracy_matched', r.get('accuracy', 0)) * 100
        mm = r.get('accuracy_mismatched', 0) * 100
        print(f'{\"\":<12} {\"silverspoon-kd HKD\":<25} {m:<10.1f} {mm:<10.1f}')
    else:
        print(f'{\"\":<12} {\"silverspoon-kd HKD\":<25} {\"---\":<10} {\"---\":<10}')

    # silverspoon-kd BKD
    r = load(f'{cfg_name}_mnli_bkd.json')
    if r:
        m = r.get('accuracy_matched', r.get('accuracy', 0)) * 100
        mm = r.get('accuracy_mismatched', 0) * 100
        print(f'{\"\":<12} {\"silverspoon-kd BKD\":<25} {m:<10.1f} {mm:<10.1f}')
    else:
        print(f'{\"\":<12} {\"silverspoon-kd BKD\":<25} {\"---\":<10} {\"---\":<10}')

    print()
"
fi

echo "=== Evaluation complete. Results in $RESULTS_DIR ==="
