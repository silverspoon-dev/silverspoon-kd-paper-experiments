#!/bin/bash
# ── Smoke Test ──────────────────────────────────────────────────────────────
# Validate all experiment pipelines locally on macOS (32GB RAM).
#
# Runs each experiment for 2 training steps with batch_size=1 to verify:
#   - Hydra config resolution & model loading
#   - Dataset streaming / preparation
#   - Forward + backward passes (no NaN, no crashes)
#   - Checkpoint saving & loading (for chained experiments)
#
# Memory: ~8-15 GB peak per experiment (well within 32GB unified memory).
# Every experiment group is covered, including the LoLCATs linearization
# (its generated architecture ships in compiled/).
#
# Usage:
#   bash scripts/smoke_test.sh                      # run all experiments
#   bash scripts/smoke_test.sh --clean               # remove prior smoke outputs first
#   bash scripts/smoke_test.sh --clean-only           # just clean, don't run
#   bash scripts/smoke_test.sh --category qwen3      # single category
#
# After a successful smoke test, clean up before cluster submission:
#   bash scripts/smoke_test.sh --clean-only
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNS_DIR="$PROJECT_DIR/runs"

# ── Parse arguments ─────────────────────────────────────────────────────────
CLEAN=false
CLEAN_ONLY=false
CATEGORY=""
shift_next=false
for arg in "$@"; do
    case $arg in
        --clean) CLEAN=true ;;
        --clean-only) CLEAN_ONLY=true ;;
        --category=*) CATEGORY="${arg#*=}" ;;
        --category) shift_next=true ;;
        *)
            if [ "$shift_next" = "true" ]; then
                CATEGORY="$arg"
                shift_next=false
            fi
            ;;
    esac
done

# ── Cleanup function ───────────────────────────────────────────────────────
clean_smoke_runs() {
    echo "Cleaning smoke test outputs from runs/ ..."
    if [ -d "$RUNS_DIR" ]; then
        local count=0
        for dir in "$RUNS_DIR"/*/; do
            [ -d "$dir" ] || continue
            # Smoke test runs have at most checkpoint-10 (max_steps=10)
            local max_ckpt=0
            for ckpt in "${dir}"checkpoint-*/; do
                [ -d "$ckpt" ] || continue
                local step="${ckpt%/}"
                step="${step##*checkpoint-}"
                [ "$step" -gt "$max_ckpt" ] 2>/dev/null && max_ckpt="$step"
            done
            if [ "$max_ckpt" -gt 0 ] && [ "$max_ckpt" -le 10 ]; then
                echo "  rm -rf $(basename "$dir")"
                rm -rf "$dir"
                count=$((count + 1))
            fi
        done
        echo "Removed $count smoke test run(s)."
    fi
}

if [ "$CLEAN" = "true" ] || [ "$CLEAN_ONLY" = "true" ]; then
    clean_smoke_runs
    [ "$CLEAN_ONLY" = "true" ] && exit 0
fi

# ── Mac-compatible overrides ───────────────────────────────────────────────
# These override experiment-specific values via "$@" passthrough in each script.
# Hydra last-wins rule ensures these take precedence.
SMOKE_ARGS=(
    training.max_steps=10
    training.batch_size=4
    training.num_epochs=1
    training.eval_on_start=false
    training.eval_strategy=no
    training.save_strategy=no
    training.torch_compile=false
    training.attn_implementation=sdpa
    training.report_to=none
    training.dataloader_num_workers=0
    training.dataloader_pin_memory=false
    training.optim=adamw_torch
    training.gradient_checkpointing=false
    training.early_stopping_patience=0
    training.disable_tqdm=false
    training.log_level=info
    training.disable_console_logs=false
)

# ── State tracking ─────────────────────────────────────────────────────────
PASS=0
FAIL=0
SKIP=0
declare -a RESULTS=()

smoke_run() {
    # Usage: smoke_run "label" command [args...]
    # Append SMOKE_ARGS and SMOKE_EXTRA (if set) to the command.
    # SMOKE_EXTRA overrides come last (Hydra last-wins).
    local label="$1"; shift
    echo ""
    echo "================================================================"
    echo "  SMOKE: $label"
    echo "================================================================"
    if "$@" "${SMOKE_ARGS[@]}" ${SMOKE_EXTRA:-}; then
        echo "  >> PASS: $label"
        PASS=$((PASS + 1))
        RESULTS+=("PASS  $label")
    else
        echo "  >> FAIL: $label"
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL  $label")
    fi
}

smoke_run_raw() {
    # Like smoke_run but without appending SMOKE_ARGS (for non-Hydra commands).
    local label="$1"; shift
    echo ""
    echo "================================================================"
    echo "  SMOKE: $label"
    echo "================================================================"
    if "$@"; then
        echo "  >> PASS: $label"
        PASS=$((PASS + 1))
        RESULTS+=("PASS  $label")
    else
        echo "  >> FAIL: $label"
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL  $label")
    fi
}

smoke_skip() {
    local label="$1"
    local reason="$2"
    echo ""
    echo "  >> SKIP: $label ($reason)"
    SKIP=$((SKIP + 1))
    RESULTS+=("SKIP  $label -- $reason")
}

should_run() {
    [ -z "$CATEGORY" ] && return 0
    [[ "$1" == *"$CATEGORY"* ]] && return 0
    return 1
}

# ── 5. GPT-2 Compression ──────────────────────────────────────────────────
if should_run "gpt2"; then
    # BKD needs to save a checkpoint for the BKD->HKD->LoRA chain
    SMOKE_EXTRA="training.save_strategy=steps training.save_steps=10 training.save_total_limit=1" \
        smoke_run "gpt2: BKD (small->96M)"        bash "$SCRIPT_DIR/gpt2_distillation/bkd.sh"
    smoke_run "gpt2: HKD (small->96M)"         bash "$SCRIPT_DIR/gpt2_distillation/hkd.sh"
    smoke_run "gpt2: ResKD (small->96M)"       bash "$SCRIPT_DIR/gpt2_distillation/reskd.sh"
    smoke_run "gpt2: baseline (96M)"           bash "$SCRIPT_DIR/gpt2_distillation/baseline.sh"
    SMOKE_EXTRA="training.save_strategy=steps training.save_steps=10 training.save_total_limit=1" \
        smoke_run "gpt2: BKD->HKD chain"           bash "$SCRIPT_DIR/gpt2_distillation/bkd_hkd.sh"
    smoke_run "gpt2: BKD->HKD->LoRA chain"    bash "$SCRIPT_DIR/gpt2_distillation/bkd_hkd_lora.sh"
fi

# ── 2. GPT-2 Cross-Architecture (MHA → MLA/DeepSeek-V3) ──────────────────
# BKD is excluded: layer interfaces differ between GPT-2 (MHA) and DeepSeek-V3
# (MLA), so feeding teacher layer inputs to student layers is incompatible.
# HKD/ResKD work because they align full model outputs, not layer-level inputs.
if should_run "gpt2_cross_arch"; then
    smoke_run "cross-arch: HKD (GPT2->DeepSeek)"   bash "$SCRIPT_DIR/gpt2_cross_arch/hkd.sh"
    smoke_run "cross-arch: ResKD (GPT2->DeepSeek)"  bash "$SCRIPT_DIR/gpt2_cross_arch/reskd.sh"
    smoke_run "cross-arch: baseline (DeepSeek)"     bash "$SCRIPT_DIR/gpt2_cross_arch/baseline.sh"
fi

# ── 3. BERT Pre-Training ──────────────────────────────────────────────────
if should_run "bert_distillation"; then
    # Test both students: T6 (mild compression) and T4-tiny (aggressive)
    STUDENT=bert_T6 smoke_run "bert: BKD (base->T6)"           bash "$SCRIPT_DIR/bert_distillation/bkd.sh"
    STUDENT=bert_T6 smoke_run "bert: HKD (base->T6)"           bash "$SCRIPT_DIR/bert_distillation/hkd.sh"
    STUDENT=bert_T6 smoke_run "bert: ResKD (base->T6)"         bash "$SCRIPT_DIR/bert_distillation/reskd.sh"
    STUDENT=bert_T6 smoke_run "bert: baseline (T6)"             bash "$SCRIPT_DIR/bert_distillation/baseline.sh"
    STUDENT=bert_T4_tiny smoke_run "bert: BKD (base->T4_tiny)" bash "$SCRIPT_DIR/bert_distillation/bkd.sh"
fi

# ── 4. BERT Downstream ────────────────────────────────────────────────────
if should_run "bert_downstream"; then
    smoke_run "bert-downstream: finetune teachers"  bash "$SCRIPT_DIR/bert_downstream/finetune_teachers.sh"
    # TextBrewer standalone test (max_steps=10 to avoid full 30-epoch training)
    smoke_run_raw "bert-downstream: TextBrewer (T6, MNLI)" \
        python "$SCRIPT_DIR/bert_downstream/run_textbrewer.py" --student T6 --max_steps 10
fi

# ── 5. Qwen3 Quantization-Aware Training (1.7B, INT4) ─────────────────────
# INT4 only — INT8 QAT is nearly lossless; INT4 is where KD has the most impact.
if should_run "quantization"; then
    smoke_run "qat: int4 BKD (1.7B)"      bash "$SCRIPT_DIR/qwen3_quantization/int4_bkd.sh"
    smoke_run "qat: int4 HKD (1.7B)"      bash "$SCRIPT_DIR/qwen3_quantization/int4_hkd.sh"
    # ResKD uses liger-kernel for efficient cross-entropy (CUDA-only)
    if python -c "import liger_kernel" 2>/dev/null; then
        smoke_run "qat: int4 ResKD (1.7B)"    bash "$SCRIPT_DIR/qwen3_quantization/int4_reskd.sh"
    else
        smoke_skip "qat: int4 ResKD (1.7B)" "liger-kernel not installed (CUDA-only)"
    fi
    smoke_run "qat: int4 baseline (1.7B)" bash "$SCRIPT_DIR/qwen3_quantization/int4_baseline.sh"
fi

# ── 6. GPT-2 Linearized Attention (LoLCATs) ───────────────────────────────
if should_run "linearization"; then
    if [ -f "$PROJECT_DIR/compiled/lolcats_gpt2.py" ]; then
        smoke_run "lolcats: BKD"       bash "$SCRIPT_DIR/gpt2_linearization/lolcats_bkd.sh"
        smoke_run "lolcats: HKD"       bash "$SCRIPT_DIR/gpt2_linearization/lolcats_hkd.sh"
        smoke_run "lolcats: ResKD"     bash "$SCRIPT_DIR/gpt2_linearization/lolcats_reskd.sh"
        smoke_run "lolcats: baseline"  bash "$SCRIPT_DIR/gpt2_linearization/lolcats_baseline.sh"
    else
        smoke_skip "lolcats: all" "compiled/lolcats_gpt2.py missing"
    fi
fi

# ── 1. VGG Compression (CIFAR-10, VGG16) ──────────────────────────────────
if should_run "vision_bkd"; then
    smoke_run "vision-bkd: teacher (VGG16)"   bash "$SCRIPT_DIR/vision_bkd/teacher.sh"
    smoke_run "vision-bkd: BKD (VGG16->DS)"   bash "$SCRIPT_DIR/vision_bkd/bkd.sh"
    smoke_run "vision-bkd: ResKD"              bash "$SCRIPT_DIR/vision_bkd/reskd.sh"
    smoke_run "vision-bkd: baseline"           bash "$SCRIPT_DIR/vision_bkd/baseline.sh"
fi

# ── 2. VGG Relational KD (CIFAR-100, ResNet50) ────────────────────────────
# ResNet50 uses BatchNorm which requires batch_size >= 2.
if should_run "vision_relkd"; then
    SMOKE_EXTRA="training.batch_size=2" \
        smoke_run "vision-relkd: teacher (ResNet50)"     bash "$SCRIPT_DIR/vision_relkd/teacher.sh"
    smoke_run "vision-relkd: ResKD"                       bash "$SCRIPT_DIR/vision_relkd/reskd.sh"
    smoke_run "vision-relkd: HKD+MSE"                     bash "$SCRIPT_DIR/vision_relkd/hkd_mse.sh"
    smoke_run "vision-relkd: HKD+RelKD angle"             bash "$SCRIPT_DIR/vision_relkd/hkd_relkd_angle.sh"
    SMOKE_EXTRA="training.batch_size=2" \
        smoke_run "vision-relkd: baseline"                 bash "$SCRIPT_DIR/vision_relkd/baseline.sh"
fi

# ── 9. Framework Benchmarking (import checks) ─────────────────────────────
if should_run "framework_comparison"; then
    smoke_run_raw "framework: textbrewer imports"   python -c "from textbrewer import GeneralDistiller; print('OK')"
    smoke_run_raw "framework: torchdistill imports" python -c "from torchdistill.losses.mid_level import KDLoss; print('OK')"
    smoke_run_raw "framework: torchtune imports"    python -c "from torchtune.modules.loss import ForwardKLLoss; print('OK')"
fi

# ── Report ──────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  SMOKE TEST RESULTS: $PASS passed, $FAIL failed, $SKIP skipped"
echo "================================================================"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "Fix failures above before submitting to the cluster."
    echo "Re-run individual categories with: bash scripts/smoke_test.sh --category <name>"
    exit 1
else
    echo "All pipelines validated. Clean up before cluster submission:"
    echo "  bash scripts/smoke_test.sh --clean-only"
    exit 0
fi
