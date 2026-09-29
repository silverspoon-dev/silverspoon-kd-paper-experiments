#!/bin/bash
#SBATCH --job-name=bench-distill
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

source "$SLURM_SUBMIT_DIR/slurm/_common.sh"
setup_env

# Quick benchmark: 200 steps each, QAT Qwen3 1.7B (where memory matters).
# Uses experiment.name override to avoid collisions with real runs.
# Captures full output (no grep) so errors are visible.
BENCH_STEPS=200

run_bench() {
    local label="$1"; shift
    echo ""
    echo "========================================"
    echo "  Benchmark: $label"
    echo "========================================"
    CUDA_VISIBLE_DEVICES=0 python train.py \
        teacher=qwen3_1.7B student=qwen3_1.7B_pretrained data=dolma \
        run.student_dtype=int4 student.short_name=qwen3_1.7B_int4 \
        training.learning_rate=2e-5 training.lr_scheduler_type=cosine \
        training.optim=adamw_8bit \
        training.eval_strategy=no training.save_strategy=no \
        training.early_stopping_patience=0 \
        training.report_to=none training.disable_tqdm=false \
        training.max_steps=$BENCH_STEPS \
        "experiment.name=bench_${label}" \
        "$@" 2>&1
    echo "--- $label done ---"
}

run_bench "bkd_standard" \
    distiller=bkd loss=mse training.batch_size=4

run_bench "bkd_backward_per_block" \
    distiller=bkd loss=mse training.batch_size=4 \
    +distiller.backward_per_block=true

run_bench "hkd" \
    distiller=hkd loss=mse training.batch_size=2 \
    training.gradient_checkpointing=true \
    +training.per_device_eval_batch_size=1

run_bench "reskd" \
    distiller=reskd \
    distiller.alpha=0 distiller.temperature=1.0 \
    +distiller.use_liger_kernel=true +distiller.output_head_layer=lm_head \
    training.batch_size=2 training.gradient_checkpointing=true

echo ""
echo "=== All benchmarks complete ==="
echo ""
echo "--- Summary ---"
for d in runs/bench_*; do
    [ -d "$d" ] || continue
    label=$(basename "$d")
    mem_file="$d/gpu_peak_memory.json"
    if [ -f "$mem_file" ]; then
        echo "$label: $(cat "$mem_file")"
    else
        echo "$label: no gpu_peak_memory.json"
    fi
done
