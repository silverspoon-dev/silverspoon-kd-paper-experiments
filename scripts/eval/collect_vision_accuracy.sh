#!/bin/bash
# Collect CIFAR image-classification accuracy from trainer_state.json for
# every matching run directory and write one JSON per run under
# ``results/vision/``.  This is the post-training eval for the vision
# experiments: we already compute eval_accuracy during training via
# compute_metrics, so no extra forward-pass eval is needed — we just read
# the best checkpoint's eval_accuracy out of the trainer_state and stash
# it in a tidy results file for tables.
#
# Usage:
#   DEVICES=0 bash scripts/eval/collect_vision_accuracy.sh
#   PATTERN='*vgg16*' bash scripts/eval/collect_vision_accuracy.sh  # subset
#
# Produces: results/vision/<run_name>.json with
#   {"run": ..., "best_eval_accuracy": ..., "best_epoch": ..., "global_step": ...}
source "$(dirname "$0")/../_common.sh"

RESULTS_DIR="$PROJECT_DIR/results/vision"
mkdir -p "$RESULTS_DIR"

PATTERN="${PATTERN:-*cifar*}"

python3 << PY
import json, glob, os

runs_dir = "$RUNS_DIR"
results_dir = "$RESULTS_DIR"
pattern = "$PATTERN"

written = 0
for run_dir in sorted(glob.glob(os.path.join(runs_dir, pattern))):
    if not os.path.isdir(run_dir):
        continue
    run = os.path.basename(run_dir)
    # Find the latest checkpoint dir (highest step number).
    ckpts = sorted(
        glob.glob(os.path.join(run_dir, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else 0,
    )
    if not ckpts:
        print(f"SKIP (no checkpoints): {run}")
        continue
    ts_path = os.path.join(ckpts[-1], "trainer_state.json")
    if not os.path.exists(ts_path):
        print(f"SKIP (no trainer_state): {run}")
        continue
    ts = json.load(open(ts_path))
    log = ts.get("log_history", [])
    evals = [e for e in log if "eval_accuracy" in e]
    if not evals:
        print(f"SKIP (no eval_accuracy): {run}")
        continue
    best = max(evals, key=lambda e: e["eval_accuracy"])
    out = {
        "run": run,
        "best_eval_accuracy": best["eval_accuracy"],
        "best_epoch": best.get("epoch"),
        "best_step": best.get("step"),
        "final_global_step": ts.get("global_step"),
        "max_steps": ts.get("max_steps"),
        "num_evals": len(evals),
    }
    out_path = os.path.join(results_dir, run + ".json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    written += 1
    print(f"  {best['eval_accuracy']*100:6.2f}%  {run}")

print(f"\nWrote {written} result files to {results_dir}")
PY
