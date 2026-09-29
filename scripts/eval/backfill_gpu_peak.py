#!/usr/bin/env python3
"""One-time backfill: fetch peak GPU memory from WandB system events.

Queries only runs that are missing gpu_peak_memory.json locally AND
missing peak_gpu_mb in the WandB cache.  Writes results to both places.

Usage:
    python scripts/eval/backfill_gpu_peak.py                  # dry-run
    python scripts/eval/backfill_gpu_peak.py --write           # write results
    python scripts/eval/backfill_gpu_peak.py --runs-dir /path  # custom runs dir
"""

import argparse
import json
import os
import sys
from pathlib import Path


def find_project_root():
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "configs").is_dir():
            return p
        p = p.parent
    return Path.cwd()


def main():
    parser = argparse.ArgumentParser(description="Backfill peak GPU memory from WandB")
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--write", action="store_true", help="Actually write results (default: dry-run)")
    args = parser.parse_args()

    runs_dir = args.runs_dir or str(find_project_root() / "runs")
    cache_file = os.path.join(os.path.dirname(__file__), ".wandb_cache.json")

    # Load existing cache
    cache = {}
    if os.path.isfile(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)
        print(f"Loaded cache: {len(cache)} runs")

    # Find runs missing peak GPU data
    missing = []
    for entry in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, entry)
        if not os.path.isdir(run_dir) or entry.startswith(("_", ".", "eval_results")):
            continue
        # Skip if local file exists
        if os.path.isfile(os.path.join(run_dir, "gpu_peak_memory.json")):
            continue
        # Skip if cache already has it
        if entry in cache and "peak_gpu_mb" in cache[entry]:
            continue
        missing.append(entry)

    print(f"Runs missing peak GPU: {len(missing)}")
    if not missing:
        print("Nothing to backfill.")
        return

    for name in missing:
        print(f"  {name}")

    if not args.write:
        print("\nDry-run mode. Pass --write to fetch and save.")
        return

    # Fetch from WandB
    try:
        import wandb
    except ImportError:
        print("wandb not installed", file=sys.stderr)
        sys.exit(1)

    api = wandb.Api()
    project = os.environ.get(
        "WANDB_PROJECT_PATH",
        "<wandb-entity>/silverspoon-kd-paper-experiments",
    )

    try:
        all_runs = {r.name: r for r in api.runs(project, filters={"state": "finished"}, per_page=200)}
    except Exception as e:
        print(f"WandB API error: {e}", file=sys.stderr)
        sys.exit(1)

    found = 0
    for i, name in enumerate(missing):
        run = all_runs.get(name)
        if run is None:
            print(f"  [{i+1}/{len(missing)}] {name}: not found in WandB")
            continue

        try:
            sys_metrics = run.history(stream="events", samples=500)
            mem_cols = [c for c in sys_metrics.columns if "memoryAllocatedBytes" in c]
            if not mem_cols:
                print(f"  [{i+1}/{len(missing)}] {name}: no GPU memory columns")
                continue
            peak_bytes = max(sys_metrics[c].max() for c in mem_cols)
            if not peak_bytes or peak_bytes <= 0:
                print(f"  [{i+1}/{len(missing)}] {name}: zero/null GPU memory")
                continue
            peak_mib = round(peak_bytes / (1024**2))
            peak_gib = round(peak_bytes / (1024**3), 3)
            print(f"  [{i+1}/{len(missing)}] {name}: {peak_mib} MiB ({peak_gib} GiB)")
            found += 1

            # Write local gpu_peak_memory.json
            run_dir = os.path.join(runs_dir, name)
            mem_path = os.path.join(run_dir, "gpu_peak_memory.json")
            if os.path.isdir(run_dir):
                with open(mem_path, "w") as f:
                    json.dump({"peak_memory_mib": peak_mib, "peak_memory_gib": peak_gib,
                               "source": "wandb_backfill"}, f)

            # Update cache
            if name not in cache:
                cache[name] = {}
            cache[name]["peak_gpu_mb"] = peak_mib

        except Exception as e:
            print(f"  [{i+1}/{len(missing)}] {name}: error: {e}")

    # Save updated cache
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"\nBackfilled {found}/{len(missing)} runs. Cache saved to {cache_file}")


if __name__ == "__main__":
    main()
