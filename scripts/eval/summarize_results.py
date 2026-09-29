#!/usr/bin/env python3
"""Generate paper-ready result tables from experiment outputs.

Reads trainer_state.json, eval_results JSON, and downstream result files,
then prints both a terminal summary and LaTeX-ready tables.

Usage:
    python scripts/eval/summarize_results.py                    # all experiments
    python scripts/eval/summarize_results.py --group qat        # single group
    python scripts/eval/summarize_results.py --latex             # emit LaTeX tables
    python scripts/eval/summarize_results.py --runs-dir /path   # custom runs dir
"""

import argparse
import json
import glob
import os
import re
import sys
from collections import defaultdict, OrderedDict
from pathlib import Path

# WandB project path for GPU memory queries (--memory flag).
# Set WANDB_PROJECT_PATH=<entity>/<project> to point at your own logs.
WANDB_PROJECT = os.environ.get(
    "WANDB_PROJECT_PATH",
    "<wandb-entity>/silverspoon-kd-paper-experiments",
)


# ── Known model parameter counts ─────────────────────────────────────────────
# Hardcoded from model cards / config.json.  Used when we cannot read the
# actual config from the run directory.

KNOWN_PARAM_COUNTS = {
    # LLM teachers
    "qwen3_1.7B":       1_700_000_000,
    "gpt2_small":         124_000_000,
    "bert_base_uncased":  110_000_000,
    "bert_base_cased":    110_000_000,
    # LLM students
    "qwen3_1.7B_int4":  1_700_000_000,  # same arch, quantised weights
    "qwen3_1.7B_int8":  1_700_000_000,
    "gpt2_112M":          112_000_000,
    "gpt2_96M":            96_000_000,
    "deepseek_v3_96M":     96_000_000,
    "gpt2_small_lolcats": 125_000_000,  # GPT-2 124M + ~1M learned feature map params
    # BERT students
    "bert_T6":             67_000_000,
    "bert_T4_tiny":        14_000_000,
    "bert_small_uncased":  29_000_000,
    "bert_T12_nano":       11_000_000,
    # Vision
    "resnet50_cifar100":   25_000_000,
    "vgg11bn_cifar100":     9_000_000,  # feature extractor only
    "vgg16_cifar10":      138_000_000,
    "vgg16_cifar10_teacher": 138_000_000,
    "vgg16_cifar10_depthwise_separable": 3_000_000,
}

# Teacher eval result directory/file names (in eval_results/).
# Used to add a "Teacher" reference row to each table.
# Values are (dir_or_file, label) tuples.
TEACHER_EVAL = {
    "gpt2_compression": ("gpt2_teacher", "Teacher (GPT-2 124M)"),
    "cross_arch":       ("gpt2_teacher", "Teacher (GPT-2 124M)"),
    "linearization":    ("gpt2_teacher", "Teacher (GPT-2 124M)"),
    "qat":              ("qwen3_1.7B_teacher", "Teacher (Qwen3-1.7B BF16)"),
}

# Map experiment group key → (teacher_model_key, student_model_key)
GROUP_MODEL_KEYS = {
    "qat":              ("qwen3_1.7B",       "qwen3_1.7B_int4"),
    "gpt2_compression": ("gpt2_small",        "gpt2_96M"),
    "cross_arch":       ("gpt2_small",        "deepseek_v3_96M"),
    "linearization":    ("gpt2_small",        "gpt2_small_lolcats"),
    "bert_downstream":  ("bert_base_cased",   "bert_T6"),
    "vgg_relkd":        ("resnet50_cifar100",  "vgg11bn_cifar100"),
    "vgg_compression":  ("vgg16_cifar10",      "vgg16_cifar10_depthwise_separable"),
}


def format_params(n):
    """Format a parameter count as a human-readable string (e.g. '96M', '1.7B')."""
    if n is None:
        return None
    if n >= 1_000_000_000:
        v = n / 1_000_000_000
        return f"{v:.1f}B" if v != int(v) else f"{int(v)}B"
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.0f}M" if v == int(v) else f"{v:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def inject_teacher_row(rows, group_key, runs_dir):
    """Prepend a 'Teacher' reference row to *rows* if teacher eval results exist."""
    teacher_info = TEACHER_EVAL.get(group_key)
    if teacher_info is None:
        return
    dir_name, label = teacher_info
    eval_dir = os.path.join(runs_dir, "eval_results", dir_name)
    r = read_lm_eval_results(eval_dir)
    if r is None:
        # Also try as a flat JSON file (e.g. qat eval format)
        json_path = eval_dir + ".json"
        if os.path.isfile(json_path):
            data = read_json(json_path)
            r_nested = data.get("results", data)
            r = {}
            for task, metrics in r_nested.items():
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        r[f"{task}/{k}"] = v
        if not r:
            return
    teacher_row = {
        "name": label,
        "run_name": dir_name,
        # LM eval metrics (GPT-2, Qwen3)
        "wikitext_ppl": r.get("wikitext/word_perplexity,none"),
        "lambada_acc": r.get("lambada_openai/acc,none"),
        "hellaswag": r.get("hellaswag/acc_norm,none"),
        # QAT metrics (Qwen3)
        "mmlu": r.get("mmlu/acc,none"),
        "arc_easy": r.get("arc_easy/acc,none"),
        "arc_challenge": r.get("arc_challenge/acc_norm,none"),
    }
    rows.insert(0, teacher_row)


def format_title_with_models(title, group_key):
    """Append teacher → student model sizes to the table title."""
    keys = GROUP_MODEL_KEYS.get(group_key)
    if keys is None:
        return title
    teacher_key, student_key = keys
    teacher_n = format_params(KNOWN_PARAM_COUNTS.get(teacher_key))
    student_n = format_params(KNOWN_PARAM_COUNTS.get(student_key))
    if teacher_n and student_n:
        return f"{title}  [{teacher_n} → {student_n}]"
    return title


def inject_trainer_state_columns(rows, runs_dir):
    """Add columns from trainer_state.json for each run.

    Adds:
    - final_eval_loss: best distillation alignment loss (teacher vs student).
      Teacher rows get 0.0, baselines get None.
    - train_steps: "current/total" training step progress.
    """
    for r in rows:
        run_name = r.get("run_name", "")
        if not run_name:
            r["final_eval_loss"] = None
            r["train_steps"] = None
            continue
        # Teacher row
        if "teacher" in run_name.lower() or "Teacher" in r.get("name", ""):
            r["final_eval_loss"] = 0.0
            r["train_steps"] = None
            continue
        # Baseline (no distillation) — still want training steps
        is_baseline = "standard" in run_name or "baseline" in r.get("name", "").lower() or "ptq" in run_name.lower()

        run_dir = os.path.join(runs_dir, run_name)
        ts_path = latest_trainer_state(run_dir)
        if ts_path is None:
            r["final_eval_loss"] = None
            r["train_steps"] = None
            continue
        try:
            ts = read_json(ts_path)
            # Training steps
            global_step = ts.get("global_step")
            max_steps = ts.get("max_steps")
            if global_step is not None and max_steps is not None and max_steps > 0:
                # If the final model was saved, the run completed — show max/max
                # even if the last checkpoint was earlier.
                final_model = (os.path.isdir(os.path.join(run_dir, "student_model"))
                               or os.path.isdir(os.path.join(run_dir, "model")))
                if final_model and global_step < max_steps:
                    r["train_steps"] = f"{max_steps}/{max_steps}"
                else:
                    r["train_steps"] = f"{global_step}/{max_steps}"
            else:
                r["train_steps"] = None

            # Final eval loss (skip for baselines)
            if is_baseline:
                r["final_eval_loss"] = None
            else:
                evals = [e for e in ts.get("log_history", [])
                         if "eval_loss" in e and "eval_runtime" in e]
                if evals:
                    r["final_eval_loss"] = min(e["eval_loss"] for e in evals)
                else:
                    r["final_eval_loss"] = None
        except Exception:
            r["final_eval_loss"] = None
            r["train_steps"] = None


# ── GPU memory from WandB ────────────────────────────────────────────────────

_wandb_cache = None  # run_name -> {peak_gpu_mb, steps_per_sec, runtime_h}


_WANDB_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".wandb_cache.json")


def load_wandb_cache():
    """Query WandB API for throughput and runtime of all finished runs.

    Results are cached to ``.wandb_cache.json`` next to this script.
    Subsequent runs load from cache unless ``--wandb-refresh`` is passed.
    Peak GPU memory comes from local ``gpu_peak_memory.json`` files
    (much faster than querying WandB system events per run).
    """
    global _wandb_cache
    if _wandb_cache is not None:
        return _wandb_cache

    # Try loading from cache file first
    if os.path.isfile(_WANDB_CACHE_FILE):
        try:
            _wandb_cache = read_json(_WANDB_CACHE_FILE)
            print(f"  (loaded WandB cache: {len(_wandb_cache)} runs from {_WANDB_CACHE_FILE})",
                  file=sys.stderr)
            return _wandb_cache
        except Exception:
            pass

    _wandb_cache = _refresh_wandb_cache()
    return _wandb_cache


def _refresh_wandb_cache(runs_dir=None):
    """Fetch fresh data from WandB API and save to cache file.

    Collects throughput and runtime from run.summary (fast), then
    queries per-run system events for peak GPU memory (slower — one
    extra API call per run).

    When *runs_dir* is provided, only runs whose name matches a
    directory in runs_dir are fetched (skips archived/crashed runs).
    """
    global _wandb_cache
    try:
        import wandb
    except ImportError:
        print("  (wandb not installed — skipping --wandb)", file=sys.stderr)
        _wandb_cache = {}
        return _wandb_cache

    # Build set of local run names for filtering
    local_runs = set()
    if runs_dir and os.path.isdir(runs_dir):
        local_runs = {d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d)) and not d.startswith(("_", "."))}

    api = wandb.Api()
    _wandb_cache = {}
    try:
        runs = list(api.runs(WANDB_PROJECT, filters={"state": "finished"}, per_page=200))
    except Exception as e:
        print(f"  (wandb API error: {e} — skipping --wandb)", file=sys.stderr)
        return _wandb_cache

    # Filter to local runs only (if runs_dir provided)
    if local_runs:
        runs = [r for r in runs if r.name in local_runs]

    n_runs = len(runs)
    for i, run in enumerate(runs):
        try:
            entry = {}
            # Throughput + runtime from summary (fast, no extra API call)
            s = run.summary
            if s.get("train_steps_per_second"):
                entry["steps_per_sec"] = round(s["train_steps_per_second"], 2)
            if s.get("train_samples_per_second"):
                entry["samples_per_sec"] = round(s["train_samples_per_second"], 1)
            if s.get("train_runtime"):
                entry["runtime_h"] = round(s["train_runtime"] / 3600, 2)

            # Peak GPU memory from system events stream
            try:
                sys_metrics = run.history(stream="events", samples=500)
                mem_cols = [c for c in sys_metrics.columns
                            if "memoryAllocatedBytes" in c]
                if mem_cols:
                    peak_bytes = max(sys_metrics[c].max() for c in mem_cols)
                    if peak_bytes and peak_bytes > 0:
                        entry["peak_gpu_mb"] = round(peak_bytes / (1024**2))
            except Exception:
                pass

            if entry:
                _wandb_cache[run.name] = entry
        except Exception:
            pass

        if (i + 1) % 25 == 0 or i == n_runs - 1:
            print(f"\r  (fetching WandB data: {i + 1}/{n_runs} runs...)",
                  end="", file=sys.stderr)

    print(file=sys.stderr)  # newline after progress

    # Save cache
    try:
        with open(_WANDB_CACHE_FILE, "w") as f:
            json.dump(_wandb_cache, f, indent=2)
    except Exception:
        pass

    print(f"  (fetched WandB metrics for {len(_wandb_cache)} runs, cached to {_WANDB_CACHE_FILE})",
          file=sys.stderr)
    return _wandb_cache


def _lookup_wandb(run_name):
    """Look up WandB metrics for a run name (exact or partial match).

    Handles the common mismatch between eval_results dir names (which
    are the full experiment name) and WandB display names (which are
    the same).  Also handles shortened names produced by the collector
    functions (e.g. ``bkd__scratch`` matching
    ``silverspoon-kd__bkd__gpt2_small__gpt2_96M__scratch``).
    """
    if _wandb_cache is None:
        return {}
    # Exact match
    if run_name in _wandb_cache:
        return _wandb_cache[run_name]
    # Partial: the shortened display name is a substring of the full WandB name
    # Pick the LONGEST matching cached name (most specific) to avoid
    # e.g. "bkd" matching "bkd_attn".
    best_match = None
    best_len = 0
    for cached_name, entry in _wandb_cache.items():
        # Check both directions: short in long, long in short
        if run_name in cached_name or cached_name in run_name:
            if len(cached_name) > best_len:
                best_match = entry
                best_len = len(cached_name)
        # Also try stripping common prefixes for collector-shortened names
        stripped = cached_name.replace("silverspoon-kd__", "").replace("hf__standard__", "").replace("hf__lora__", "")
        if run_name in stripped or stripped in run_name:
            if len(cached_name) > best_len:
                best_match = entry
                best_len = len(cached_name)
    return best_match or {}


def _read_gpu_peak_json(run_name, runs_dir=None):
    """Read gpu_peak_memory.json from a run directory (fallback when WandB has no data)."""
    if runs_dir is None:
        return None
    for candidate in [run_name, f"textbrewer__{run_name}"]:
        path = os.path.join(runs_dir, candidate, "gpu_peak_memory.json")
        if os.path.isfile(path):
            try:
                import json
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
    return None


def get_peak_gpu_mb(run_name, runs_dir=None):
    """Look up peak GPU memory (MB) for a run."""
    val = _lookup_wandb(run_name).get("peak_gpu_mb")
    if val is None and runs_dir:
        data = _read_gpu_peak_json(run_name, runs_dir)
        if data and "peak_memory_mib" in data:
            val = round(data["peak_memory_mib"])
    return val


def get_steps_per_sec(run_name):
    """Look up training throughput (steps/sec) for a run."""
    return _lookup_wandb(run_name).get("steps_per_sec")


def get_runtime_h(run_name):
    """Look up total training runtime (hours) for a run."""
    return _lookup_wandb(run_name).get("runtime_h")


def find_project_root():
    """Walk up from this script to find the project root (has configs/ dir)."""
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "configs").is_dir():
            return p
        p = p.parent
    return Path.cwd()


# ── Helpers ──────────────────────────────────────────────────────────────────

def read_json(path):
    with open(path) as f:
        return json.load(f)


def best_eval_metric(trainer_state_path, metric="eval_accuracy"):
    """Extract the best eval metric from a trainer_state.json log_history."""
    ts = read_json(trainer_state_path)
    log = ts.get("log_history", [])
    evals = [e for e in log if metric in e]
    if not evals:
        return None
    best = max(evals, key=lambda e: e[metric])
    return {
        "value": best[metric],
        "epoch": best.get("epoch"),
        "step": best.get("step"),
        "global_step": ts.get("global_step"),
        "max_steps": ts.get("max_steps"),
        "n_evals": len(evals),
    }


def vision_eval_accuracy(run_name, runs_dir):
    """Check results/vision/<run_name>.json for post-hoc evaluated accuracy."""
    results_dir = os.path.join(os.path.dirname(runs_dir), "results", "vision")
    result_path = os.path.join(results_dir, f"{run_name}.json")
    if os.path.isfile(result_path):
        data = read_json(result_path)
        return data.get("best_eval_accuracy")
    return None


def latest_trainer_state(run_dir):
    """Find the latest checkpoint's trainer_state.json in a run directory."""
    ckpts = sorted(
        glob.glob(os.path.join(run_dir, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else 0,
    )
    if not ckpts:
        return None
    ts_path = os.path.join(ckpts[-1], "trainer_state.json")
    return ts_path if os.path.exists(ts_path) else None


def read_lm_eval_results(eval_dir):
    """Read lm-eval results JSON from an eval_results subdirectory.

    Uses recursive glob to handle results stored in any subdirectory
    (e.g. ``None/results_*.json`` from LoLCATs eval runs).
    """
    jsons = glob.glob(os.path.join(eval_dir, "**/results*.json"), recursive=True)
    if not jsons:
        return None
    data = read_json(jsons[0])
    # lm-eval results may be nested under "results" key or at top level
    results = data.get("results", None)
    if results is None:
        # Flat format: tasks at top level (e.g. LoLCATs eval via simple_evaluate)
        results = {k: v for k, v in data.items() if isinstance(v, dict) and "alias" in v}
    out = {}
    for task, metrics in results.items():
        if task.startswith("mmlu_"):
            continue  # skip mmlu subtasks
        for k, v in metrics.items():
            if v is not None and not k.endswith("_stderr,none"):
                out[f"{task}/{k}"] = v
    return out


# ── Group collectors ─────────────────────────────────────────────────────────

def collect_qat(runs_dir):
    """QAT experiments across bit widths (Qwen3-1.7B)."""
    eval_dir = os.path.join(runs_dir, "eval_results")
    rows = []
    # Sweep bit widths from most to least aggressive quantization
    for bits in [4, 5, 6, 7, 8]:
        tag = f"int{bits}"
        entries = [
            (f"ptq_{tag}", f"PTQ INT{bits}", None),
            (f"qat_{tag}_baseline", f"QAT INT{bits} baseline", f"hf__standard__qwen3_1.7B_{tag}__scratch"),
            (f"qat_{tag}_bkd", f"QAT INT{bits} +bkd", f"silverspoon-kd__bkd__qwen3_1.7B__qwen3_1.7B_{tag}__scratch"),
            (f"qat_{tag}_hkd", f"QAT INT{bits} +hkd", f"silverspoon-kd__hkd__qwen3_1.7B__qwen3_1.7B_{tag}__scratch"),
            (f"qat_{tag}_reskd", f"QAT INT{bits} +reskd", f"silverspoon-kd__reskd__qwen3_1.7B__qwen3_1.7B_{tag}__scratch"),
        ]
        for name, display, run_name in entries:
            path = os.path.join(eval_dir, f"{name}.json")
            if not os.path.exists(path):
                continue
            data = read_json(path)
            r = data.get("results", data)
            rows.append({
                "name": display,
                "run_name": run_name,
                "mmlu": r.get("mmlu", {}).get("acc,none"),
                "arc_easy": r.get("arc_easy", {}).get("acc,none"),
                "arc_challenge": r.get("arc_challenge", {}).get("acc_norm,none"),
                "hellaswag": r.get("hellaswag", {}).get("acc_norm,none"),
                "wikitext_ppl": r.get("wikitext", {}).get("word_perplexity,none"),
            })
    return rows


def collect_gpt2_distillation(runs_dir):
    """GPT-2 decoder distillation (96M and 112M students)."""
    eval_dir = os.path.join(runs_dir, "eval_results")
    rows = []
    # Stage-1-only runs (BKD without fine-tune) that have broken eval
    # because the LM head was never trained — annotate them.
    STAGE1_ONLY = {"bkd"}
    # LoRA adapter whose base model is missing — can never be evaluated.
    SKIP_RUNS = {".32727b67."}
    for d in sorted(glob.glob(os.path.join(eval_dir, "*gpt2_96M*"))
                    + glob.glob(os.path.join(eval_dir, "*gpt2_112M*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        if any(tag in name for tag in SKIP_RUNS):
            continue
        r = read_lm_eval_results(d)
        if r is None:
            continue
        # Build readable display name: [size] method init
        size = "112M" if "gpt2_112M" in name else "96M"
        short = name.replace("silverspoon-kd__", "").replace("gpt2_small__", "")
        short = short.replace("hf__standard__", "").replace("hf__lora__", "lora__")
        short = short.replace("gpt2_112M__", "").replace("gpt2_96M__", "")
        short = short.replace("__scratch", "")
        # Clean init suffix: from-bkd.2dca3ba9.latest → from-BKD (old naming)
        #                    from-bkd → from-BKD (new naming)
        import re
        short = re.sub(r"from-(\w+?)(?:\.[a-f0-9]+\.latest)?(?=__|$)", lambda m: f"from-{m.group(1).upper()}", short)
        if short == "" or short == "scratch":
            short = "baseline"
        # Annotate stage-1-only runs (check before adding size prefix)
        is_stage1 = any(short == tag or short.startswith(tag + "__") for tag in STAGE1_ONLY)
        short = f"[{size}] {short}"
        if is_stage1:
            short += " (stage-1 only)"
        rows.append({
            "name": short,
            "run_name": name,
            "wikitext_ppl": r.get("wikitext/word_perplexity,none"),
            "lambada_acc": r.get("lambada_openai/acc,none"),
            "hellaswag": r.get("hellaswag/acc_norm,none"),
        })
    return rows


def collect_gpt2_cross_arch(runs_dir):
    """GPT-2 cross-architecture (DeepSeek-V3 96M student)."""
    eval_dir = os.path.join(runs_dir, "eval_results")
    rows = []
    for d in sorted(glob.glob(os.path.join(eval_dir, "*deepseek_v3_96M*"))):
        if not os.path.isdir(d):
            continue
        r = read_lm_eval_results(d)
        if r is None:
            continue
        name = os.path.basename(d)
        # Clean display name: strip toolkit/model prefixes, keep method + init
        short = name.replace("silverspoon-kd__", "").replace("gpt2_small__", "").replace("deepseek_v3_96M__", "")
        short = short.replace("hf__standard__", "").replace("__scratch", "")
        # Readable init suffixes
        if "from-hkd" in short:
            method = short.split("__")[0] if "__" in short else "baseline"
            short = "FT from HKD" if method in ("", "baseline") else f"{method} → FT"
        elif "from-reskd" in short:
            short = "FT from ReSKD"
        elif short.startswith("hkd__mlp_only"):
            short = "HKD (MLP only)"
        elif short in ("", "scratch"):
            short = "baseline"
        rows.append({
            "name": short,
            "run_name": name,
            "wikitext_ppl": r.get("wikitext/word_perplexity,none"),
            "lambada_acc": r.get("lambada_openai/acc,none"),
            "hellaswag": r.get("hellaswag/acc_norm,none"),
        })
    return rows


def collect_gpt2_linearization(runs_dir):
    """GPT-2 linearization (LoLCATs student).

    Hybrid (sliding-window + linear) is the canonical LoLCATs approach.
    Pure-linear runs are annotated as such.
    """
    eval_dir = os.path.join(runs_dir, "eval_results")
    rows = []
    for d in sorted(glob.glob(os.path.join(eval_dir, "*lolcats*"))):
        if not os.path.isdir(d):
            continue
        r = read_lm_eval_results(d)
        if r is None:
            continue
        name = os.path.basename(d)
        # The saved config records whether the run used hybrid (sliding-window +
        # linear) or pure linear attention; a "hybrid" run-name marker is the
        # fallback when the config lacks the field.
        run_dir = os.path.join(runs_dir, name)
        is_hybrid = "hybrid" in name
        if not is_hybrid and os.path.isdir(run_dir):
            cfg_path = os.path.join(run_dir, "config.yaml")
            if os.path.isfile(cfg_path):
                try:
                    import yaml
                    cfg = yaml.safe_load(open(cfg_path))
                    is_hybrid = cfg.get("student", {}).get("window_size") is not None
                except Exception:
                    pass

        # Build readable display name
        short = name
        short = short.replace("silverspoon-kd__", "").replace("gpt2_small__", "")
        short = short.replace("gpt2_small_lolcats_hybrid__", "").replace("gpt2_small_lolcats__", "")
        short = short.replace("hf__standard__", "").replace("hf__lora__", "lora__")
        short = short.replace("__scratch", "")
        # Clean up init suffixes (handles both old and new naming)
        short = short.replace("from-bkd_attn_mse1000", "from-BKD_attn")
        import re
        short = re.sub(r"from-(\w+?)(?:\.[a-f0-9]+\.latest)?(?=__|$)", lambda m: f"from-{m.group(1).upper()}", short)
        if short == "" or short == "scratch":
            short = "baseline"
        # Annotate pure-linear (non-hybrid) runs
        if not is_hybrid:
            short += " [pure-linear]"

        rows.append({
            "name": short,
            "run_name": name,
            "wikitext_ppl": r.get("wikitext/word_perplexity,none"),
            "lambada_acc": r.get("lambada_openai/acc,none"),
            "hellaswag": r.get("hellaswag/acc_norm,none"),
        })
    return rows


def collect_bert_encoder(runs_dir):
    """BERT encoder pre-training distillation (Stage 1 MLM + Stage 2 downstream)."""
    import math
    rows = []
    patterns = [
        "hf__standard__bert_T6__scratch",
        "hf__standard__bert_T4_tiny__scratch",
        "silverspoon-kd__bkd__bert_base_uncased__bert_T6__scratch",
        "silverspoon-kd__bkd__bert_base_uncased__bert_T4_tiny__scratch",
        "silverspoon-kd__hkd__bert_base_uncased__bert_T6__scratch",
        "silverspoon-kd__hkd__bert_base_uncased__bert_T4_tiny__scratch",
        "silverspoon-kd__reskd__bert_base_uncased__bert_T6__scratch",
        "silverspoon-kd__reskd__bert_base_uncased__bert_T4_tiny__scratch",
    ]

    def _lookup_downstream_mnli(runs_dir, name):
        """Find Stage 2 MNLI accuracy for a pre-training run."""
        # Determine student and distiller from the pre-training run name
        student = "bert_T6" if "bert_T6" in name else "bert_T4_tiny"
        if name.startswith("hf__standard__"):
            distiller = "standard"
        elif "__bkd__" in name:
            distiller = "bkd"
        elif "__hkd__" in name:
            distiller = "hkd"
        elif "__reskd__" in name:
            distiller = "reskd"
        else:
            return None
        # Look for the downstream fine-tuned run that initialised from the
        # uncased pre-training checkpoint (hf__standard__{s}__mnli__from-{d}
        # or its hash-suffixed variant from-{d}.HASH.latest). We must NOT
        # match `from-{d}-mnli` — those init from a single-stage downstream
        # distillation checkpoint and belong to the BERT Downstream table,
        # not the pre-training Stage 2 column.
        patterns = [
            f"hf__standard__{student}__mnli__from-{distiller}",
            f"hf__standard__{student}__mnli__from-{distiller}.*",
        ]
        matches = []
        for pattern in patterns:
            matches.extend(glob.glob(os.path.join(runs_dir, pattern)))
        for match in sorted(set(matches), reverse=True):  # prefer most recent
            if match.endswith(".old") or ".old" in os.path.basename(match):
                continue
            ts = latest_trainer_state(match)
            if ts is None:
                continue
            info = best_eval_metric(ts, "eval_accuracy")
            if info is not None:
                return info["value"]
        return None

    for pattern in patterns:
        d = os.path.join(runs_dir, pattern)
        if not os.path.isdir(d):
            continue
        ts_path = latest_trainer_state(d)
        if ts_path is None:
            continue
        # For loss, we want the minimum (not max like accuracy)
        ts = read_json(ts_path)
        log = ts.get("log_history", [])
        evals = [e for e in log if "eval_loss" in e and "eval_runtime" in e]
        if not evals:
            continue
        best = min(evals, key=lambda e: e["eval_loss"])
        name = os.path.basename(d)
        short = (name.replace("hf__standard__", "baseline__")
                     .replace("silverspoon-kd__", "")
                     .replace("__bert_base_uncased", "")
                     .replace("__scratch", ""))
        eval_loss = best["eval_loss"]
        is_baseline = name.startswith("hf__standard__")

        # Get e2e loss (student's own MLM forward loss during distillation eval)
        e2e_entries = [e for e in log if "eval_loss/e2e" in e]
        e2e_info = min(e2e_entries, key=lambda e: e["eval_loss/e2e"]) if e2e_entries else None
        e2e_loss = e2e_info["eval_loss/e2e"] if e2e_info else None

        # MLM PPL: for baselines, eval_loss IS the MLM loss.
        # For distillation runs, e2e_loss is the MLM loss (if available).
        mlm_loss = eval_loss if is_baseline else e2e_loss
        mlm_ppl = math.exp(mlm_loss) if mlm_loss is not None and mlm_loss < 20 else None

        # Stage 2: downstream MNLI accuracy (from fine-tuned checkpoint)
        mnli = _lookup_downstream_mnli(runs_dir, name)

        rows.append({
            "name": short,
            "run_name": name,
            "eval_loss": eval_loss if not is_baseline else None,
            "mlm_ppl": mlm_ppl,
            "e2e_loss": e2e_loss,
            "mnli": mnli,
        })
    return rows


def collect_bert_downstream(runs_dir):
    """BERT downstream MNLI head-to-head (silverspoon vs TextBrewer)."""
    results_dir = os.path.join(os.path.dirname(runs_dir), "results", "downstream")
    rows = []
    # Readable name mapping for static result JSONs
    _DOWNSTREAM_NAMES = {
        "bert_T4_tiny_mnli_baseline": "T4-tiny baseline (no pretrain)",
        "bert_T4_tiny_mnli_bkd":     "T4-tiny BKD → FT",
        "bert_T4_tiny_mnli_hkd":     "T4-tiny HKD (single-stage)",
        "bert_T6_mnli_baseline":     "T6 baseline",
        "bert_T6_mnli_bkd":         "T6 BKD → FT",
        "bert_T6_mnli_hkd":         "T6 HKD (single-stage)",
        "teacher_mnli":              "Teacher (BERT-base)",
        "textbrewer_bert_T4_tiny_mnli": "TextBrewer T4-tiny",
        "textbrewer_bert_T6_mnli":     "TextBrewer T6",
    }
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        name = os.path.basename(f).replace(".json", "")
        # Skip eval_glue.sh outputs (full run names) — loop 2 handles them
        # with cleaner display names and deduplication.
        if name.startswith("hf__standard__") or name.startswith("silverspoon-kd__"):
            continue
        data = read_json(f)
        short = _DOWNSTREAM_NAMES.get(name, name)
        # Derive run_name from model_path so _prep_rows can enrich with
        # peak_gpu_mb / steps_per_sec / runtime_h from the wandb cache.
        # The model_path points to the run that actually produced the
        # evaluated weights — for BKD this is the "from-bkd" FT step, not
        # the BKD distillation step (which produces unusable weights alone).
        # Reference rows (teacher / TextBrewer / baselines) get run_name=None
        # by suffix-filtering, so they stay --- in the system-metric columns.
        run_name = None
        if any(name.endswith(s) for s in ("_bkd", "_hkd", "_reskd")):
            mp = data.get("model_path")
            if mp:
                parts = mp.split("/runs/", 1)
                if len(parts) == 2:
                    run_name = parts[1].split("/", 1)[0]
        rows.append({
            "name": short,
            "run_name": run_name,
            "matched": data.get("accuracy_matched", data.get("accuracy")),
            "mismatched": data.get("accuracy_mismatched"),
            "epoch": data.get("epoch"),
        })
    # NOTE: Stage-2 fine-tuned models (hf__standard__bert_T*__mnli__from-*)
    # are excluded here — they belong to the §5.2 BERT Pre-Training experiment
    # and are shown in that table's "MNLI % (Stage 2)" column instead.

    # Stage-1 distillation runs with eval_accuracy (bert_base_cased direct MNLI distillation)
    # Patterns cover both old names (with loss field: __mnli__LOSS__scratch)
    # and new names (without loss field: __mnli__scratch).
    for pattern in [
        "silverspoon-kd__*__bert_base_cased__bert_T6__mnli__scratch",
        "silverspoon-kd__*__bert_base_cased__bert_T6__mnli__*__scratch",
        "silverspoon-kd__*__bert_base_cased__bert_T4_tiny__mnli__scratch",
        "silverspoon-kd__*__bert_base_cased__bert_T4_tiny__mnli__*__scratch",
        "hf__standard__bert_T6__mnli__from-bkd*",
        "hf__standard__bert_T4_tiny__mnli__from-bkd*",
    ]:
        for d in sorted(glob.glob(os.path.join(runs_dir, pattern))):
            name = os.path.basename(d)
            if ".old" in name:
                continue
            ts_path = latest_trainer_state(d)
            if ts_path is None:
                continue
            info = best_eval_metric(ts_path, "eval_accuracy")
            if info is None:
                continue
            # Extract student and method
            student = "T6" if "bert_T6" in name else "T4-tiny"
            if "from-bkd" in name:
                method = "BKD → FT"
            elif "__bkd__" in name:
                method = "BKD"
            elif "__hkd__" in name:
                method = "HKD"
            elif "__reskd__" in name:
                method = "ReSKD"
            else:
                method = "?"
            suffix = " (projfix)" if "projfix" in name else ""
            short = f"{student} {method} (downstream){suffix}"
            rows.append({
                "name": short,
                "run_name": name,
                "matched": info["value"],
                "mismatched": None,
                "epoch": info.get("epoch"),
            })
    return rows




def collect_vision_cifar100(runs_dir):
    """Vision RelKD CIFAR-100 (ResNet50 → VGG11-BN)."""
    rows = []
    for pattern in ["hf__standard__resnet50_cifar100*", "hf__standard__vgg11bn_cifar100*",
                     "silverspoon-kd__*cifar100*"]:
        for d in sorted(glob.glob(os.path.join(runs_dir, pattern))):
            if ".old" in os.path.basename(d) or "_archive" in d:
                continue
            ts_path = latest_trainer_state(d)
            if ts_path is None:
                continue
            info = best_eval_metric(ts_path, "eval_accuracy")
            name = os.path.basename(d)
            short = name.replace("silverspoon-kd__", "").replace("hf__standard__", "")
            short = short.replace("resnet50_cifar100__vgg11bn_cifar100__", "").replace("__scratch", "")
            if info:
                rows.append({"name": short, "run_name": name, "accuracy": info["value"], "epoch": info.get("epoch"),
                             "step": f"{info.get('global_step', '?')}/{info.get('max_steps', '?')}"})
            else:
                # Distillers don't log eval_accuracy; check post-hoc eval results
                acc = vision_eval_accuracy(name, runs_dir)
                ts = read_json(ts_path)
                evals = [e for e in ts.get("log_history", []) if "eval_loss" in e]
                if evals or acc is not None:
                    rows.append({"name": short, "run_name": name, "accuracy": acc,
                                 "eval_loss": evals[-1].get("eval_loss") if evals else None,
                                 "step": f"{ts.get('global_step', '?')}/{ts.get('max_steps', '?')}"})
    return rows


def collect_vision_cifar10(runs_dir):
    """Vision BKD CIFAR-10 (VGG16 → VGG16-DS)."""
    rows = []
    for pattern in ["hf__standard__vgg16*", "silverspoon-kd__*vgg16*"]:
        for d in sorted(glob.glob(os.path.join(runs_dir, pattern))):
            if ".old" in os.path.basename(d) or "_archive" in d:
                continue
            ts_path = latest_trainer_state(d)
            if ts_path is None:
                continue
            info = best_eval_metric(ts_path, "eval_accuracy")
            name = os.path.basename(d)
            short = name.replace("silverspoon-kd__", "").replace("hf__standard__", "")
            short = short.replace("vgg16_cifar10__vgg16_cifar10_depthwise_separable__", "DS ")
            if info:
                rows.append({"name": short, "run_name": name, "accuracy": info["value"], "epoch": info.get("epoch")})
            else:
                # Distillers don't log eval_accuracy; check post-hoc eval results
                acc = vision_eval_accuracy(name, runs_dir)
                ts = read_json(ts_path)
                step = ts.get("global_step")
                max_steps = ts.get("max_steps")
                rows.append({"name": short, "run_name": name, "accuracy": acc,
                             "epoch": None,
                             "step": f"{step}/{max_steps}" if step and max_steps else None})
    return rows


# ── Run status scanner ───────────────────────────────────────────────────────

# Every experiment run we expect to exist, grouped by experiment.
# Each entry is (run_name_pattern, description).  Patterns use shell-style
# globs so that hash-suffixed names (e.g. from-bkd.abc123.latest) match.
EXPECTED_RUNS = {
    "VGG Compression": [
        ("hf__standard__vgg16_cifar10_teacher__scratch", "Teacher (VGG16)"),
        ("hf__standard__vgg16_cifar10_depthwise_separable__scratch", "Baseline (DS)"),
        ("silverspoon-kd__bkd__vgg16_cifar10__vgg16_cifar10_depthwise_separable__*", "BKD"),
        ("hf__standard__vgg16_cifar10_depthwise_separable__from-bkd*", "BKD → FT"),
        ("silverspoon-kd__hkd__vgg16_cifar10__vgg16_cifar10_depthwise_separable__from-bkd*", "BKD → HKD"),
        ("silverspoon-kd__reskd__vgg16_cifar10__vgg16_cifar10_depthwise_separable__from-bkd*", "BKD → ReSKD"),
        ("silverspoon-kd__reskd__vgg16_cifar10__vgg16_cifar10_depthwise_separable__scratch", "ReSKD (scratch)"),
    ],
    "VGG Relational KD": [
        ("hf__standard__resnet50_cifar100__scratch", "Teacher (ResNet50)"),
        ("hf__standard__vgg11bn_cifar100__scratch", "Baseline (VGG11-BN)"),
        ("silverspoon-kd__reskd__resnet50_cifar100__vgg11bn_cifar100__scratch", "ReSKD"),
        ("silverspoon-kd__hkd__resnet50_cifar100__vgg11bn_cifar100__scratch", "HKD (MSE)"),
        ("silverspoon-kd__hkd__resnet50_cifar100__vgg11bn_cifar100__relkd_distance__scratch", "HKD (RelKD-D)"),
        ("silverspoon-kd__hkd__resnet50_cifar100__vgg11bn_cifar100__relkd_angle__scratch", "HKD (RelKD-A)"),
        ("silverspoon-kd__hkd__resnet50_cifar100__vgg11bn_cifar100__relkd_da__scratch", "HKD (RelKD-DA)"),
        ("hf__standard__vgg11bn_cifar100__from-hkd*", "HKD → FT"),
    ],
    "BERT Pre-Training": [
        ("hf__standard__bert_T6__scratch", "T6 baseline"),
        ("hf__standard__bert_T4_tiny__scratch", "T4-tiny baseline"),
        ("silverspoon-kd__bkd__bert_base_uncased__bert_T6__scratch", "T6 BKD"),
        ("silverspoon-kd__bkd__bert_base_uncased__bert_T4_tiny__scratch", "T4-tiny BKD"),
        ("silverspoon-kd__hkd__bert_base_uncased__bert_T6__scratch", "T6 HKD"),
        ("silverspoon-kd__hkd__bert_base_uncased__bert_T4_tiny__scratch", "T4-tiny HKD"),
        ("silverspoon-kd__reskd__bert_base_uncased__bert_T6__scratch", "T6 ReSKD"),
        ("silverspoon-kd__reskd__bert_base_uncased__bert_T4_tiny__scratch", "T4-tiny ReSKD"),
    ],
    "BERT Downstream": [
        ("hf__standard__bert_base_cased__mnli__scratch", "Teacher"),
        ("textbrewer__bert_T6__mnli", "TextBrewer T6"),
        ("textbrewer__bert_T4-tiny__mnli", "TextBrewer T4-tiny"),
        ("silverspoon-kd__hkd__bert_base_cased__bert_T6__mnli__scratch", "HKD T6 stage-1"),
        ("silverspoon-kd__bkd__bert_base_cased__bert_T6__mnli__scratch", "BKD T6 stage-1"),
        ("hf__standard__bert_T6__mnli__from-hkd*", "HKD T6 → FT"),
        ("hf__standard__bert_T6__mnli__from-bkd*", "BKD T6 → FT"),
    ],
    "GPT-2 Compression": [
        ("hf__standard__gpt2_96M__scratch", "96M Baseline"),
        ("silverspoon-kd__bkd__gpt2_small__gpt2_96M__scratch", "96M BKD"),
        ("silverspoon-kd__hkd__gpt2_small__gpt2_96M__from-bkd*", "96M BKD → HKD"),
        ("silverspoon-kd__reskd__gpt2_small__gpt2_96M__scratch", "96M ReSKD"),
        ("hf__lora__gpt2_96M__from-hkd*", "96M LoRA (from HKD)"),
        ("hf__standard__gpt2_112M__scratch", "112M Baseline"),
        ("silverspoon-kd__bkd__gpt2_small__gpt2_112M__scratch", "112M BKD"),
        ("silverspoon-kd__hkd__gpt2_small__gpt2_112M__from-bkd*", "112M BKD → HKD"),
        ("silverspoon-kd__reskd__gpt2_small__gpt2_112M__scratch", "112M ReSKD"),
    ],
    "Cross-Architecture": [
        ("hf__standard__deepseek_v3_96M__scratch", "Baseline"),
        ("silverspoon-kd__hkd__gpt2_small__deepseek_v3_96M__scratch", "HKD"),
        ("silverspoon-kd__reskd__gpt2_small__deepseek_v3_96M__scratch", "ReSKD"),
        ("hf__standard__deepseek_v3_96M__from-hkd*", "HKD → FT"),
        ("hf__standard__deepseek_v3_96M__from-reskd*", "ReSKD → FT"),
    ],
    "GPT-2 Attention Linearization": [
        ("hf__standard__gpt2_small_lolcats__scratch", "Baseline"),
        ("silverspoon-kd__hkd__gpt2_small__gpt2_small_lolcats__scratch", "HKD"),
        ("silverspoon-kd__reskd__gpt2_small__gpt2_small_lolcats__scratch", "ReSKD"),
        ("silverspoon-kd__bkd__gpt2_small__gpt2_small_lolcats__scratch", "BKD (stage-1)"),
        ("silverspoon-kd__hkd__gpt2_small__gpt2_small_lolcats__from-bkd*", "BKD → HKD"),
        ("silverspoon-kd__reskd__gpt2_small__gpt2_small_lolcats__from-bkd*", "BKD → ReSKD"),
        ("hf__standard__gpt2_small_lolcats__from-bkd*", "BKD → FT"),
    ],
    "Qwen3 Quantization": [
        ("hf__standard__qwen3_1.7B_int4__scratch", "QAT INT4 baseline"),
        ("silverspoon-kd__bkd__qwen3_1.7B__qwen3_1.7B_int4__scratch", "QAT INT4 +BKD"),
        ("silverspoon-kd__hkd__qwen3_1.7B__qwen3_1.7B_int4__scratch", "QAT INT4 +HKD"),
        ("silverspoon-kd__reskd__qwen3_1.7B__qwen3_1.7B_int4__scratch", "QAT INT4 +ReSKD"),
        ("hf__standard__qwen3_1.7B_int5__scratch", "QAT INT5 baseline"),
        ("silverspoon-kd__bkd__qwen3_1.7B__qwen3_1.7B_int5__scratch", "QAT INT5 +BKD"),
        ("silverspoon-kd__hkd__qwen3_1.7B__qwen3_1.7B_int5__scratch", "QAT INT5 +HKD"),
        ("silverspoon-kd__reskd__qwen3_1.7B__qwen3_1.7B_int5__scratch", "QAT INT5 +ReSKD"),
        ("hf__standard__qwen3_1.7B_int6__scratch", "QAT INT6 baseline"),
        ("silverspoon-kd__bkd__qwen3_1.7B__qwen3_1.7B_int6__scratch", "QAT INT6 +BKD"),
        ("silverspoon-kd__hkd__qwen3_1.7B__qwen3_1.7B_int6__scratch", "QAT INT6 +HKD"),
        ("silverspoon-kd__reskd__qwen3_1.7B__qwen3_1.7B_int6__scratch", "QAT INT6 +ReSKD"),
        ("hf__standard__qwen3_1.7B_int8__scratch", "QAT INT8 baseline"),
        ("silverspoon-kd__bkd__qwen3_1.7B__qwen3_1.7B_int8__scratch", "QAT INT8 +BKD"),
        ("silverspoon-kd__hkd__qwen3_1.7B__qwen3_1.7B_int8__scratch", "QAT INT8 +HKD"),
        ("silverspoon-kd__reskd__qwen3_1.7B__qwen3_1.7B_int8__scratch", "QAT INT8 +ReSKD"),
    ],
}


def classify_run(run_dir):
    """Classify a run directory's status.

    Returns one of:
        "completed"  — student_model/ or model/ exists (training finished)
        "running"    — checkpoint-* exists but no final model (still training)
        "errored"    — only config.yaml exists (crashed before first checkpoint)
        "not_found"  — directory doesn't exist at all
    """
    if not os.path.isdir(run_dir):
        return "not_found"
    contents = set(os.listdir(run_dir))
    if "student_model" in contents or "model" in contents or "lora_model" in contents:
        return "completed"
    if any(c.startswith("checkpoint-") for c in contents):
        return "running"
    if "config.yaml" in contents:
        return "errored"
    return "not_found"


def get_run_progress(run_dir):
    """For running jobs, extract the current step / max_steps."""
    ts_path = latest_trainer_state(run_dir)
    if ts_path is None:
        return None
    try:
        ts = read_json(ts_path)
        return f"{ts.get('global_step', '?')}/{ts.get('max_steps', '?')}"
    except Exception:
        return None


def check_eval_status(run_name, eval_results_dir):
    """Check whether a completed training run has post-training lm-eval results."""
    return os.path.isdir(os.path.join(eval_results_dir, run_name))


# Groups whose results come from lm-eval (post-training eval_results/).
# Other groups (BERT, vision) get results from trainer_state during training.
LM_EVAL_GROUPS = {"GPT-2 Compression", "Cross-Architecture", "GPT-2 Attention Linearization"}


def collect_run_status(runs_dir):
    """Scan all expected runs and classify their status."""
    eval_results_dir = os.path.join(runs_dir, "eval_results")
    status_by_group = {}
    for group, expected in EXPECTED_RUNS.items():
        statuses = []
        for pattern, description in expected:
            # Expand glob
            matches = sorted(glob.glob(os.path.join(runs_dir, pattern)))
            if not matches:
                statuses.append((description, pattern, "not_found", None, None))
            else:
                for m in matches:
                    name = os.path.basename(m)
                    status = classify_run(m)
                    progress = get_run_progress(m) if status == "running" else None
                    # For lm-eval groups, check if post-training eval exists
                    eval_done = None
                    if group in LM_EVAL_GROUPS and status == "completed":
                        eval_done = check_eval_status(name, eval_results_dir)
                    statuses.append((description, name, status, progress, eval_done))
        status_by_group[group] = statuses
    return status_by_group


STATUS_SYMBOLS = {
    "completed": "\033[32m✓\033[0m",   # green checkmark
    "running":   "\033[33m◉\033[0m",   # yellow circle
    "errored":   "\033[31m✗\033[0m",   # red cross
    "not_found": "\033[90m·\033[0m",   # gray dot
}

STATUS_LABELS = {
    "completed": "done",
    "running":   "running",
    "errored":   "errored",
    "not_found": "not started",
}


def print_run_status(runs_dir):
    """Print a status overview of all expected experiment runs."""
    status_by_group = collect_run_status(runs_dir)
    counts = defaultdict(int)
    eval_counts = {"evaluated": 0, "awaiting": 0}

    print(f"\n{'=' * 70}")
    print("  Run Status Overview")
    print(f"{'=' * 70}")

    for group, statuses in status_by_group.items():
        print(f"\n  {group}:")
        for desc, name, status, progress, eval_done in statuses:
            sym = STATUS_SYMBOLS.get(status, "?")
            label = STATUS_LABELS.get(status, status)
            extra = f" ({progress})" if progress else ""
            # For completed lm-eval runs, show eval status
            if eval_done is True:
                extra += " \033[32m[eval ✓]\033[0m"
                eval_counts["evaluated"] += 1
            elif eval_done is False:
                extra += " \033[33m[awaiting eval]\033[0m"
                eval_counts["awaiting"] += 1
            # Truncate long run names
            display_name = name if len(name) <= 50 else "..." + name[-47:]
            print(f"    {sym} {desc:<25} {label:<12} {display_name}{extra}")
            counts[status] += 1

    parts = [
        f"{counts['completed']} done",
        f"{counts['running']} running",
        f"{counts['errored']} errored",
        f"{counts['not_found']} not started",
    ]
    if eval_counts["awaiting"]:
        parts.append(f"\033[33m{eval_counts['awaiting']} awaiting post-training eval\033[0m")
    print(f"\n  Summary: {', '.join(parts)}")


# ── Threshold flags ──────────────────────────────────────────────────────────
# Per-group, per-metric thresholds.  Each entry is (threshold, direction):
#   direction ">" → flag red if value > threshold  (e.g. PPL too high)
#   direction "<" → flag red if value < threshold  (e.g. accuracy too low)
# Thresholds flag results with unacceptable quality degradation relative
# to the teacher, not just catastrophic failures.

THRESHOLDS = {
    "qat": {
        # Teacher: MMLU 0.56, ARC-E 0.72, ARC-C 0.43, HS 0.60, PPL 21
        # INT8 is near-lossless and should pass; INT4 shows unacceptable degradation
        "wikitext_ppl":   (25,    ">"),   # INT8 ~20-21 (pass), INT4 30-37 (flag)
        "mmlu":           (0.50,  "<"),   # INT8 ~0.55 (pass), INT4 0.41-0.48 (flag)
        "arc_easy":       (0.65,  "<"),   # INT8 ~0.72 (pass), INT4 0.58-0.66 (flag)
        "arc_challenge":  (0.40,  "<"),   # INT8 ~0.43 (pass), INT4 0.32-0.37 (flag)
        "hellaswag":      (0.57,  "<"),   # INT8 ~0.60 (pass), INT4 0.54-0.55 (flag)
    },
    "gpt2_compression": {
        # Teacher: PPL 37, LAMBADA 0.326, HS 0.311
        # Good: LoRA/HKD PPL 105-110, LAMBADA 0.17-0.18
        # Bad: BKD stage-1 PPL 242K+, LAMBADA 0.000 (expected — BKD is pre-pretraining)
        "wikitext_ppl":   (200,   ">"),   # baselines 132-163 (pass), BKD 242K (flag)
        "lambada_acc":    (0.10,  "<"),   # baselines 0.10+ (pass), ReSKD 0.06 (flag)
        "hellaswag":      (0.265, "<"),   # teacher 0.311, baselines 0.267+ (pass)
    },
    "cross_arch": {
        # Teacher: PPL 37, LAMBADA 0.326, HS 0.311
        # Even best (from-hkd): PPL 150, LAMBADA 0.13, HS 0.269
        # Cross-arch is fundamentally hard; flag anything clearly broken
        "wikitext_ppl":   (175,   ">"),   # from-hkd 150 (pass), HKD 282/333 (flag)
        "lambada_acc":    (0.10,  "<"),   # from-hkd 0.13 (pass), ReSKD 0.05 HKD 0.004 (flag)
        "hellaswag":      (0.265, "<"),   # from-hkd 0.269 (pass), HKD 0.260 (flag)
    },
    "linearization": {
        # Teacher: PPL 37, LAMBADA 0.326
        # Healthy hybrid: PPL 330, LAMBADA 0.31 (nearly teacher-level)
        # Many broken pure-linear runs: PPL in millions
        "wikitext_ppl":   (500,   ">"),   # hybrid 330 (pass), broken runs (flag)
        "lambada_acc":    (0.02,  "<"),   # baseline 0.05 (pass), broken 0.000 (flag)
    },
    "bert_pretraining": {
        # Baseline T6: MLM PPL 14, T4-tiny: 24, HKD T6: 41
        # Bad: BKD T6 297, BKD T4-tiny 2634, HKD T4-tiny 2376
        "mlm_ppl":        (100,   ">"),   # HKD T6 41 (pass), BKD/HKD T4-tiny 2K+ (flag)
    },
    "bert_downstream": {
        # TextBrewer T4-tiny: 81%, T6: 83%; teacher: 84%
        # SK T4-tiny HKD: 69%, BKD: 63%, baseline: 32% (chance)
        "matched":        (0.60,  "<"),   # BKD 63% (pass), baseline 32% (flag)
    },
    "vgg_relkd": {
        # RelKD paper (Park et al. CVPR 2019, Table 4): ResNet50→VGG11-BN CIFAR-100
        # Paper baseline: 71.26%, RKD-D: 72.27%, RKD-DA: 72.97%, HKD: 74.26%
        "accuracy":       (0.7126, "<"),  # flag anything below the paper's baseline
    },
    "vgg_compression": {
        "accuracy":       (0.70,  "<"),   # teacher 87%, baseline 83%
    },
}

_ANSI_RED = "\033[31m"
_ANSI_RESET = "\033[0m"


def _is_flagged(value, col, group_key):
    """Check if a cell value exceeds the threshold for its group/column."""
    thresholds = THRESHOLDS.get(group_key)
    if thresholds is None:
        return False
    rule = thresholds.get(col)
    if rule is None:
        return False
    threshold, direction = rule
    if not isinstance(value, (int, float)):
        return False
    if direction == ">" and value > threshold:
        return True
    if direction == "<" and value < threshold:
        return True
    return False


# ── Formatters ───────────────────────────────────────────────────────────────

def fmt(val, fmt_str=".4f"):
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:{fmt_str}}"
    return str(val)


def print_table(title, headers, rows, key_col="name", group_key=None,
                flag_thresholds=False):
    """Print a nicely formatted terminal table."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    if not rows:
        print("  (no results)")
        return

    # Compute column widths (based on raw text, no ANSI codes)
    cols = list(headers.keys())
    widths = {c: max(len(headers[c]), max((len(fmt(r.get(c, ""), ".4f" if isinstance(r.get(c), float) else "")) for r in rows), default=0)) for c in cols}

    # Header
    header_line = "  ".join(headers[c].ljust(widths[c]) if c == key_col else headers[c].rjust(widths[c]) for c in cols)
    print(header_line)
    print("  ".join("-" * widths[c] for c in cols))

    # Track flagged count for summary
    n_flagged = 0

    # Rows
    for r in rows:
        row_name = r.get("name") or ""
        is_teacher = "Teacher" in row_name or "teacher" in (r.get("run_name") or "")
        is_stage1 = "stage-1 only" in row_name
        cells = []
        for c in cols:
            v = r.get(c)
            s = fmt(v) if isinstance(v, float) else (str(v) if v is not None else "—")
            aligned = s.ljust(widths[c]) if c == key_col else s.rjust(widths[c])
            # Flag bad values (skip teacher/stage-1 rows — they're references or expected-broken)
            flagged_cols = r.get("_flagged_cols", set())
            if flag_thresholds and not is_teacher and not is_stage1 and (
                _is_flagged(v, c, group_key) or c in flagged_cols
            ):
                aligned = f"{_ANSI_RED}{aligned}{_ANSI_RESET}"
                n_flagged += 1
            cells.append(aligned)
        print("  ".join(cells))

    if flag_thresholds and n_flagged > 0:
        print(f"  {_ANSI_RED}({n_flagged} value(s) flagged as below expected threshold){_ANSI_RESET}")


def print_latex_table(title, headers, rows, key_col="name", label=""):
    """Print a LaTeX-ready table."""
    cols = list(headers.keys())
    n = len(cols)
    align = "l" + "r" * (n - 1)
    print(f"\n% {title}")
    print("\\begin{table}[ht]")
    print("\\centering")
    print(f"\\caption{{{title}}}")
    if label:
        print(f"\\label{{tab:{label}}}")
    print(f"\\begin{{tabular}}{{{align}}}")
    print("\\toprule")
    print(" & ".join(headers[c] for c in cols) + " \\\\")
    print("\\midrule")
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                cells.append(f"{v:.4f}" if v < 1 else f"{v:.2f}")
            elif v is None:
                cells.append("—")
            else:
                cells.append(str(v).replace("_", "\\_"))
        print(" & ".join(cells) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


# ── Paper-tables template engine ─────────────────────────────────────────────
#
# Renders LaTeX templates in paper/generated/_templates/<name>.tex.tmpl into
# paper/generated/<name>.tex, substituting {{ var.path|filter1|filter2 }}
# placeholders with values pulled from the existing collector outputs. Each
# experiment group has a NAMESPACE_BUILDER that maps collected rows into a
# nested dict keyed by logical name (e.g. "baseline", "bkd", "bkd_ft").
#
# This isolates *formatting* in the .tmpl files (LaTeX-readable, mirrors
# what the paper already has) and keeps the *data plumbing* in Python. The
# rendered .tex files are what main.tex \input{}s — main.tex itself never
# changes after the initial wiring.

_TMPL_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][\w\.]*)((?:\s*\|\s*\w+)*)\s*\}\}")
_NA = "---"


def _filter_pct(v):  return f"{v*100:.2f}" if v is not None else _NA
def _filter_pct1(v): return f"{v*100:.1f}" if v is not None else _NA
def _filter_dec1(v): return f"{v:.1f}" if v is not None else _NA
def _filter_dec2(v): return f"{v:.2f}" if v is not None else _NA
def _filter_dec3(v): return f"{v:.3f}" if v is not None else _NA
def _filter_dec4(v): return f"{v:.4f}" if v is not None else _NA
def _filter_int(v):  return f"{int(round(v))}" if v is not None else _NA
def _filter_thou(v):
    if v is None: return _NA
    return f"{int(round(v)):,}".replace(",", r"\,")  # thin-space sep
def _filter_comma(v):
    if v is None: return _NA
    return f"{int(round(v)):,}".replace(",", "{,}")  # literal-comma sep
def _filter_dec2thou(v):
    if v is None: return _NA
    return f"{v:,.2f}".replace(",", r"\,")
def _filter_dec2comma(v):
    if v is None: return _NA
    return f"{v:,.2f}".replace(",", "{,}")
# Multi-stage bar overlays in the summary figure: render half/third/two-thirds
# of a percentage value, used to render split-color segments per chain stage.
def _filter_pct_half(v):        return f"{v*50:.2f}" if v is not None else _NA
def _filter_pct_third(v):       return f"{v*100/3:.2f}" if v is not None else _NA
def _filter_pct_two_thirds(v):  return f"{v*200/3:.2f}" if v is not None else _NA
# Signed percentage delta (1dp) wrapped in math mode for LaTeX:
# +5.7 -> "$+5.7\%$", -32.0 -> "$-32.0\%$"
def _filter_signed_pct1(v):
    if v is None: return _NA
    sign = "+" if v > 0 else ""
    return f"${sign}{v:.1f}\\%$"
# Ratio in math mode with 2dp + \times suffix: 1.67 -> "$1.67\times$"
def _filter_ratio(v):
    if v is None: return _NA
    return f"${v:.2f}\\times$"
def _filter_raw(v):  return str(v) if v is not None else _NA

_FILTERS = {
    "pct": _filter_pct, "pct1": _filter_pct1,
    "dec1": _filter_dec1, "dec2": _filter_dec2, "dec3": _filter_dec3, "dec4": _filter_dec4,
    "int": _filter_int,
    "thou": _filter_thou, "comma": _filter_comma,
    "dec2thou": _filter_dec2thou, "dec2comma": _filter_dec2comma,
    "pct_half": _filter_pct_half, "pct_third": _filter_pct_third,
    "pct_two_thirds": _filter_pct_two_thirds,
    "signed_pct1": _filter_signed_pct1, "ratio": _filter_ratio,
    "raw": _filter_raw,
}


def _ns_lookup(ns, path):
    """Walk dotted path through nested dict; return None if any segment missing."""
    for part in path.split("."):
        if ns is None:
            return None
        ns = ns.get(part) if isinstance(ns, dict) else None
    return ns


def render_template(tmpl_str, namespace):
    """Substitute {{ var.path|filter1|filter2 }} placeholders in tmpl_str."""
    def replace(m):
        path = m.group(1)
        filters = re.findall(r"\|\s*(\w+)", m.group(2) or "")
        v = _ns_lookup(namespace, path)
        if not filters:
            return _filter_raw(v)
        for f in filters:
            if f not in _FILTERS:
                raise ValueError(f"Unknown template filter: |{f}")
            v = _FILTERS[f](v)
        return str(v) if v is not None else _NA
    return _TMPL_PATTERN.sub(replace, tmpl_str)


def _parse_mb(s):
    """Parse '692 MB' → 692. Returns None if input is None/empty/unparsable."""
    if not s:
        return None
    try:
        return int(s.split()[0])
    except (ValueError, IndexError):
        return None


def _row_metrics(row, accuracy_key="accuracy"):
    """Extract the common (acc, mem, sps, time) tuple a row exposes."""
    return {
        "acc":  row.get(accuracy_key),
        "mem":  _parse_mb(row.get("peak_gpu_mb")),
        "sps":  row.get("steps_per_sec"),
        "time": row.get("runtime_h"),
    }


def _ns_by_run_name(rows, mapping, metrics_fn=_row_metrics):
    """Build a namespace dict by matching each row's run_name against the
    glob patterns in `mapping` (a dict of {pattern: logical_key})."""
    import fnmatch
    ns = {}
    for row in rows:
        name = row.get("run_name") or row.get("name", "")
        for pattern, key in mapping.items():
            if fnmatch.fnmatchcase(name, pattern):
                ns[key] = metrics_fn(row)
                break
    return ns


# ── Per-table namespace builders ─────────────────────────────────────────────
# Each builder takes (runs_dir) → namespace dict for one paper table.

def _prep_rows(rows, runs_dir):
    """Apply trainer-state + wandb injections (mutates rows in place)."""
    inject_trainer_state_columns(rows, runs_dir)
    inject_wandb_columns(rows, {}, runs_dir=runs_dir)


def _ns_vision_bkd(runs_dir):
    rows = collect_vision_cifar10(runs_dir)
    _prep_rows(rows, runs_dir)
    mapping = {
        "hf__standard__vgg16_cifar10_teacher__scratch": "teacher",
        "hf__standard__vgg16_cifar10_depthwise_separable__scratch": "baseline",
        "silverspoon-kd__bkd__vgg16_cifar10__vgg16_cifar10_depthwise_separable__scratch": "bkd",
        "hf__standard__vgg16_cifar10_depthwise_separable__from-bkd*": "bkd_ft",
        "silverspoon-kd__hkd__vgg16_cifar10__vgg16_cifar10_depthwise_separable__from-bkd*": "bkd_hkd",
        "silverspoon-kd__reskd__vgg16_cifar10__vgg16_cifar10_depthwise_separable__from-bkd*": "bkd_reskd",
        "silverspoon-kd__reskd__vgg16_cifar10__vgg16_cifar10_depthwise_separable__scratch": "reskd",
    }
    return _ns_by_run_name(rows, mapping)


def _ns_relkd(runs_dir):
    rows = collect_vision_cifar100(runs_dir)
    _prep_rows(rows, runs_dir)
    mapping = {
        "hf__standard__resnet50_cifar100__scratch": "teacher",
        "hf__standard__vgg11bn_cifar100__scratch": "baseline",
        "silverspoon-kd__reskd__resnet50_cifar100__vgg11bn_cifar100__scratch": "reskd",
        "silverspoon-kd__hkd__resnet50_cifar100__vgg11bn_cifar100__scratch": "hkd",
        "silverspoon-kd__hkd__resnet50_cifar100__vgg11bn_cifar100__relkd_distance__scratch": "relkd_d",
        "silverspoon-kd__hkd__resnet50_cifar100__vgg11bn_cifar100__relkd_angle__scratch":    "relkd_a",
        "silverspoon-kd__hkd__resnet50_cifar100__vgg11bn_cifar100__relkd_da__scratch":       "relkd_da",
    }
    return _ns_by_run_name(rows, mapping)


def _row_lm_metrics(row):
    """LM-eval metric extractor for GPT-2 group templates."""
    return {
        "ppl":        row.get("wikitext_ppl"),
        "lambada":    row.get("lambada_acc"),
        "hellaswag":  row.get("hellaswag"),
        "mem":        _parse_mb(row.get("peak_gpu_mb")),
        "sps":        row.get("steps_per_sec"),
        "time":       row.get("runtime_h"),
    }


def _ns_gpt2_multistage(runs_dir):
    rows = collect_gpt2_distillation(runs_dir)
    inject_teacher_row(rows, "gpt2_compression", runs_dir)
    _prep_rows(rows, runs_dir)
    mapping = {
        "gpt2_teacher": "teacher",
        "hf__standard__gpt2_96M__scratch": "baseline",
        "silverspoon-kd__bkd__gpt2_small__gpt2_96M__scratch": "bkd",
        "silverspoon-kd__hkd__gpt2_small__gpt2_96M__scratch": "hkd",
        "silverspoon-kd__reskd__gpt2_small__gpt2_96M__scratch": "reskd",
        "silverspoon-kd__hkd__gpt2_small__gpt2_96M__from-bkd*": "bkd_hkd",
        "hf__lora__gpt2_96M__from-hkd*": "bkd_hkd_lora",
    }
    return _ns_by_run_name(rows, mapping, metrics_fn=_row_lm_metrics)


def _ns_gpt2_deepseek(runs_dir):
    rows = collect_gpt2_cross_arch(runs_dir)
    inject_teacher_row(rows, "cross_arch", runs_dir)
    _prep_rows(rows, runs_dir)
    mapping = {
        "gpt2_teacher": "teacher",
        "hf__standard__deepseek_v3_96M__scratch": "baseline",
        "silverspoon-kd__hkd__gpt2_small__deepseek_v3_96M__scratch": "hkd",
        "silverspoon-kd__reskd__gpt2_small__deepseek_v3_96M__scratch": "reskd",
        "hf__standard__deepseek_v3_96M__from-hkd*": "hkd_ft",
        "hf__standard__deepseek_v3_96M__from-reskd*": "reskd_ft",
    }
    return _ns_by_run_name(rows, mapping, metrics_fn=_row_lm_metrics)


def _ns_gpt2_linear(runs_dir):
    rows = collect_gpt2_linearization(runs_dir)
    inject_teacher_row(rows, "linearization", runs_dir)
    _prep_rows(rows, runs_dir)
    mapping = {
        "gpt2_teacher": "teacher",
        "hf__standard__gpt2_small_lolcats__scratch": "baseline",
        "silverspoon-kd__bkd__gpt2_small__gpt2_small_lolcats__scratch": "bkd",
        "silverspoon-kd__hkd__gpt2_small__gpt2_small_lolcats__scratch": "hkd",
        "silverspoon-kd__hkd__gpt2_small__gpt2_small_lolcats__from-bkd*": "bkd_hkd",
        "silverspoon-kd__reskd__gpt2_small__gpt2_small_lolcats__scratch": "reskd",
        "silverspoon-kd__reskd__gpt2_small__gpt2_small_lolcats__from-bkd*": "bkd_reskd",
        "hf__standard__gpt2_small_lolcats__from-bkd*": "bkd_ft",
    }
    return _ns_by_run_name(rows, mapping, metrics_fn=_row_lm_metrics)


def _ns_qwen3_qat(runs_dir):
    rows = collect_qat(runs_dir)
    inject_teacher_row(rows, "qat", runs_dir)
    _prep_rows(rows, runs_dir)

    def qat_metrics(row):
        return {
            "mmlu":  row.get("mmlu"),
            "arc_e": row.get("arc_easy"),
            "arc_c": row.get("arc_challenge"),
            "hs":    row.get("hellaswag"),
            "ppl":   row.get("wikitext_ppl"),
            "mem":   _parse_mb(row.get("peak_gpu_mb")),
            "sps":   row.get("steps_per_sec"),
            "time":  row.get("runtime_h"),
        }

    # PTQ rows have run_name=None; match by display name instead.
    ns = {}
    name_map = {
        "Teacher (Qwen3-1.7B BF16)": "bf16",
        "PTQ INT4": "int4_ptq",
        "PTQ INT5": "int5_ptq",
        "PTQ INT6": "int6_ptq",
    }
    run_map = {
        "hf__standard__qwen3_1.7B_int4__scratch": "int4_qat",
        "silverspoon-kd__bkd__qwen3_1.7B__qwen3_1.7B_int4__scratch": "int4_bkd",
        "silverspoon-kd__hkd__qwen3_1.7B__qwen3_1.7B_int4__scratch": "int4_hkd",
        "silverspoon-kd__reskd__qwen3_1.7B__qwen3_1.7B_int4__scratch": "int4_reskd",
        "hf__standard__qwen3_1.7B_int5__scratch": "int5_qat",
        "silverspoon-kd__bkd__qwen3_1.7B__qwen3_1.7B_int5__scratch": "int5_bkd",
        "silverspoon-kd__hkd__qwen3_1.7B__qwen3_1.7B_int5__scratch": "int5_hkd",
        "silverspoon-kd__reskd__qwen3_1.7B__qwen3_1.7B_int5__scratch": "int5_reskd",
        "hf__standard__qwen3_1.7B_int6__scratch": "int6_qat",
        "silverspoon-kd__bkd__qwen3_1.7B__qwen3_1.7B_int6__scratch": "int6_bkd",
        "silverspoon-kd__hkd__qwen3_1.7B__qwen3_1.7B_int6__scratch": "int6_hkd",
        "silverspoon-kd__reskd__qwen3_1.7B__qwen3_1.7B_int6__scratch": "int6_reskd",
    }
    for row in rows:
        rn = row.get("run_name")
        nm = row.get("name")
        if rn in run_map:
            ns[run_map[rn]] = qat_metrics(row)
        elif nm in name_map:
            ns[name_map[nm]] = qat_metrics(row)
    return ns


def _row_bert_pretrain(row):
    return {
        "align":    row.get("eval_loss"),
        "ppl":      row.get("mlm_ppl"),
        "mlm_loss": row.get("e2e_loss"),
        "mnli":     row.get("mnli"),
        "mem":      _parse_mb(row.get("peak_gpu_mb")),
        "sps":      row.get("steps_per_sec"),
        "time":     row.get("runtime_h"),
    }


def _ns_bert_pretrain(runs_dir):
    rows = collect_bert_encoder(runs_dir)
    _prep_rows(rows, runs_dir)
    mapping = {
        "hf__standard__bert_T6__scratch":                                          "t6_baseline",
        "hf__standard__bert_T4_tiny__scratch":                                     "t4_baseline",
        "silverspoon-kd__bkd__bert_base_uncased__bert_T6__scratch":                "t6_bkd",
        "silverspoon-kd__bkd__bert_base_uncased__bert_T4_tiny__scratch":           "t4_bkd",
        "silverspoon-kd__hkd__bert_base_uncased__bert_T6__scratch":                "t6_hkd",
        "silverspoon-kd__hkd__bert_base_uncased__bert_T4_tiny__scratch":           "t4_hkd",
        "silverspoon-kd__reskd__bert_base_uncased__bert_T6__scratch":              "t6_reskd",
        "silverspoon-kd__reskd__bert_base_uncased__bert_T4_tiny__scratch":         "t4_reskd",
    }
    return _ns_by_run_name(rows, mapping, metrics_fn=_row_bert_pretrain)


def _row_bert_downstream(row):
    return {
        "matched":    row.get("matched"),
        "mismatched": row.get("mismatched"),
        "mem":        _parse_mb(row.get("peak_gpu_mb")),
        "sps":        row.get("steps_per_sec"),
        "time":       row.get("runtime_h"),
    }


def _ns_bert_downstream(runs_dir):
    """BERT Downstream (head-to-head vs TextBrewer). Mostly static-JSON-driven
    rows since reference baselines/teacher don't have run dirs."""
    rows = collect_bert_downstream(runs_dir)
    _prep_rows(rows, runs_dir)
    # Static-JSON rows are matched by display name, glob rows by run_name.
    name_map = {
        "Teacher (BERT-base)":             "teacher",
        "T6 baseline":                     "t6_baseline",
        "TextBrewer T6":                   "t6_textbrewer",
        "T6 HKD (single-stage)":           "t6_hkd_ft",   # historical "HKD → FT" label
        "T4-tiny baseline (no pretrain)":  "t4_baseline",
        "TextBrewer T4-tiny":              "t4_textbrewer",
        "T4-tiny HKD (single-stage)":      "t4_hkd_ft",
        "T4-tiny BKD → FT":                "t4_bkd_ft",
    }
    ns = {}
    for row in rows:
        nm = row.get("name")
        if nm in name_map:
            ns[name_map[nm]] = _row_bert_downstream(row)
    return ns


def _ns_results_summary(runs_dir):
    """Aggregate headline metrics for the 8-panel summary figure.
    Returns a nested dict keyed by panel: {vgg_bkd: {...}, relkd: {...}, ...}.
    Only the metric on each panel's x-axis (acc / MNLI / LAMBADA / MMLU) is
    needed; we extract it from the per-table namespaces."""
    vb = _ns_vision_bkd(runs_dir)
    rk = _ns_relkd(runs_dir)
    bp = _ns_bert_pretrain(runs_dir)
    bd = _ns_bert_downstream(runs_dir)
    gm = _ns_gpt2_multistage(runs_dir)
    gd = _ns_gpt2_deepseek(runs_dir)
    gl = _ns_gpt2_linear(runs_dir)
    qq = _ns_qwen3_qat(runs_dir)

    def acc(d, key):
        e = d.get(key)
        return e.get("acc") if e else None

    def lambada(d, key):
        e = d.get(key)
        return e.get("lambada") if e else None

    def matched(d, key):
        e = d.get(key)
        return e.get("matched") if e else None

    def mnli(d, key):
        e = d.get(key)
        return e.get("mnli") if e else None

    def mmlu(d, key):
        e = d.get(key)
        return e.get("mmlu") if e else None

    return {
        "vgg_bkd": {
            "teacher":   acc(vb, "teacher"),
            "baseline":  acc(vb, "baseline"),
            "bkd":       acc(vb, "bkd"),
            "reskd":     acc(vb, "reskd"),
            "bkd_ft":    acc(vb, "bkd_ft"),
            "bkd_hkd":   acc(vb, "bkd_hkd"),
            "bkd_reskd": acc(vb, "bkd_reskd"),
        },
        "relkd": {
            "teacher":   acc(rk, "teacher"),
            "baseline":  acc(rk, "baseline"),
            "reskd":     acc(rk, "reskd"),
            "hkd":       acc(rk, "hkd"),
            "relkd_d":   acc(rk, "relkd_d"),
            "relkd_a":   acc(rk, "relkd_a"),
            "relkd_da":  acc(rk, "relkd_da"),
        },
        "bert_pretrain": {
            "teacher":     0.8359,  # paper-anchored teacher MNLI (no run dir)
            "t6_baseline": mnli(bp, "t6_baseline"),
            "t6_bkd":      mnli(bp, "t6_bkd"),
            "t6_hkd":      mnli(bp, "t6_hkd"),
            "t6_reskd":    mnli(bp, "t6_reskd"),
            "t4_baseline": mnli(bp, "t4_baseline"),
            "t4_bkd":      mnli(bp, "t4_bkd"),
            "t4_hkd":      mnli(bp, "t4_hkd"),
            "t4_reskd":    mnli(bp, "t4_reskd"),
        },
        "bert_downstream": {
            "teacher":        matched(bd, "teacher"),
            "t6_baseline":    matched(bd, "t6_baseline"),
            "t6_textbrewer":  matched(bd, "t6_textbrewer"),
            "t6_hkd_ft":      matched(bd, "t6_hkd_ft"),
            "t4_baseline":    matched(bd, "t4_baseline"),
            "t4_textbrewer":  matched(bd, "t4_textbrewer"),
            "t4_hkd_ft":      matched(bd, "t4_hkd_ft"),
            "t4_bkd_ft":      matched(bd, "t4_bkd_ft"),
        },
        "gpt2_compression": {
            "teacher":      lambada(gm, "teacher"),
            "baseline":     lambada(gm, "baseline"),
            "bkd":          lambada(gm, "bkd"),
            "hkd":          lambada(gm, "hkd"),
            "reskd":        lambada(gm, "reskd"),
            "bkd_hkd":      lambada(gm, "bkd_hkd"),
            "bkd_hkd_lora": lambada(gm, "bkd_hkd_lora"),
        },
        "cross_arch": {
            "teacher":   lambada(gd, "teacher"),
            "baseline":  lambada(gd, "baseline"),
            "hkd":       lambada(gd, "hkd"),
            "reskd":     lambada(gd, "reskd"),
            "hkd_ft":    lambada(gd, "hkd_ft"),
            "reskd_ft":  lambada(gd, "reskd_ft"),
        },
        "linearization": {
            "teacher":    lambada(gl, "teacher"),
            "baseline":   lambada(gl, "baseline"),
            "bkd":        lambada(gl, "bkd"),
            "hkd":        lambada(gl, "hkd"),
            "bkd_hkd":    lambada(gl, "bkd_hkd"),
            "reskd":      lambada(gl, "reskd"),
            "bkd_reskd":  lambada(gl, "bkd_reskd"),
            "bkd_ft":     lambada(gl, "bkd_ft"),
        },
        "qat": {
            "bf16":  mmlu(qq, "bf16"),
            "ptq":   mmlu(qq, "int4_ptq"),
            "qat":   mmlu(qq, "int4_qat"),
            "bkd":   mmlu(qq, "int4_bkd"),
            "hkd":   mmlu(qq, "int4_hkd"),
            "reskd": mmlu(qq, "int4_reskd"),
        },
    }


def _delta_pct(sk, comp):
    """Percentage delta (SK relative to competitor); negative = SK better."""
    if sk is None or comp is None or comp == 0:
        return None
    return (sk - comp) / abs(comp) * 100


def _load_comparison_jsons(runs_dir):
    """Read all results/*_comparison.json into a single nested dict keyed by
    "<source_stem>/<dotted_section_path>"."""
    results_dir = Path(runs_dir).parent / "results"
    out = {}
    for path in sorted(results_dir.glob("*_comparison.json")):
        source = path.stem.replace("_comparison", "")
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        data.pop("config", None)
        def walk(d, prefix):
            if isinstance(d, dict):
                if any(k in d for k in ("wall_clock_sec", "tokens_per_sec", "peak_gpu_mb")):
                    out[prefix] = d
                else:
                    for k, v in d.items():
                        walk(v, f"{prefix}/{k}" if prefix else k)
        walk(data, source)
    return out


def _ns_framework(runs_dir):
    """tab:framework — head-to-head wall + memory at fixed batch.
    Each "section" pairs a competitor row with one or more SilverSpoon-KD rows."""
    j = _load_comparison_jsons(runs_dir)

    def metrics(entry):
        return {"wall": entry.get("wall_clock_sec"),
                "tput": entry.get("samples_per_sec"),
                "mem":  entry.get("peak_gpu_mb")}

    def section(comp_key, sk_keys):
        comp = j.get(comp_key, {})
        comp_m = metrics(comp)
        out = {"competitor": comp_m}
        for label, key in sk_keys.items():
            sk = j.get(key, {})
            m = metrics(sk)
            m["d_wall"] = _delta_pct(m["wall"], comp_m["wall"])
            m["d_mem"]  = _delta_pct(m["mem"],  comp_m["mem"])
            out[label] = m
        return out

    return {
        "vision":    section("multi_gpu/vision/torchdistill", {"sk": "multi_gpu/vision/silverspoon"}),
        "nlp_reskd": section("multi_gpu/nlp/textbrewer",      {"sk": "multi_gpu/nlp/silverspoon"}),
        "nlp_hkd":   section("textbrewer/textbrewer_fp32",
                             {"sk_fp32": "textbrewer/silverspoon_fp32_fused"}),
        "llm_hkd":   section("distillkit/distillkit_bf16",    {"sk": "distillkit/silverspoon_liger"}),
        "llm_lora":  section("torchtune/torchtune_kd",        {"sk": "torchtune/silverspoon_reskd_lora"}),
    }


def _ns_framework_oom(runs_dir):
    """tab:framework_oom — throughput frontier under DDP/replicated placement.
    All toolkits at their own DDP max batch. SK is replicated here (not sharded)
    so the comparison is apples-to-apples on placement."""
    j = _load_comparison_jsons(runs_dir)

    def metrics(entry):
        return {"bs":  entry.get("batch_size"),
                "tok": entry.get("tokens_per_sec"),
                "mem": entry.get("peak_gpu_mb")}

    def row(dk_key, tb_key, sk_key):
        dk = metrics(j.get(dk_key, {}))
        tb = metrics(j.get(tb_key, {}))
        sk = metrics(j.get(sk_key, {}))
        if dk["bs"] and sk["bs"]:
            sk["ratio"] = sk["bs"] / dk["bs"]
        return {"distillkit": dk, "textbrewer": tb, "sk": sk}

    return {
        "seq1024": row("multi_gpu/oom_boundary/distillkit_seq1024",
                       "multi_gpu/oom_boundary/textbrewer_seq1024",
                       "multi_gpu/oom_boundary/replicated_seq1024"),
        "seq2048": row("multi_gpu/oom_boundary/distillkit_seq2048",
                       "multi_gpu/oom_boundary/textbrewer_seq2048",
                       "multi_gpu/oom_boundary/replicated_seq2048"),
        "seq4096": row("multi_gpu/oom_boundary/distillkit_seq4096",
                       "multi_gpu/oom_boundary/textbrewer_seq4096",
                       "multi_gpu/oom_boundary/replicated_seq4096"),
    }


def _ns_framework_capability(runs_dir):
    """tab:framework_capability — teacher-size feasibility frontier.
    Each toolkit at its best placement: TB/DK replicated (their only option),
    SK sharded. Cells: pre-formatted '{bs} & {tok|thou} & {mem|thou}' or
    \\multicolumn{3}{c}{OOM} when the configuration cannot run."""
    j = _load_comparison_jsons(runs_dir)

    def cell(entry):
        if not entry or "error" in entry:
            return r"\multicolumn{3}{c}{\textit{OOM}}"
        bs = entry.get("batch_size")
        tok = entry.get("tokens_per_sec")
        mem = entry.get("peak_gpu_mb")
        return f"{bs} & {_filter_thou(tok)} & {_filter_thou(mem)}"

    def row(teacher):
        return {
            "dk": {"cell": cell(j.get(f"multi_gpu/teacher_scaling/distillkit_{teacher}", {}))},
            "tb": {"cell": cell(j.get(f"multi_gpu/teacher_scaling/textbrewer_{teacher}", {}))},
            "sk": {"cell": cell(j.get(f"multi_gpu/teacher_scaling/sk_sharded_{teacher}", {}))},
        }

    return {
        "qwen3_4B":  row("Qwen3-4B"),
        "qwen3_8B":  row("Qwen3-8B"),
        "qwen3_14B": row("Qwen3-14B"),
    }


# Registry: <template stem> → (namespace_builder, paper_table_label)
PAPER_TABLES = {
    "vision_bkd":      (_ns_vision_bkd,      "tab:vision_bkd"),
    "relkd":           (_ns_relkd,           "tab:relkd"),
    "gpt2_multistage": (_ns_gpt2_multistage, "tab:gpt2_multistage"),
    "gpt2_deepseek":   (_ns_gpt2_deepseek,   "tab:gpt2_deepseek"),
    "gpt2_linear":     (_ns_gpt2_linear,     "tab:gpt2_linear"),
    "qwen3_qat":       (_ns_qwen3_qat,       "tab:qwen3_qat"),
    "framework":       (_ns_framework,       "tab:framework"),
    "framework_capability": (_ns_framework_capability, "tab:framework_capability"),
    "bert_pretrain":        (_ns_bert_pretrain,        "tab:bert_pretrain"),
    "bert_downstream":      (_ns_bert_downstream,      "tab:bert_downstream"),
    "results_summary": (_ns_results_summary, "fig:results_summary"),
}


def render_paper_tables(runs_dir, output_dir, templates_dir=None):
    """Render every template in templates_dir whose stem is in PAPER_TABLES,
    writing .tex files to output_dir. Returns list of (stem, output_path)."""
    if templates_dir is None:
        templates_dir = os.path.join(output_dir, "_templates")
    if not os.path.isdir(templates_dir):
        raise FileNotFoundError(f"Templates dir not found: {templates_dir}")
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for stem, (builder, _label) in PAPER_TABLES.items():
        tmpl_path = os.path.join(templates_dir, f"{stem}.tex.tmpl")
        if not os.path.exists(tmpl_path):
            print(f"  SKIP (no template): {stem}", file=sys.stderr)
            continue
        with open(tmpl_path) as f:
            tmpl = f.read()
        ns = builder(runs_dir)
        rendered = render_template(tmpl, ns)
        out_path = os.path.join(output_dir, f"{stem}.tex")
        with open(out_path, "w") as f:
            f.write(rendered)
        written.append((stem, out_path))
    return written


# ── Toolkit comparisons ────────────────────────────────────────────────────

def collect_toolkit_comparisons(runs_dir):
    """Collect toolkit speed/memory comparison results from results/*.json.

    Returns a list of row dicts with raw floats for display by
    print_toolkit_table().  Each row carries '_source' (file stem),
    '_section' (JSON nesting path), and the raw metrics.
    """
    results_dir = Path(runs_dir).parent / "results"
    # Skip small-scale benchmarks (CIFAR-10/100, <500 MB GPU) — not meaningful
    _SKIP_BENCHMARKS = {"torchdistill_comparison", "torchdistill_resnet_cifar100_comparison"}
    # Skip entries that compare different paradigms — not apples-to-apples.
    # Keyed by (source_stem, entry_key) so we only skip specific contexts.
    _SKIP_ENTRIES = {
        # SK HKD vs torchdistill/TextBrewer response-based KD in DDP (different paradigm).
        # The single-GPU textbrewer HKD-to-HKD comparison IS fair and is kept.
        ("multi_gpu", "silverspoon_hkd"),
        # SK full-FT vs torchtune LoRA (different training strategy & model quality)
        ("torchtune", "silverspoon_reskd_no_chunk"),
        ("torchtune", "silverspoon_reskd"),
        ("torchtune", "silverspoon_reskd_liger"),
    }
    rows = []
    for path in sorted(results_dir.glob("*_comparison.json")):
        if path.stem in _SKIP_BENCHMARKS:
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        source = path.stem.replace("_comparison", "")
        data.pop("config", None)

        def _add_entry(section, key, entry):
            if not isinstance(entry, dict) or "error" in entry:
                return
            if (source, key) in _SKIP_ENTRIES:
                return
            if "wall_clock_sec" in entry or "samples_per_sec" in entry or "peak_gpu_mb" in entry:
                rows.append({
                    "_source": source,
                    "_section": section,
                    "_toolkit_label": entry.get("toolkit", entry.get("framework", key)),
                    "_raw_wall_sec": entry.get("wall_clock_sec"),
                    "_raw_samples_s": entry.get("samples_per_sec"),
                    "_raw_tokens_s": entry.get("tokens_per_sec"),
                    "_raw_gpu_mb": entry.get("peak_gpu_mb"),
                    "_raw_sec_step": entry.get("sec_per_step"),
                    "_batch_size": entry.get("batch_size"),
                    "_placement": entry.get("placement"),
                    # Flat "<section>/<toolkit>" name for consumers that key on it
                    "name": f"{section}/{entry.get('toolkit', entry.get('framework', key))}",
                })
            else:
                for sub_key, sub_entry in entry.items():
                    _add_entry(f"{section}/{key}" if section else key, sub_key, sub_entry)

        for key, entry in data.items():
            _add_entry(source, key, entry)
    return rows


def _is_sk_toolkit_entry(name):
    """Check if a toolkit benchmark row belongs to silverspoon-kd."""
    fw = name.rsplit("/", 1)[-1].lower()
    return "silverspoon-kd" in fw or fw.startswith("sk ")


# ── Competitor group mapping ────────────────────────────────────────────────
# Maps (source file stem, JSON section prefix) → display group.
# Entries from the same display group appear together under one separator.

_TOOLKIT_GROUPS = OrderedDict([
    # (source, section_contains) → (group_label, setting_label)
    # torchdistill — multi-GPU vision
    ("multi_gpu/vision",        ("vs torchdistill (ResNet50 / ImageNet, vision)", None)),
    # TextBrewer — single-GPU + multi-GPU NLP
    ("textbrewer",              ("vs TextBrewer (BERT → T6 / MNLI, encoder NLP)", None)),
    ("multi_gpu/nlp",           ("vs TextBrewer (BERT → T6 / MNLI, encoder NLP)", None)),
    # DistillKit — single-GPU + multi-GPU LLM
    ("distillkit",              ("vs DistillKit (Qwen3-4B → 0.6B / Dolma, LLM)", None)),
    ("multi_gpu/llm",           ("vs DistillKit (Qwen3-4B → 0.6B / Dolma, LLM)", None)),
    # torchtune — single-GPU LLM
    ("torchtune",               ("vs torchtune (Qwen3-4B → 0.6B / Dolma, LLM)", None)),
    # SK-only sections
    ("multi_gpu/placement",     ("Teacher Placement (silverspoon-kd only, Qwen3 LLM)", None)),
    # OOM Frontier kept last because it re-prints the column header (Tok/s,
    # BS× instead of Wall, Δ Mem). Any group that follows would render under
    # the wrong header.
    ("multi_gpu/oom_boundary",  ("OOM Frontier (Qwen3 LLM)", None)),
])


def _classify_toolkit_row(row):
    """Return the display group key for a toolkit row."""
    section = row.get("_section", "")
    source = row.get("_source", "")
    for pattern, (group_label, _) in _TOOLKIT_GROUPS.items():
        if pattern in section or pattern in source:
            return group_label
    return f"Other ({source})"


def _fmt_delta(sk_val, baseline_val):
    """Format a percentage delta: positive = SK worse, negative = SK better."""
    if sk_val is None or baseline_val is None or baseline_val == 0:
        return "—"
    pct = (sk_val - baseline_val) / abs(baseline_val) * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


def print_toolkit_table(title, rows, flag=False):
    """Print the toolkit comparison as a single table with group separators
    and per-metric delta columns."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    if not rows:
        print("  (no results)")
        return

    # Classify rows into display groups, ordered by _TOOLKIT_GROUPS so the
    # OOM Frontier group (which re-prints the column header) always appears
    # last. Without this, group order depends on row-encounter order which
    # varies with filesystem listing.
    canonical_order = []
    seen_labels = set()
    for _, (label, _) in _TOOLKIT_GROUPS.items():
        if label not in seen_labels:
            canonical_order.append(label)
            seen_labels.add(label)
    grouped = OrderedDict((label, []) for label in canonical_order)
    for r in rows:
        g = _classify_toolkit_row(r)
        grouped.setdefault(g, []).append(r)
    # Drop empty groups so the output stays clean.
    grouped = OrderedDict((k, v) for k, v in grouped.items() if v)

    # Column widths — fixed layout
    W_SETTING = 34
    W_FW = 18
    W_NUM = 10
    W_DELTA = 8

    # Header
    hdr = (f"{'Setting':<{W_SETTING}}  {'Toolkit':<{W_FW}}  "
           f"{'Wall (s)':>{W_NUM}}  {'Samp/s':>{W_NUM}}  {'GPU (MB)':>{W_NUM}}  "
           f"{chr(916)+' Wall':>{W_DELTA}}  {chr(916)+' Mem':>{W_DELTA}}")
    print(hdr)
    sep = "─" * len(hdr)
    print(sep)

    n_flagged = 0
    for group_label, group_rows in grouped.items():
        # OOM Frontier groups compare throughput (tokens/sec) instead of wall
        # time, because each row runs at a different max batch size — wall
        # time at different batch sizes is misleading.
        is_oom_group = "OOM Frontier" in group_label

        # Group separator — re-print column header for OOM Frontier since
        # it swaps Wall (s) → Tok/s and replaces Δ Mem with BS× (batch-size
        # advantage). Memory is intentionally maxed out on these runs, so a
        # memory delta would be misleading — the win is fitting more samples
        # in the same memory budget.
        if is_oom_group:
            oom_hdr = (f"{'Setting':<{W_SETTING}}  {'Toolkit':<{W_FW}}  "
                       f"{'Tok/s':>{W_NUM}}  {'Samp/s':>{W_NUM}}  {'GPU (MB)':>{W_NUM}}  "
                       f"{chr(916)+' Tput':>{W_DELTA}}  {'BS×':>{W_DELTA}}")
            print("─" * len(oom_hdr))
            print(oom_hdr)
            print("─" * len(oom_hdr))
        print(f"  ── {group_label} ──")

        # Within each group, identify sub-groups (by section prefix before last "/")
        # and find competitor baselines for delta computation.
        # For OOM Frontier, sub-group by sequence length so each DK baseline
        # is matched to the corresponding SK rows at the same seq length.
        sub_groups = OrderedDict()
        for r in group_rows:
            section = r.get("_section", "")
            if is_oom_group:
                m = re.search(r'seq=(\d+)', r.get("_toolkit_label", ""))
                if m:
                    section = f"{section}/seq{m.group(1)}"
            sub_groups.setdefault(section, []).append(r)

        for section, sub_rows in sub_groups.items():
            competitors = [r for r in sub_rows if not _is_sk_toolkit_entry(r["name"])]

            def _find_baseline(sk_row):
                """Find the matching competitor baseline for an SK row.
                Matches HKD entries to HKD baselines, response-based to
                response-based, etc. Falls back to the first competitor."""
                if not competitors:
                    return None
                sk_label = sk_row.get("_toolkit_label", "").lower()
                for c in competitors:
                    c_label = c.get("_toolkit_label", "").lower()
                    # Match by paradigm: both HKD or both non-HKD
                    if ("hkd" in sk_label) == ("hkd" in c_label):
                        return c
                return competitors[0]

            for r in sub_rows:
                fw_label = r.get("_toolkit_label", "?")
                is_sk = _is_sk_toolkit_entry(r["name"])

                # Derive setting label from toolkit label or section
                setting = fw_label
                # Strip the toolkit name to get just the setting description
                for strip in ["torchdistill ", "TextBrewer", "DistillKit", "torchtune",
                              "silverspoon-kd ", "sk "]:
                    if setting.lower().startswith(strip.lower()):
                        setting = setting[len(strip):].strip()
                        break
                if not setting or setting == fw_label:
                    # Fallback: use the section path after source
                    parts = section.split("/")
                    setting = parts[-1] if len(parts) > 1 else section

                # Short toolkit name
                if is_sk:
                    fw_short = "silverspoon-kd"
                else:
                    # Extract competitor name
                    for name in ["torchdistill", "TextBrewer", "DistillKit", "torchtune"]:
                        if name.lower() in fw_label.lower():
                            fw_short = name
                            break
                    else:
                        fw_short = fw_label[:W_FW]

                # Format metrics
                wall = f"{r['_raw_wall_sec']:.1f}" if r.get("_raw_wall_sec") else "—"
                sps = f"{r['_raw_samples_s']:.1f}" if r.get("_raw_samples_s") else "—"
                tps = f"{r['_raw_tokens_s']:.0f}" if r.get("_raw_tokens_s") else "—"
                mem = f"{r['_raw_gpu_mb']:.0f}" if r.get("_raw_gpu_mb") else "—"

                # Deltas (only for SK rows with a competitor baseline)
                baseline = _find_baseline(r) if is_sk else None
                if is_sk and baseline is not None:
                    d_wall = _fmt_delta(r.get("_raw_wall_sec"), baseline.get("_raw_wall_sec"))
                    d_mem = _fmt_delta(r.get("_raw_gpu_mb"), baseline.get("_raw_gpu_mb"))
                    # Throughput delta: higher is better, so invert the sign
                    # convention — positive means SK is faster.
                    d_tput = _fmt_delta(r.get("_raw_tokens_s"), baseline.get("_raw_tokens_s"))
                else:
                    d_wall = "—"
                    d_mem = "—"
                    d_tput = "—"

                if is_oom_group:
                    # OOM Frontier: show throughput (tok/s) + Δ Tput, plus BS×
                    # (batch-size advantage vs competitor at the same seq length).
                    # Each row runs at its own max batch size — the win is
                    # fitting more in the same memory budget, so a memory delta
                    # would mistake "uses the budget you gave it" for a regression.
                    b_tps = baseline.get("_raw_tokens_s") if baseline else None
                    tps_flagged = (flag and is_sk and b_tps is not None
                                   and r.get("_raw_tokens_s") and r["_raw_tokens_s"] < b_tps)
                    tps_s = f"{_ANSI_RED}{tps:>{W_NUM}}{_ANSI_RESET}" if tps_flagged else f"{tps:>{W_NUM}}"
                    sps_s = f"{sps:>{W_NUM}}"
                    mem_s = f"{mem:>{W_NUM}}"
                    # Throughput delta: "+" means SK faster (green), "-" slower (red)
                    if is_sk and d_tput != "—":
                        if d_tput.startswith("+"):
                            d_tput_s = f"\033[32m{d_tput:>{W_DELTA}}\033[0m"
                        else:
                            d_tput_s = f"{_ANSI_RED}{d_tput:>{W_DELTA}}{_ANSI_RESET}"
                            n_flagged += 1
                    else:
                        d_tput_s = f"{d_tput:>{W_DELTA}}"
                    # BS× advantage (SK batch size / competitor batch size)
                    sk_bs = r.get("_batch_size")
                    b_bs = baseline.get("_batch_size") if baseline else None
                    if is_sk and sk_bs and b_bs:
                        ratio = sk_bs / b_bs
                        bs_str = f"{ratio:.2f}×"
                        if ratio > 1.0:
                            bs_s = f"\033[32m{bs_str:>{W_DELTA}}\033[0m"
                        elif ratio < 1.0:
                            bs_s = f"{_ANSI_RED}{bs_str:>{W_DELTA}}{_ANSI_RESET}"
                        else:
                            bs_s = f"{bs_str:>{W_DELTA}}"
                    else:
                        bs_s = f"{'—':>{W_DELTA}}"
                    print(f"{setting:<{W_SETTING}}  {fw_short:<{W_FW}}  "
                          f"{tps_s}  {sps_s}  {mem_s}  {d_tput_s}  {bs_s}")
                else:
                    # Standard groups: wall time + delta
                    # Flag SK rows that are slower than competitor
                    b_wall = baseline.get("_raw_wall_sec") if baseline else None
                    wall_flagged = (flag and is_sk and b_wall is not None
                                    and r.get("_raw_wall_sec") and r["_raw_wall_sec"] > b_wall)

                    wall_s = f"{_ANSI_RED}{wall:>{W_NUM}}{_ANSI_RESET}" if wall_flagged else f"{wall:>{W_NUM}}"
                    sps_flagged = (flag and is_sk and baseline is not None
                                   and baseline.get("_raw_samples_s") and r.get("_raw_samples_s")
                                   and r["_raw_samples_s"] < baseline["_raw_samples_s"])
                    sps_s = f"{_ANSI_RED}{sps:>{W_NUM}}{_ANSI_RESET}" if sps_flagged else f"{sps:>{W_NUM}}"
                    mem_s = f"{mem:>{W_NUM}}"
                    # Color deltas: red if SK worse, green if SK better
                    if is_sk and d_wall != "—":
                        if d_wall.startswith("+"):
                            d_wall_s = f"{_ANSI_RED}{d_wall:>{W_DELTA}}{_ANSI_RESET}"
                            n_flagged += 1
                        else:
                            d_wall_s = f"\033[32m{d_wall:>{W_DELTA}}\033[0m"
                    else:
                        d_wall_s = f"{d_wall:>{W_DELTA}}"
                    if is_sk and d_mem != "—":
                        if d_mem.startswith("+"):
                            d_mem_s = f"{_ANSI_RED}{d_mem:>{W_DELTA}}{_ANSI_RESET}"
                            n_flagged += 1
                        else:
                            d_mem_s = f"\033[32m{d_mem:>{W_DELTA}}\033[0m"
                    else:
                        d_mem_s = f"{d_mem:>{W_DELTA}}"

                    print(f"{setting:<{W_SETTING}}  {fw_short:<{W_FW}}  "
                          f"{wall_s}  {sps_s}  {mem_s}  {d_wall_s}  {d_mem_s}")

        print()  # blank line between groups

    if flag and n_flagged > 0:
        print(f"  {_ANSI_RED}({n_flagged} delta(s) where silverspoon-kd underperforms){_ANSI_RESET}")


# ── Main ─────────────────────────────────────────────────────────────────────

GROUPS = {
    # Ordered to match paper experiment numbering (1-9)
    "vgg_compression": ("VGG Compression (VGG16 138M → VGG16-DS 3M)", collect_vision_cifar10,
                        {"name": "Run", "accuracy": "Accuracy"}),
    "vgg_relkd": ("VGG Relational KD (ResNet50 25M → VGG11-BN 9M)", collect_vision_cifar100,
                  {"name": "Run", "accuracy": "Accuracy", "step": "Steps"}),
    "bert_pretraining": ("BERT Pre-Training (BERT-base 110M → T6 67M / T4-tiny 14M)", collect_bert_encoder,
                         {"name": "Method", "train_steps": "Steps",
                          "eval_loss": "Align Loss (T↔S)", "mlm_ppl": "MLM PPL (data)",
                          "e2e_loss": "MLM Loss (data)", "mnli": "MNLI % (Stage 2)"}),
    "bert_downstream": ("BERT Downstream (BERT-base 110M, vs TextBrewer)", collect_bert_downstream,
                        {"name": "Run", "train_steps": "Steps",
                         "matched": "MNLI-m", "mismatched": "MNLI-mm"}),
    "gpt2_compression": ("GPT-2 Compression (124M → 112M / 96M)", collect_gpt2_distillation,
                         {"name": "Method", "train_steps": "Steps", "final_eval_loss": "KD Loss (T↔S)",
                          "wikitext_ppl": "PPL (data)", "lambada_acc": "LAMB Acc", "hellaswag": "HS(N)"}),
    "cross_arch": ("Cross-Architecture (GPT-2 124M → DeepSeek-V3 96M)", collect_gpt2_cross_arch,
                   {"name": "Method", "train_steps": "Steps", "final_eval_loss": "KD Loss (T↔S)",
                    "wikitext_ppl": "PPL (data)", "lambada_acc": "LAMB Acc", "hellaswag": "HS(N)"}),
    "linearization": ("GPT-2 Attention Linearization (GPT-2 124M → LoLCATs ~125M)", collect_gpt2_linearization,
                      {"name": "Method", "train_steps": "Steps", "final_eval_loss": "KD Loss (T↔S)",
                       "wikitext_ppl": "PPL (data)", "lambada_acc": "LAMB Acc", "hellaswag": "HS(N)"}),
    "qat": ("Qwen3 Quantization (Qwen3-1.7B BF16 → INT4/INT5/INT6/INT8)", collect_qat,
            {"name": "Method", "train_steps": "Steps", "final_eval_loss": "KD Loss (T↔S)",
             "mmlu": "MMLU", "arc_easy": "ARC-E", "arc_challenge": "ARC-C(N)",
             "hellaswag": "HS(N)", "wikitext_ppl": "PPL (data)"}),
    "toolkit": ("Toolkit Benchmarking", collect_toolkit_comparisons,
                  {"name": "Benchmark/Toolkit", "wall_sec": "Wall (s)", "sec_step": "s/step",
                   "samples_s": "Samples/s", "gpu_mb": "GPU (MB)"}),
}


def inject_wandb_columns(rows, headers, name_key="name", runs_dir=None):
    """Add Peak GPU, steps/sec, and runtime columns from WandB cache."""
    headers["peak_gpu_mb"] = "Peak GPU"
    headers["steps_per_sec"] = "Steps/s"
    headers["runtime_h"] = "Time (h)"
    for r in rows:
        # Prefer explicit run_name (original experiment name) over display name
        run_name = r.get("run_name") or r.get(name_key, r.get("run", ""))
        mb = get_peak_gpu_mb(run_name, runs_dir=runs_dir)
        sps = get_steps_per_sec(run_name)
        rt = get_runtime_h(run_name)
        r["peak_gpu_mb"] = f"{mb} MB" if mb else None
        r["steps_per_sec"] = sps
        r["runtime_h"] = rt


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment results")
    parser.add_argument("--runs-dir", default=None, help="Path to runs/ directory")
    parser.add_argument("--group", default=None, choices=list(GROUPS.keys()), help="Show only this group")
    parser.add_argument("--latex", action="store_true", help="Emit LaTeX tables")
    parser.add_argument("--no-status", action="store_true", help="Skip the run status overview")
    parser.add_argument("--status-only", action="store_true", help="Show only run status, no result tables")
    parser.add_argument("--wandb", action="store_true",
                        help="Add GPU memory + throughput + runtime columns (uses cached WandB data)")
    parser.add_argument("--wandb-refresh", action="store_true",
                        help="Re-fetch WandB data from API (slow, ~1 min) instead of using cache")
    parser.add_argument("--memory", action="store_true",
                        help="Alias for --wandb")
    parser.add_argument("--flag", action="store_true",
                        help="Flag cells red that fall below expected thresholds")
    parser.add_argument("--paper-tables", nargs="?", const="DEFAULT", default=None,
                        metavar="OUTPUT_DIR",
                        help="Render paper-ready LaTeX tables from templates "
                             "(paper/generated/_templates/*.tex.tmpl) into OUTPUT_DIR. "
                             "If OUTPUT_DIR is omitted, defaults to "
                             "paper/generated/ next to the runs/ directory.")
    args = parser.parse_args()

    if args.runs_dir:
        runs_dir = args.runs_dir
    else:
        project_root = find_project_root()
        runs_dir = str(project_root / "runs")

    if not os.path.isdir(runs_dir):
        print(f"Error: runs directory not found: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    # Load WandB metrics cache if requested. --paper-tables also needs WandB
    # data to populate the system-metric columns (Peak Mem, Steps/s, Time).
    use_wandb = args.wandb or args.memory or args.wandb_refresh or args.paper_tables is not None
    if use_wandb:
        if args.wandb_refresh:
            _refresh_wandb_cache(runs_dir=runs_dir)
        else:
            load_wandb_cache()

    # Paper-tables mode: render LaTeX templates and exit.
    if args.paper_tables is not None:
        if args.paper_tables == "DEFAULT":
            output_dir = str(Path(runs_dir).parent / "paper" / "generated")
        else:
            output_dir = args.paper_tables
        templates_dir = os.path.join(output_dir, "_templates")
        print(f"Rendering paper tables from {templates_dir} → {output_dir}")
        written = render_paper_tables(runs_dir, output_dir, templates_dir)
        for stem, path in written:
            print(f"  ✓ {stem} → {path}")
        if not written:
            print("  (no templates rendered)", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

    # Result tables
    if not args.status_only:
        groups_to_show = [args.group] if args.group else list(GROUPS.keys())
        for group_name in groups_to_show:
            title, collector, headers = GROUPS[group_name]
            headers = dict(headers)  # copy so we can mutate
            rows = collector(runs_dir)
            # Toolkit comparison uses its own dedicated display
            if group_name == "toolkit":
                if args.latex:
                    print_latex_table(title, headers, rows, label=group_name)
                else:
                    print_toolkit_table(title, rows, flag=args.flag)
                continue
            inject_teacher_row(rows, group_name, runs_dir)
            inject_trainer_state_columns(rows, runs_dir)
            # Only show columns that are in the headers definition
            inject_wandb_columns(rows, headers, runs_dir=runs_dir)
            if args.latex:
                print_latex_table(title, headers, rows, label=group_name)
            else:
                print_table(title, headers, rows, group_key=group_name,
                            flag_thresholds=args.flag)

    # Run status overview (terminal only, not in LaTeX mode)
    if not args.latex and not args.no_status:
        print_run_status(runs_dir)

    if not args.latex:
        print(f"\n{'=' * 70}")
        print(f"  Runs directory: {runs_dir}")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
