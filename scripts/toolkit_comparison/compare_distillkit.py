"""
Benchmark: silverspoon-kd vs DistillKit — single-GPU LLM knowledge distillation.

Runs both toolkits with comparable settings on a single GPU:
  - Teacher: Qwen3-4B (frozen, bf16)
  - Student: Qwen3-0.6B (bf16, from pretrained)
  - Data: Dolma text (tokenized, max_length=1024)
  - Duration: fixed number of training steps (default 500)
  - Loss: KL divergence (weight=0.5, T=2) + cross-entropy (weight=0.5)

Both toolkits use:
  - bfloat16 precision
  - Gradient checkpointing
  - AdamW optimizer with cosine scheduler

Measures per run:
  - Wall-clock time (seconds)
  - Throughput (samples/sec)
  - Peak GPU memory (MB)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/toolkit_comparison/compare_distillkit.py
    CUDA_VISIBLE_DEVICES=0 python scripts/toolkit_comparison/compare_distillkit.py --num_steps 500
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))


# ── Shared utilities ──────────────────────────────────────────────────────────

def reset_gpu_stats(device):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)


def get_peak_memory_mb(device):
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1024 / 1024


class TokenizedTextDataset(Dataset):
    """Simple dataset wrapping tokenized input_ids/attention_mask/labels."""
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def prepare_dolma_data(tokenizer, max_length=1024, max_samples=5000):
    """Load Dolma data from tokenized cache or streaming download."""
    from datasets import Dataset as HFDataset
    import glob as glob_mod

    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")
    cache_pattern = os.path.join(cache_dir, "dolma_tokenized_*")
    cache_paths = sorted(glob_mod.glob(cache_pattern), key=os.path.getmtime, reverse=True)

    if cache_paths:
        print(f"  Loading from tokenized cache: {cache_paths[0]}")
        ds = HFDataset.load_from_disk(cache_paths[0])
        if len(ds) > max_samples:
            ds = ds.select(range(max_samples))
        rows = []
        for row in ds["input_ids"]:
            rows.append([x if x is not None else 0 for x in row])
        input_ids = torch.tensor(rows, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        print(f"  Loaded {len(input_ids)} samples of length {input_ids.shape[1]}")
        return TokenizedTextDataset(input_ids, attention_mask, labels)

    print("  No tokenized cache found, downloading Dolma subset...")
    from datasets import load_dataset
    ds = load_dataset("allenai/dolma3_pool", split="train", streaming=True)

    all_ids = []
    eos_id = tokenizer.eos_token_id
    for i, sample in enumerate(ds):
        if len(all_ids) // max_length >= max_samples:
            break
        text = sample.get("text", "")
        if len(text) > 100:
            encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
            all_ids.extend(encoded)
            all_ids.append(eos_id)

    n = min(len(all_ids) // max_length, max_samples)
    all_ids = all_ids[: n * max_length]
    chunks = [all_ids[i * max_length : (i + 1) * max_length] for i in range(n)]

    input_ids = torch.tensor(chunks, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    print(f"  Prepared {len(chunks)} samples of length {max_length}")
    return TokenizedTextDataset(input_ids, attention_mask, labels)


# ── DistillKit run ───────────────────────────────────────────────────────────

def _to_hf_dataset(train_dataset):
    """Convert a TokenizedTextDataset to an HF Dataset (needed by TRL/DistillKit)."""
    from datasets import Dataset as HFDataset
    return HFDataset.from_dict({
        "input_ids": train_dataset.input_ids.tolist(),
        "attention_mask": train_dataset.attention_mask.tolist(),
        "labels": train_dataset.labels.tolist(),
    })


def run_distillkit(teacher_name, student_name, train_dataset, num_steps, lr, batch_size, device):
    """DistillKit: KL + CE distillation with DistillationTrainer (single GPU)."""
    from distillkit.main import do_distill  # noqa: F401 — triggers correct import order
    from distillkit.trainer import DistillationTrainer
    from distillkit.signals import OnlineSignalSource
    from distillkit.configuration import (
        DistillationRunConfig, DatasetConfiguration, LocalDataset,
        TeacherModelConfig, LossFunctionConfig, LossFunction,
    )
    from trl import SFTConfig

    # DistillKit (TRL-based) requires an HF Dataset with column_names
    hf_dataset = _to_hf_dataset(train_dataset)

    tokenizer = AutoTokenizer.from_pretrained(student_name, trust_remote_code=True)

    print(f"  Loading teacher ({teacher_name})...")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"  Loading student ({student_name})...")
    student = AutoModelForCausalLM.from_pretrained(
        student_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)

    signal = OnlineSignalSource(teacher, vocab_size=tokenizer.vocab_size)

    config = DistillationRunConfig(
        model=student_name,
        dataset=DatasetConfiguration(train_dataset=LocalDataset(disk_path="/tmp/dummy")),
        teacher=TeacherModelConfig(path=teacher_name),
        sequence_length=1024,
        output_path="/tmp/dk_single_gpu",
        loss_functions=[
            LossFunctionConfig(function=LossFunction.KL, weight=0.5, temperature=2.0),
            LossFunctionConfig(function=LossFunction.CROSS_ENTROPY, weight=0.5),
        ],
        trust_remote_code=True,
    )
    sft_args = SFTConfig(
        output_dir="/tmp/dk_single_gpu",
        max_steps=num_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        bf16=True,
        report_to="none",
        logging_steps=num_steps,
        save_strategy="no",
        eval_strategy="no",
        disable_tqdm=True,
        gradient_checkpointing=False,
        dataset_text_field=None,
    )

    trainer = DistillationTrainer(
        model=student, config=config, signal_source=signal,
        true_vocab_size=tokenizer.vocab_size, args=sft_args,
        train_dataset=hf_dataset, processing_class=tokenizer,
    )

    reset_gpu_stats(device)
    start = time.perf_counter()
    result = trainer.train()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_mem = get_peak_memory_mb(device)

    final_loss = result.training_loss if hasattr(result, "training_loss") else 0.0

    del teacher, student, trainer, signal
    gc.collect()
    torch.cuda.empty_cache()

    return final_loss, elapsed, peak_mem


# ── silverspoon-kd run ────────────────────────────────────────────────────────

def run_silverspoon(teacher_name, student_name, train_dataset, num_steps, lr, batch_size, device,
                    mode="no_chunk"):
    """silverspoon-kd: ResponseBasedDistiller with KL + CE (single GPU).

    Args:
        mode: Loss computation strategy:
            "no_chunk" — full logit KL (no chunking, speed-fair vs competitors)
            "chunked"  — chunked KL (chunk_size=256, memory-efficient fallback)
            "liger"    — Liger fused linear kernel (fused lm_head + KL, best of both)
    """
    from silverspoon_kd import ResponseBasedDistiller, TrainingArguments
    from silverspoon_kd.losses.kl import kl_divergence_loss

    print(f"  Loading teacher ({teacher_name})...")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"  Loading student ({student_name}) from pretrained...")
    student = AutoModelForCausalLM.from_pretrained(
        student_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)

    output_dir = f"/tmp/sk_distillkit_cmp_{mode}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        run_name=f"sk_distillkit_cmp_{mode}",
        max_steps=num_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        report_to="none",
        logging_steps=9999,
        save_strategy="no",
        eval_strategy="no",
        disable_tqdm=True,
        alpha=0.5,
        auto_dtype_match=True,
        gradient_checkpointing=False,
    )

    def prepare_teacher_inputs(inputs):
        result = {"input_ids": inputs["input_ids"]}
        if "attention_mask" in inputs:
            result["attention_mask"] = inputs["attention_mask"]
        return result

    # Select loss computation strategy
    distiller_kwargs = {}
    if mode == "liger":
        soft_loss_fn = kl_divergence_loss(temperature=2.0)  # not used, but required
        distiller_kwargs["use_liger_kernel"] = True
        distiller_kwargs["output_head_layer"] = "lm_head"
    elif mode == "chunked":
        soft_loss_fn = kl_divergence_loss(temperature=2.0, chunk_size=256)
    else:  # no_chunk
        soft_loss_fn = kl_divergence_loss(temperature=2.0, chunk_size=0)

    trainer = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        train_dataset=train_dataset,
        args=training_args,
        soft_loss_fn=soft_loss_fn,
        prepare_teacher_inputs=prepare_teacher_inputs,
        **distiller_kwargs,
    )

    reset_gpu_stats(device)
    start = time.perf_counter()
    train_result = trainer.train()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_mem = get_peak_memory_mb(device)

    final_loss = train_result.training_loss if hasattr(train_result, "training_loss") else 0.0

    del teacher, student, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return final_loss, elapsed, peak_mem


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DistillKit vs silverspoon-kd single-GPU benchmark")
    parser.add_argument("--num_steps", type=int, default=500, help="Training steps per run")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--teacher", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--student", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Teacher: {args.teacher}, Student: {args.student}")
    print(f"Steps: {args.num_steps}, Batch: {args.batch_size}, LR: {args.lr}")
    print()

    # Prepare tokenizer and data
    print("Preparing tokenizer and data...")
    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    train_dataset = prepare_dolma_data(tokenizer, max_length=args.max_length, max_samples=args.max_samples)

    all_results = {}

    # 1. DistillKit
    print(f"\n{'=' * 60}")
    print("  DistillKit (KL + CE, bf16, gradient checkpointing)")
    print(f"{'=' * 60}")
    try:
        dk_loss, dk_elapsed, dk_mem = run_distillkit(
            args.teacher, args.student, train_dataset,
            args.num_steps, args.lr, args.batch_size, device,
        )
        total_samples = args.num_steps * args.batch_size
        all_results["distillkit_bf16"] = {
            "toolkit": "DistillKit (bf16, gradient ckpt)",
            "teacher": args.teacher,
            "student": args.student,
            "final_loss": round(dk_loss, 4),
            "wall_clock_sec": round(dk_elapsed, 2),
            "sec_per_step": round(dk_elapsed / args.num_steps, 4),
            "samples_per_sec": round(total_samples / dk_elapsed, 1),
            "peak_gpu_mb": round(dk_mem, 0),
        }
        print(json.dumps(all_results["distillkit_bf16"], indent=2))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  DistillKit run failed: {e}")
        all_results["distillkit_bf16"] = {"error": str(e)}

    gc.collect()
    torch.cuda.empty_cache()

    # 2-4. silverspoon-kd (three loss computation modes)
    sk_modes = [
        ("no_chunk", "silverspoon-kd (bf16, no chunking)", "silverspoon_no_chunk"),
        ("chunked",  "silverspoon-kd (bf16, chunked KL)",  "silverspoon_chunked"),
        ("liger",    "silverspoon-kd (bf16, Liger kernel)", "silverspoon_liger"),
    ]
    for mode, label, key in sk_modes:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")
        try:
            sk_loss, sk_elapsed, sk_mem = run_silverspoon(
                args.teacher, args.student, train_dataset,
                args.num_steps, args.lr, args.batch_size, device,
                mode=mode,
            )
            total_samples = args.num_steps * args.batch_size
            all_results[key] = {
                "toolkit": label,
                "teacher": args.teacher,
                "student": args.student,
                "final_loss": round(sk_loss, 4),
                "wall_clock_sec": round(sk_elapsed, 2),
                "sec_per_step": round(sk_elapsed / args.num_steps, 4),
                "samples_per_sec": round(total_samples / sk_elapsed, 1),
                "peak_gpu_mb": round(sk_mem, 0),
            }
            print(json.dumps(all_results[key], indent=2))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  {label} failed: {e}")
            all_results[key] = {"error": str(e)}

        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  COMPARISON SUMMARY ({args.num_steps} steps, batch={args.batch_size})")
    print(f"  Teacher: {args.teacher} → Student: {args.student}")
    print(f"{'=' * 80}")
    print(f"{'Toolkit':<45} {'Loss':>8} {'Time (s)':>10} {'Samp/sec':>10} {'GPU (MB)':>10}")
    print("-" * 85)
    for key, r in all_results.items():
        if key == "config":
            continue
        if "error" in r:
            print(f"{key:<45} {'ERROR':>8} {chr(8212):>10} {chr(8212):>10} {chr(8212):>10}")
        else:
            print(
                f"{r['toolkit']:<45} {r['final_loss']:>8.4f} "
                f"{r['wall_clock_sec']:>10.1f} {r['samples_per_sec']:>10.1f} "
                f"{r['peak_gpu_mb']:>10.0f}"
            )

    all_results["config"] = vars(args)
    out_path = args.output or str(PROJECT_DIR / "results" / "distillkit_comparison.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
