"""
Benchmark: silverspoon-kd vs torchtune — Qwen3 decoder LLM knowledge distillation.

Runs both toolkits with comparable settings:
  - Teacher: Qwen3 (frozen, bf16) — default: Qwen3-4B
  - Student: smaller Qwen3 — default: Qwen3-0.6B
      - torchtune: LoRA adapters (rank=64, alpha=128) on q/v/output + MLP
      - silverspoon-kd: full fine-tuning from random init
  - Data: Dolma text (tokenized, max_length=1024)
  - Duration: fixed number of training steps (default 1000)

Runs performed:
  1. torchtune KD         — ForwardKLLoss (LoRA student), kd_ratio=0.5
  2. silverspoon-kd full  — ResponseBasedDistiller (full fine-tuning, T=2, α=0.5)
  3. silverspoon-kd LoRA  — ResponseBasedDistiller (LoRA student, T=2, α=0.5) — apples-to-apples

Measures per run:
  - Final training loss
  - Wall-clock time (seconds)
  - Throughput (samples/sec)
  - Peak GPU memory (MB)

Key caveats (documented for paper):
  1. LoRA vs full fine-tuning: torchtune trains ~2-3% of parameters; silverspoon-kd trains 100%
  2. No temperature: torchtune's Forward KL has no temperature scaling
  3. No intermediate distillation: torchtune only does logit-level KD

Usage:
    # Default (4B → 0.6B, fits on single 40GB GPU):
    CUDA_VISIBLE_DEVICES=0 python scripts/toolkit_comparison/compare_torchtune.py [--num_steps 1000]

    # Full paper reproduction (8B → 4B, needs ~80GB):
    CUDA_VISIBLE_DEVICES=0,1 python scripts/toolkit_comparison/compare_torchtune.py \\
        --teacher Qwen/Qwen3-8B --student Qwen/Qwen3-4B --num_steps 1000
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
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
    import glob

    # Try loading from existing tokenized cache
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")
    cache_pattern = os.path.join(cache_dir, "dolma_tokenized_*")
    cache_paths = sorted(glob.glob(cache_pattern), key=os.path.getmtime, reverse=True)

    if cache_paths:
        print(f"  Loading from tokenized cache: {cache_paths[0]}")
        ds = HFDataset.load_from_disk(cache_paths[0])
        if len(ds) > max_samples:
            ds = ds.select(range(max_samples))
        # Replace None values (sparse Arrow int columns) with 0
        rows = []
        for row in ds["input_ids"]:
            rows.append([x if x is not None else 0 for x in row])
        input_ids = torch.tensor(rows, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        print(f"  Loaded {len(input_ids)} samples of length {input_ids.shape[1]}")
        return TokenizedTextDataset(input_ids, attention_mask, labels)

    # Fallback: download and tokenize from Dolma streaming
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


def ensure_model_downloaded(model_name):
    """Ensure HuggingFace model weights are available locally."""
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(model_name)
    except Exception as e:
        print(f"  Warning: could not download {model_name}: {e}")


# ── torchtune run ─────────────────────────────────────────────────────────────

def run_torchtune_kd(teacher_name, student_name, train_dataset, num_steps, lr, batch_size, device):
    """torchtune KD: ForwardKLLoss with LoRA student.

    Uses HF models for loading (avoids torchtune weight-key mapping), but
    torchtune's ForwardKLLoss for the actual distillation loss computation.
    """
    from torchtune.modules.loss import ForwardKLLoss

    print(f"  Loading teacher ({teacher_name}) via HuggingFace...")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"  Loading student ({student_name}) with LoRA via PEFT...")
    from peft import LoraConfig, get_peft_model
    student_base = AutoModelForCausalLM.from_pretrained(
        student_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    student = get_peft_model(student_base, lora_config)
    student.print_trainable_parameters()

    # Use torchtune's ForwardKLLoss (non-chunked variant, works with flat tensors)
    kd_loss_fn = ForwardKLLoss(ignore_index=-100)
    ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    kd_ratio = 0.5

    # Training setup
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
    )

    student.train()

    reset_gpu_stats(device)
    start = time.perf_counter()

    step = 0
    total_loss = 0.0
    while step < num_steps:
        for batch in train_loader:
            if step >= num_steps:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Student forward
            student_out = student(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_out.logits

            # Teacher forward
            with torch.no_grad():
                teacher_out = teacher(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_out.logits

            # Shift for causal LM: predict next token
            shift_student = student_logits[..., :-1, :].contiguous()
            shift_teacher = teacher_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten
            B, S, V = shift_student.shape
            flat_student = shift_student.view(B * S, V)
            flat_teacher = shift_teacher.view(B * S, V)
            flat_labels = shift_labels.view(B * S)

            # torchtune's ForwardKLLoss + CE
            kd_loss = kd_loss_fn(flat_student, flat_teacher, flat_labels)
            ce_loss = ce_loss_fn(flat_student.float(), flat_labels)
            loss = (1.0 - kd_ratio) * ce_loss + kd_ratio * kd_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

            total_loss = loss.item()
            step += 1

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_mem = get_peak_memory_mb(device)

    del teacher, student, student_base
    gc.collect()
    torch.cuda.empty_cache()

    return total_loss, elapsed, peak_mem


# ── silverspoon-kd run ────────────────────────────────────────────────────────

def _run_silverspoon_reskd(teacher_name, student_name, train_dataset, num_steps, lr, batch_size,
                          device, use_lora=False, loss_mode="chunked"):
    """silverspoon-kd response-based KD: ResponseBasedDistiller.

    Args:
        loss_mode: "no_chunk" (full logit KL), "chunked" (chunk_size=256),
                   or "liger" (fused lm_head + KL kernel).
    """
    from silverspoon_kd import ResponseBasedDistiller, TrainingArguments
    from silverspoon_kd.losses.kl import kl_divergence_loss

    print(f"  Loading teacher ({teacher_name}) via HuggingFace...")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    if use_lora:
        print(f"  Loading student ({student_name}) with LoRA via PEFT...")
        from peft import LoraConfig, get_peft_model
        student = AutoModelForCausalLM.from_pretrained(
            student_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device)
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        student = get_peft_model(student, lora_config)
        student.print_trainable_parameters()
    else:
        print(f"  Loading student ({student_name}) from random init...")
        from transformers import AutoConfig
        student_config = AutoConfig.from_pretrained(student_name, trust_remote_code=True)
        student = AutoModelForCausalLM.from_config(student_config).to(torch.bfloat16).to(device)

    tag = "sk_reskd_lora" if use_lora else "sk_reskd_full"
    output_dir = f"/tmp/{tag}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        run_name=tag,
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
    if loss_mode == "liger":
        soft_loss_fn = kl_divergence_loss(temperature=2.0)
        distiller_kwargs["use_liger_kernel"] = True
        distiller_kwargs["output_head_layer"] = "lm_head"
    elif loss_mode == "chunked":
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


def run_silverspoon_reskd(teacher_name, student_name, train_dataset, num_steps, lr, batch_size, device,
                          loss_mode="chunked"):
    """silverspoon-kd response-based KD with full fine-tuning."""
    return _run_silverspoon_reskd(teacher_name, student_name, train_dataset, num_steps, lr, batch_size,
                                 device, use_lora=False, loss_mode=loss_mode)


def run_silverspoon_reskd_lora(teacher_name, student_name, train_dataset, num_steps, lr, batch_size, device,
                               loss_mode="chunked"):
    """silverspoon-kd response-based KD with LoRA (apples-to-apples vs torchtune)."""
    return _run_silverspoon_reskd(teacher_name, student_name, train_dataset, num_steps, lr, batch_size,
                                 device, use_lora=True, loss_mode=loss_mode)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="torchtune vs silverspoon-kd benchmark")
    parser.add_argument("--num_steps", type=int, default=1000, help="Training steps per run")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--teacher", type=str, default="Qwen/Qwen3-4B",
                        help="Teacher model name (default: Qwen/Qwen3-4B)")
    parser.add_argument("--student", type=str, default="Qwen/Qwen3-0.6B",
                        help="Student model name (default: Qwen/Qwen3-0.6B)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Teacher: {args.teacher}, Student: {args.student}")
    print(f"Steps: {args.num_steps}, Batch: {args.batch_size}, LR: {args.lr}")
    print()

    # Ensure models are downloaded
    print("Ensuring model weights are available...")
    ensure_model_downloaded(args.teacher)
    ensure_model_downloaded(args.student)

    # Prepare tokenizer and data (use student tokenizer — Qwen3 family shares vocab)
    print("Preparing tokenizer and data...")
    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    train_dataset = prepare_dolma_data(tokenizer, max_length=args.max_length, max_samples=args.max_samples)

    all_results = {}

    # 1. torchtune KD
    print(f"\n{'=' * 60}")
    print("  torchtune KD (ForwardKL, LoRA student)")
    print(f"{'=' * 60}")
    try:
        tt_loss, tt_elapsed, tt_mem = run_torchtune_kd(
            args.teacher, args.student, train_dataset,
            args.num_steps, args.lr, args.batch_size, device,
        )
        total_samples = args.num_steps * args.batch_size
        all_results["torchtune_kd"] = {
            "toolkit": "torchtune KD (ForwardKL, LoRA)",
            "teacher": args.teacher,
            "student": args.student,
            "final_loss": round(tt_loss, 4),
            "wall_clock_sec": round(tt_elapsed, 2),
            "sec_per_step": round(tt_elapsed / args.num_steps, 4),
            "samples_per_sec": round(total_samples / tt_elapsed, 1),
            "peak_gpu_mb": round(tt_mem, 0),
            "student_type": "LoRA (rank=64, ~3% params)",
            "temperature": "N/A (no temperature in ForwardKL)",
        }
        print(json.dumps(all_results["torchtune_kd"], indent=2))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  torchtune run failed: {e}")
        all_results["torchtune_kd"] = {"error": str(e)}

    gc.collect()
    torch.cuda.empty_cache()

    # 2-4. silverspoon-kd full fine-tuning (three loss modes)
    sk_full_modes = [
        ("no_chunk", "silverspoon-kd ReSKD (full FT, no chunking)",    "silverspoon_reskd_no_chunk"),
        ("chunked",  "silverspoon-kd ReSKD (full FT, chunked KL)",     "silverspoon_reskd"),
        ("liger",    "silverspoon-kd ReSKD (full FT, Liger kernel)",   "silverspoon_reskd_liger"),
    ]
    for mode, label, key in sk_full_modes:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")
        try:
            sk_loss, sk_elapsed, sk_mem = run_silverspoon_reskd(
                args.teacher, args.student, train_dataset,
                args.num_steps, args.lr, args.batch_size, device,
                loss_mode=mode,
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
                "student_type": "Full fine-tuning (100% params, random init)",
                "temperature": 2.0,
            }
            print(json.dumps(all_results[key], indent=2))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  {label} failed: {e}")
            all_results[key] = {"error": str(e)}

        gc.collect()
        torch.cuda.empty_cache()

    # 5. silverspoon-kd LoRA (apples-to-apples vs torchtune, chunked only)
    print(f"\n{'=' * 60}")
    print("  silverspoon-kd response-based KD (LoRA, T=2, α=0.5) — apples-to-apples")
    print(f"{'=' * 60}")
    try:
        sk_lora_loss, sk_lora_elapsed, sk_lora_mem = run_silverspoon_reskd_lora(
            args.teacher, args.student, train_dataset,
            args.num_steps, args.lr, args.batch_size, device,
        )
        total_samples = args.num_steps * args.batch_size
        all_results["silverspoon_reskd_lora"] = {
            "toolkit": "silverspoon-kd response-based KD (LoRA, T=2, α=0.5)",
            "teacher": args.teacher,
            "student": args.student,
            "final_loss": round(sk_lora_loss, 4),
            "wall_clock_sec": round(sk_lora_elapsed, 2),
            "sec_per_step": round(sk_lora_elapsed / args.num_steps, 4),
            "samples_per_sec": round(total_samples / sk_lora_elapsed, 1),
            "peak_gpu_mb": round(sk_lora_mem, 0),
            "student_type": "LoRA (rank=64, ~3% params, pretrained init)",
            "temperature": 2.0,
        }
        print(json.dumps(all_results["silverspoon_reskd_lora"], indent=2))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  silverspoon-kd LoRA run failed: {e}")
        all_results["silverspoon_reskd_lora"] = {"error": str(e)}

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print(f"  COMPARISON SUMMARY ({args.num_steps} steps, batch={args.batch_size})")
    print(f"  Teacher: {args.teacher} → Student: {args.student}")
    print(f"{'=' * 90}")
    print(f"{'Configuration':<50} {'Loss':>8} {'Time (s)':>10} {'Samp/sec':>10} {'GPU (MB)':>10}")
    print("-" * 90)
    for key, r in all_results.items():
        if key == "config":
            continue
        if "error" in r:
            print(f"{key:<50} {'ERROR':>8} {'—':>10} {'—':>10} {'—':>10}")
        else:
            print(
                f"{r['toolkit']:<50} {r['final_loss']:>8.4f} "
                f"{r['wall_clock_sec']:>10.1f} {r['samples_per_sec']:>10.1f} "
                f"{r['peak_gpu_mb']:>10.0f}"
            )

    print("\n  NOTES:")
    print("  - Runs 1 vs 3 are apples-to-apples: same LoRA config, same student init")
    print("  - Run 2 (full fine-tuning) trains 100% of params from random init")
    print("  - torchtune's Forward KL uses raw logits (no temperature scaling)")
    print("  - silverspoon-kd uses temperature=2.0 for softer probability matching")

    all_results["config"] = vars(args)
    out_path = args.output or str(PROJECT_DIR / "results" / "torchtune_comparison.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
