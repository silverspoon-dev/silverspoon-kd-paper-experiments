"""
Benchmark: silverspoon-kd vs TextBrewer — logit KD on MNLI with BERT T6.

Runs both toolkits with identical:
  - Teacher: our fine-tuned BERT-base-cased MNLI checkpoint
  - Student: 6-layer BERT (T6) initialized from teacher layers [1,3,5,7,9,11]
  - Data: MNLI training set, batch_size=128, max_length=128
  - Hyperparameters: lr=1e-4, temperature=4, linear warmup 10%
  - Duration: fixed number of training steps (default 1000)

Measures per toolkit:
  - Wall-clock time (total and per step)
  - Throughput (samples/sec)
  - Peak GPU memory

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/toolkit_comparison/compare_textbrewer.py [--num_steps 1000]
"""
import argparse
import gc
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
TEACHER_PATH = PROJECT_DIR / "runs" / "hf__standard__bert_base_cased__mnli__scratch" / "model"


# ── Shared setup ────────────────────────────────────────────────────────────

def load_mnli_dataset(tokenizer, max_length=128):
    from datasets import load_dataset

    dataset = load_dataset("glue", "mnli", split="train")

    def tokenize(examples):
        return tokenizer(
            examples["premise"], examples["hypothesis"],
            truncation=True, max_length=max_length,
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["premise", "hypothesis", "idx"])
    dataset.set_format("torch")
    return dataset


def create_t6_student(teacher_model, num_labels=3):
    """Create a 6-layer BERT student by copying layers [1,3,5,7,9,11] from teacher."""
    teacher_layers = [1, 3, 5, 7, 9, 11]

    student_config = deepcopy(teacher_model.config)
    student_config.num_hidden_layers = 6

    student = AutoModelForSequenceClassification.from_config(student_config)

    # Copy embeddings
    student.bert.embeddings.load_state_dict(teacher_model.bert.embeddings.state_dict())

    # Copy selected encoder layers
    for student_idx, teacher_idx in enumerate(teacher_layers):
        student.bert.encoder.layer[student_idx].load_state_dict(
            teacher_model.bert.encoder.layer[teacher_idx].state_dict()
        )

    # Copy pooler and classifier
    student.bert.pooler.load_state_dict(teacher_model.bert.pooler.state_dict())
    student.classifier.load_state_dict(teacher_model.classifier.state_dict())

    return student


def reset_gpu_stats(device):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)


def get_peak_memory_mb(device):
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1024 / 1024


# ── TextBrewer benchmark ───────────────────────────────────────────────────

def run_textbrewer(teacher, student, dataloader, num_steps, lr, temperature, warmup_ratio, device):
    from textbrewer import TrainingConfig, DistillationConfig, GeneralDistiller

    teacher = teacher.to(device).eval()
    student = student.to(device).train()

    def adaptor(batch, model_outputs):
        return {"logits": model_outputs.logits}

    train_config = TrainingConfig(
        device=device,
        log_dir=None,
        output_dir="/tmp/textbrewer_benchmark",
        ckpt_steps=num_steps + 1,  # no checkpointing during benchmark
    )
    distill_config = DistillationConfig(
        temperature=temperature,
        kd_loss_type="ce",
        kd_loss_weight=1.0,
        hard_label_weight=0.0,
        intermediate_matches=None,
    )

    distiller = GeneralDistiller(
        train_config=train_config,
        distill_config=distill_config,
        model_T=teacher,
        model_S=student,
        adaptor_T=adaptor,
        adaptor_S=adaptor,
    )

    optimizer = AdamW(student.parameters(), lr=lr)
    num_warmup = int(num_steps * warmup_ratio)

    reset_gpu_stats(device)

    start = time.perf_counter()
    distiller.train(
        optimizer=optimizer,
        dataloader=dataloader,
        num_steps=num_steps,
        scheduler_class=get_linear_schedule_with_warmup,
        scheduler_args={"num_warmup_steps": num_warmup, "num_training_steps": num_steps},
        max_grad_norm=1.0,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    peak_mem = get_peak_memory_mb(device)
    total_samples = num_steps * dataloader.batch_size

    return {
        "toolkit": "TextBrewer",
        "wall_clock_sec": round(elapsed, 2),
        "sec_per_step": round(elapsed / num_steps, 4),
        "samples_per_sec": round(total_samples / elapsed, 1),
        "peak_gpu_mb": round(peak_mem, 0),
    }


# ── silverspoon-kd benchmark ──────────────────────────────────────────────

def run_silverspoon(teacher, student, dataloader, num_steps, lr, temperature, warmup_ratio, device,
                     bf16=False, fused_optim=False, label=None):
    import sys
    sys.path.insert(0, str(PROJECT_DIR))
    from silverspoon_kd import ResponseBasedDistiller, TrainingArguments
    from silverspoon_kd.losses import kl_divergence_loss

    teacher = teacher.to(device).eval()
    student = student.to(device).train()

    output_dir = "/tmp/silverspoon_benchmark"
    os.makedirs(output_dir, exist_ok=True)

    num_warmup = int(num_steps * warmup_ratio)
    optim_name = "adamw_torch_fused" if fused_optim else "adamw_torch"

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        run_name="benchmark_silverspoon",
        max_steps=num_steps,
        per_device_train_batch_size=dataloader.batch_size,
        learning_rate=lr,
        lr_scheduler_type="linear",
        warmup_steps=num_warmup,
        max_grad_norm=1.0,
        bf16=bf16,
        fp16=False,
        report_to="none",
        logging_steps=9999,  # suppress logging
        save_strategy="no",
        eval_strategy="no",
        disable_tqdm=True,
        alpha=0.0,  # pure KD, no hard label
        auto_dtype_match=True,
        optim=optim_name,
    )

    def prepare_teacher_inputs(inputs):
        result = {"input_ids": inputs["input_ids"]}
        if "attention_mask" in inputs:
            result["attention_mask"] = inputs["attention_mask"]
        if "token_type_ids" in inputs:
            result["token_type_ids"] = inputs["token_type_ids"]
        return result

    collator = DataCollatorWithPadding(dataloader.dataset.tokenizer if hasattr(dataloader.dataset, "tokenizer") else None)

    trainer = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        train_dataset=dataloader.dataset,
        data_collator=dataloader.collate_fn,
        args=training_args,
        soft_loss_fn=kl_divergence_loss(temperature=temperature, chunk_size=0),
        prepare_teacher_inputs=prepare_teacher_inputs,
    )

    reset_gpu_stats(device)

    start = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    peak_mem = get_peak_memory_mb(device)
    total_samples = num_steps * dataloader.batch_size

    return {
        "toolkit": label or "silverspoon-kd",
        "wall_clock_sec": round(elapsed, 2),
        "sec_per_step": round(elapsed / num_steps, 4),
        "samples_per_sec": round(total_samples / elapsed, 1),
        "peak_gpu_mb": round(peak_mem, 0),
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TextBrewer vs silverspoon-kd benchmark")
    parser.add_argument("--num_steps", type=int, default=1000, help="Training steps per toolkit")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Steps: {args.num_steps}, Batch: {args.batch_size}, LR: {args.lr}, T: {args.temperature}")
    print()

    # Load teacher
    print("Loading teacher from", TEACHER_PATH)
    teacher = AutoModelForSequenceClassification.from_pretrained(str(TEACHER_PATH))
    tokenizer = AutoTokenizer.from_pretrained(str(TEACHER_PATH))

    # Load data
    print("Loading MNLI dataset...")
    dataset = load_mnli_dataset(tokenizer, max_length=args.max_length)
    _collator = DataCollatorWithPadding(tokenizer)
    # TextBrewer checks `type(batch) is dict` (not isinstance), so we must
    # return plain dicts — HF DataCollatorWithPadding returns BatchEncoding.
    def collator(features):
        return dict(_collator(features))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collator, num_workers=4, pin_memory=True)

    # ── Shared run kwargs ─────────────────────────────────────────────
    run_kwargs = dict(
        dataloader=dataloader, num_steps=args.num_steps,
        lr=args.lr, temperature=args.temperature,
        warmup_ratio=args.warmup_ratio, device=device,
    )

    def run_and_cleanup(run_fn, teacher_model, display_label, **extra):
        print(f"\n{'=' * 60}")
        print(f"  {display_label}")
        print(f"{'=' * 60}")
        student = create_t6_student(teacher_model, num_labels=3)
        result = run_fn(teacher=deepcopy(teacher_model), student=student, **run_kwargs, **extra)
        print(json.dumps(result, indent=2))
        del student
        gc.collect()
        torch.cuda.empty_cache()
        return result

    # ── 1. TextBrewer (fp32, vanilla AdamW) ────────────────────────────
    result_tb = run_and_cleanup(run_textbrewer, teacher, "TextBrewer (fp32)")

    # ── 2. silverspoon-kd (fp32, vanilla AdamW — apples-to-apples) ────
    result_sk_vanilla = run_and_cleanup(
        run_silverspoon, teacher, "silverspoon-kd (fp32, vanilla)",
        label="sk (fp32 vanilla)")

    # ── 3. silverspoon-kd (fp32, fused AdamW) ────────────────────────
    result_sk_fused = run_and_cleanup(
        run_silverspoon, teacher, "silverspoon-kd (fp32, fused optim)",
        fused_optim=True, label="sk (fp32 fused)")

    # ── 4. silverspoon-kd (bf16, fused AdamW) ────────────────────────
    result_sk_bf16 = run_and_cleanup(
        run_silverspoon, teacher, "silverspoon-kd (bf16, fused optim)",
        bf16=True, fused_optim=True, label="sk (bf16 fused)")

    # ── 5. TextBrewer HKD (fp32, hidden MSE + logit KL) ─────────────────
    print(f"\n{'=' * 60}")
    print("  TextBrewer HKD (fp32, hidden MSE + logit KL)")
    print(f"{'=' * 60}")
    student = create_t6_student(teacher, num_labels=3)
    try:
        from textbrewer import TrainingConfig, DistillationConfig, GeneralDistiller
        teacher_hkd = deepcopy(teacher).to(device).eval()
        student.to(device).train()

        # Layer mapping: teacher [1,3,5,7,9,11] → student [0,1,2,3,4,5]
        teacher_layers = [1, 3, 5, 7, 9, 11]
        intermediate_matches = [
            {"layer_T": t_idx, "layer_S": s_idx,
             "feature": "hidden", "loss": "hidden_mse", "weight": 1}
            for s_idx, t_idx in enumerate(teacher_layers)
        ]

        def adaptor_hkd(batch, model_outputs):
            return {
                "logits": model_outputs.logits,
                "hidden": model_outputs.hidden_states,
            }

        train_config = TrainingConfig(
            device=device, log_dir=None,
            output_dir="/tmp/textbrewer_hkd",
            ckpt_steps=args.num_steps + 1,
        )
        distill_config = DistillationConfig(
            temperature=args.temperature,
            kd_loss_type="ce",
            kd_loss_weight=1.0,
            hard_label_weight=0.0,
            intermediate_matches=intermediate_matches,
        )
        distiller_tb_hkd = GeneralDistiller(
            train_config=train_config,
            distill_config=distill_config,
            model_T=teacher_hkd,
            model_S=student,
            adaptor_T=adaptor_hkd,
            adaptor_S=adaptor_hkd,
        )
        optimizer = AdamW(student.parameters(), lr=args.lr)
        num_warmup = int(args.num_steps * args.warmup_ratio)

        # Enable output_hidden_states for both models
        teacher_hkd.config.output_hidden_states = True
        student.config.output_hidden_states = True

        reset_gpu_stats(device)
        start = time.perf_counter()
        distiller_tb_hkd.train(
            optimizer=optimizer, dataloader=dataloader, num_steps=args.num_steps,
            scheduler_class=get_linear_schedule_with_warmup,
            scheduler_args={"num_warmup_steps": num_warmup, "num_training_steps": args.num_steps},
            max_grad_norm=1.0,
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        peak_mem = get_peak_memory_mb(device)
        total_samples = args.num_steps * args.batch_size
        result_tb_hkd = {
            "toolkit": "TextBrewer HKD (hidden MSE + KL)",
            "wall_clock_sec": round(elapsed, 2),
            "sec_per_step": round(elapsed / args.num_steps, 4),
            "samples_per_sec": round(total_samples / elapsed, 1),
            "peak_gpu_mb": round(peak_mem, 0),
        }
        print(json.dumps(result_tb_hkd, indent=2))
        del teacher_hkd, student, distiller_tb_hkd
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  TextBrewer HKD failed: {e}")
        result_tb_hkd = {"toolkit": "TextBrewer HKD (hidden MSE + KL)", "error": str(e)}
    gc.collect()
    torch.cuda.empty_cache()

    # ── 6. silverspoon-kd HKD (fp32, hidden MSE + logit KL) ──────────
    print(f"\n{'=' * 60}")
    print("  silverspoon-kd HKD (fp32, hidden MSE + output KL)")
    print(f"{'=' * 60}")
    student = create_t6_student(teacher, num_labels=3)
    try:
        from silverspoon_kd import HolisticDistiller, TrainingArguments
        from silverspoon_kd.alignments.alignment import Alignment
        from silverspoon_kd.losses.kl import kl_divergence_loss

        teacher.to(device).eval()
        student.to(device).train()

        teacher_layers_idx = [1, 3, 5, 7, 9, 11]
        # Hidden-state MSE alignments (same as TextBrewer's intermediate_matches)
        alignments = [
            Alignment(
                teacher_block=teacher.bert.encoder.layer[t_idx],
                student_block=student.bert.encoder.layer[s_idx],
                teacher_model_name="bert-base",
                student_model_name="bert-t6",
                teacher_module_name=f"bert.encoder.layer.{t_idx}",
                student_module_name=f"bert.encoder.layer.{s_idx}",
                loss_function="mse",
                auto_projector=False,  # same hidden dim, no projection needed
                loss_weight=1.0,
                auto_device_match=True,
                auto_dtype_match=True,
            )
            for s_idx, t_idx in enumerate(teacher_layers_idx)
        ]
        # Output KL alignment on classifier head (matches TextBrewer's kd_loss)
        alignments.append(Alignment(
            teacher_block=teacher.classifier,
            student_block=student.classifier,
            teacher_model_name="bert-base",
            student_model_name="bert-t6",
            teacher_module_name="classifier",
            student_module_name="classifier",
            loss_function=kl_divergence_loss(temperature=args.temperature),
            auto_projector=False,
            loss_weight=1.0,
            auto_device_match=True,
            auto_dtype_match=True,
        ))

        output_dir = "/tmp/silverspoon_hkd_benchmark"
        os.makedirs(output_dir, exist_ok=True)
        num_warmup = int(args.num_steps * args.warmup_ratio)

        training_args = TrainingArguments(
            output_dir=output_dir, logging_dir=output_dir,
            run_name="benchmark_silverspoon_hkd",
            max_steps=args.num_steps,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            lr_scheduler_type="linear",
            warmup_steps=num_warmup,
            max_grad_norm=1.0,
            bf16=False, fp16=False,
            report_to="none", logging_steps=9999,
            save_strategy="no", eval_strategy="no",
            disable_tqdm=True,
            auto_dtype_match=True,
        )

        def prepare_teacher_inputs(inputs):
            result = {"input_ids": inputs["input_ids"]}
            if "attention_mask" in inputs:
                result["attention_mask"] = inputs["attention_mask"]
            if "token_type_ids" in inputs:
                result["token_type_ids"] = inputs["token_type_ids"]
            return result

        hkd_trainer = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=alignments,
            train_dataset=dataloader.dataset,
            data_collator=dataloader.collate_fn,
            args=training_args,
            prepare_teacher_inputs=prepare_teacher_inputs,
        )

        reset_gpu_stats(device)
        start = time.perf_counter()
        hkd_trainer.train()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        peak_mem = get_peak_memory_mb(device)
        total_samples = args.num_steps * args.batch_size
        result_sk_hkd = {
            "toolkit": "sk HKD (hidden MSE + KL)",
            "wall_clock_sec": round(elapsed, 2),
            "sec_per_step": round(elapsed / args.num_steps, 4),
            "samples_per_sec": round(total_samples / elapsed, 1),
            "peak_gpu_mb": round(peak_mem, 0),
        }
        print(json.dumps(result_sk_hkd, indent=2))
        del student, hkd_trainer
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  silverspoon-kd HKD failed: {e}")
        result_sk_hkd = {"toolkit": "sk HKD (hidden MSE + KL)", "error": str(e)}
    gc.collect()
    torch.cuda.empty_cache()

    # ── Summary ─────────────────────────────────────────────────────────
    all_results = [result_tb, result_sk_vanilla, result_sk_fused, result_sk_bf16,
                   result_tb_hkd, result_sk_hkd]

    print(f"\n{'=' * 80}")
    print(f"  COMPARISON SUMMARY ({args.num_steps} steps, batch={args.batch_size})")
    print(f"{'=' * 80}")
    print(f"{'Configuration':<35} {'Time (s)':>10} {'Samp/sec':>10} {'GPU (MB)':>10} {'vs TB':>8}")
    print("-" * 75)
    tb_time = result_tb["wall_clock_sec"]
    for r in all_results:
        speedup = tb_time / r["wall_clock_sec"]
        print(f"{r['toolkit']:<35} {r['wall_clock_sec']:>10.1f} {r['samples_per_sec']:>10.1f} {r['peak_gpu_mb']:>10.0f} {speedup:>7.2f}x")

    results = {
        "textbrewer_fp32": result_tb,
        "silverspoon_fp32_vanilla": result_sk_vanilla,
        "silverspoon_fp32_fused": result_sk_fused,
        "silverspoon_bf16_fused": result_sk_bf16,
        "textbrewer_hkd": result_tb_hkd,
        "silverspoon_hkd": result_sk_hkd,
        "config": vars(args),
    }

    out_path = args.output or str(PROJECT_DIR / "results" / "textbrewer_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
