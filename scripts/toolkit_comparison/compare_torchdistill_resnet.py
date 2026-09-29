"""
Benchmark: silverspoon-kd vs torchdistill — ResNet-50 → ResNet-18 on CIFAR-100.

Runs both toolkits with identical:
  - Teacher: timm/resnet50.a1_in1k pretrained, adapted for CIFAR-100 (frozen)
  - Student: timm/resnet18.a1_in1k adapted for CIFAR-100 (randomly re-initialized each run)
  - Data: CIFAR-100 (32x32, 50K train, 10K test)
  - Hyperparameters: AdamW, lr=2e-4, warmup_ratio=0.1, max_grad_norm=1.0, batch_size=32
  - Duration: fixed number of training steps (default 5000)

Runs performed:
  1. torchdistill baseline (fp32)  — CrossEntropyLoss only (no teacher)
  2. torchdistill KD (fp32)        — KDLoss (temperature=2, alpha=0.5)
  3. torchdistill AT (fp32)        — ATLoss on layer1-4 + CE
  4. torchdistill baseline (bf16)  — Same as 1 with torch.autocast bf16
  5. torchdistill KD (bf16)        — Same as 2 with torch.autocast bf16
  6. torchdistill AT (bf16)        — Same as 3 with torch.autocast bf16
  7. silverspoon-kd baseline       — Standard training (no teacher)
  8. silverspoon-kd response-based KD           — ResponseBasedDistiller (T=2, α=0.5, fp32, vanilla)
  9. silverspoon-kd response-based KD fused     — Same + fused AdamW optimizer
 10. silverspoon-kd response-based KD bf16      — Same + bf16 + fused AdamW
 11. silverspoon-kd BKD            — BlockwiseDistiller (layer-wise, no torchdistill equiv)

Measures per run:
  - Test accuracy (%) on CIFAR-100 test set (10K samples)
  - Wall-clock time (seconds)
  - Throughput (samples/sec)
  - Peak GPU memory (MB)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/toolkit_comparison/compare_torchdistill_resnet.py [--num_steps 5000]
"""
import argparse
import gc
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForImageClassification,
    DefaultDataCollator,
    get_linear_schedule_with_warmup,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from data import get_train_eval_cifar100_datasets
from models import adapt_resnet_for_cifar


# ── Shared utilities ──────────────────────────────────────────────────────────

def reset_gpu_stats(device):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)


def get_peak_memory_mb(device):
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1024 / 1024


class MaterializedDataset(Dataset):
    """Materialize an iterable/map dataset into a list-backed map-style dataset."""
    def __init__(self, ds, max_samples=None):
        self.samples = []
        for i, sample in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_cifar100_data(batch_size):
    """Load CIFAR-100 train + test datasets, materialized for map-style access."""
    train_iter, test_iter, _ = get_train_eval_cifar100_datasets(
        image_processor=None, batch_size=batch_size, num_epochs=1, num_shards=4,
    )
    train_ds = MaterializedDataset(train_iter, max_samples=50000)
    test_ds = MaterializedDataset(test_iter, max_samples=10000)
    print(f"  Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")
    return train_ds, test_ds


def load_teacher(device):
    """Load frozen ResNet-50 teacher adapted for CIFAR-100."""
    model = AutoModelForImageClassification.from_pretrained("timm/resnet50.a1_in1k")
    model = adapt_resnet_for_cifar(model, num_classes=100)
    return model.to(device).eval()


def create_fresh_student(device):
    """Create a ResNet-18 student adapted for CIFAR-100 with fresh random weights."""
    model = AutoModelForImageClassification.from_pretrained("timm/resnet18.a1_in1k")
    model = adapt_resnet_for_cifar(model, num_classes=100)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    return model.to(device)


@torch.no_grad()
def evaluate_accuracy(model, dataloader, device):
    """Compute top-1 accuracy on evaluation data."""
    model.eval()
    correct = total = 0
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        logits = model(pixel_values=pixel_values).logits
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


# ── torchdistill runs ────────────────────────────────────────────────────────

def _train_loop_torchdistill(
    student, teacher, train_loader, num_steps, lr, warmup_ratio, device,
    loss_fn, needs_teacher=True, needs_targets=True, bf16=False,
):
    """Generic manual training loop using torchdistill loss functions."""
    student.train()
    optimizer = AdamW(student.parameters(), lr=lr)
    num_warmup = int(num_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, num_steps)
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()

    reset_gpu_stats(device)
    start = time.perf_counter()

    step = 0
    while step < num_steps:
        for batch in train_loader:
            if step >= num_steps:
                break
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with autocast_ctx:
                student_out = student(pixel_values=pixel_values)
                student_logits = student_out.logits

                if needs_teacher:
                    with torch.no_grad():
                        teacher_out = teacher(pixel_values=pixel_values)
                    teacher_logits = teacher_out.logits
                    student_io_dict = {".": {"output": student_logits}}
                    teacher_io_dict = {".": {"output": teacher_logits}}
                    loss_kwargs = {
                        "student_io_dict": student_io_dict,
                        "teacher_io_dict": teacher_io_dict,
                    }
                    if needs_targets:
                        loss_kwargs["targets"] = labels
                    loss = loss_fn(**loss_kwargs)
                else:
                    loss = loss_fn(student_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_mem = get_peak_memory_mb(device)
    return elapsed, peak_mem


def run_torchdistill_baseline(teacher, train_loader, num_steps, lr, warmup_ratio, device,
                              bf16=False):
    """torchdistill baseline: CE only, no teacher."""
    student = create_fresh_student(device)
    ce_loss = nn.CrossEntropyLoss()
    elapsed, peak_mem = _train_loop_torchdistill(
        student, teacher, train_loader, num_steps, lr, warmup_ratio, device,
        loss_fn=ce_loss, needs_teacher=False, bf16=bf16,
    )
    return student, elapsed, peak_mem


def run_torchdistill_kd(teacher, train_loader, num_steps, lr, warmup_ratio, device,
                        temperature=2.0, alpha=0.5, bf16=False):
    """torchdistill KD: KDLoss with temperature scaling."""
    from torchdistill.losses.mid_level import KDLoss

    student = create_fresh_student(device)
    kd_loss = KDLoss(
        student_module_path=".",
        student_module_io="output",
        teacher_module_path=".",
        teacher_module_io="output",
        temperature=temperature,
        alpha=alpha,
        beta=1.0 - alpha,
        reduction="batchmean",
    )

    elapsed, peak_mem = _train_loop_torchdistill(
        student, teacher, train_loader, num_steps, lr, warmup_ratio, device,
        loss_fn=kd_loss, needs_teacher=True, needs_targets=True, bf16=bf16,
    )
    return student, elapsed, peak_mem


def run_torchdistill_at(teacher, train_loader, num_steps, lr, warmup_ratio, device,
                        bf16=False):
    """torchdistill AT: Attention Transfer on ResNet layer1-4 + CE."""
    from torchdistill.losses.mid_level import ATLoss

    student = create_fresh_student(device)

    # Hook ResNet layer1-4 (residual block outputs)
    hook_layers = ["layer1", "layer2", "layer3", "layer4"]
    teacher_activations = {}
    student_activations = {}

    def make_hook(storage, name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            storage[name] = output
        return hook_fn

    teacher_hooks = []
    student_hooks = []
    for name in hook_layers:
        t_module = getattr(teacher.timm_model, name)
        s_module = getattr(student.timm_model, name)
        teacher_hooks.append(t_module.register_forward_hook(make_hook(teacher_activations, name)))
        student_hooks.append(s_module.register_forward_hook(make_hook(student_activations, name)))

    at_pairs = {}
    for i, name in enumerate(hook_layers):
        at_pairs[f"pair{i}"] = {
            "teacher": {"io": "output", "path": name},
            "student": {"io": "output", "path": name},
            "weight": 1.0,
        }
    at_loss_fn = ATLoss(at_pairs=at_pairs, mode="code")
    ce_loss_fn = nn.CrossEntropyLoss()

    student.train()
    optimizer = AdamW(student.parameters(), lr=lr)
    num_warmup = int(num_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, num_steps)
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()

    reset_gpu_stats(device)
    start = time.perf_counter()

    step = 0
    while step < num_steps:
        for batch in train_loader:
            if step >= num_steps:
                break
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with autocast_ctx:
                student_out = student(pixel_values=pixel_values)
                with torch.no_grad():
                    teacher_out = teacher(pixel_values=pixel_values)

                student_io_dict = {k: {"output": v} for k, v in student_activations.items()}
                teacher_io_dict = {k: {"output": v} for k, v in teacher_activations.items()}

                at_loss = at_loss_fn(student_io_dict, teacher_io_dict)
                ce_loss = ce_loss_fn(student_out.logits, labels)
                loss = ce_loss + 50.0 * at_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_mem = get_peak_memory_mb(device)

    for h in teacher_hooks + student_hooks:
        h.remove()

    return student, elapsed, peak_mem


# ── silverspoon-kd runs ──────────────────────────────────────────────────────

def _run_silverspoon_trainer(trainer, device):
    """Run a silverspoon-kd trainer and return timing/memory stats."""
    reset_gpu_stats(device)
    start = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_mem = get_peak_memory_mb(device)
    return elapsed, peak_mem


def run_silverspoon_baseline(teacher, train_ds, num_steps, lr, warmup_ratio, batch_size, device):
    """silverspoon-kd baseline: standard HF Trainer, CE only."""
    from transformers import Trainer, TrainingArguments

    student = create_fresh_student(device)
    output_dir = "/tmp/sk_resnet_baseline"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        run_name="sk_baseline",
        max_steps=num_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        lr_scheduler_type="linear",
        warmup_ratio=warmup_ratio,
        max_grad_norm=1.0,
        bf16=False,
        fp16=False,
        report_to="none",
        logging_steps=9999,
        save_strategy="no",
        eval_strategy="no",
        disable_tqdm=True,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=student,
        args=training_args,
        train_dataset=train_ds,
        data_collator=DefaultDataCollator(),
    )

    elapsed, peak_mem = _run_silverspoon_trainer(trainer, device)
    return student, elapsed, peak_mem


def run_silverspoon_reskd(teacher, train_ds, num_steps, lr, warmup_ratio, batch_size, device,
                         temperature=2.0, alpha=0.5, bf16=False, fused_optim=False):
    """silverspoon-kd response-based KD: ResponseBasedDistiller."""
    from silverspoon_kd import ResponseBasedDistiller, TrainingArguments
    from silverspoon_kd.losses import kl_divergence_loss

    student = create_fresh_student(device)
    tag = "sk_reskd"
    if bf16:
        tag += "_bf16"
    if fused_optim:
        tag += "_fused"
    output_dir = f"/tmp/sk_resnet_{tag}"
    os.makedirs(output_dir, exist_ok=True)

    optim_name = "adamw_torch_fused" if fused_optim else "adamw_torch"

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        run_name=tag,
        max_steps=num_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        lr_scheduler_type="linear",
        warmup_ratio=warmup_ratio,
        max_grad_norm=1.0,
        bf16=bf16,
        fp16=False,
        report_to="none",
        logging_steps=9999,
        save_strategy="no",
        eval_strategy="no",
        disable_tqdm=True,
        alpha=alpha,
        auto_dtype_match=True,
        optim=optim_name,
        dataloader_num_workers=4,
    )

    def prepare_teacher_inputs(inputs):
        return {"pixel_values": inputs["pixel_values"]}

    trainer = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        train_dataset=train_ds,
        data_collator=DefaultDataCollator(),
        args=training_args,
        soft_loss_fn=kl_divergence_loss(temperature=temperature),
        prepare_teacher_inputs=prepare_teacher_inputs,
    )

    elapsed, peak_mem = _run_silverspoon_trainer(trainer, device)
    return student, elapsed, peak_mem


def run_silverspoon_bkd(teacher, train_ds, num_steps, lr, warmup_ratio, batch_size, device):
    """silverspoon-kd BKD: BlockwiseDistiller with ResNet layer alignments."""
    from silverspoon_kd import BlockwiseDistiller, TrainingArguments
    from silverspoon_kd.alignments.utils import create_alignments

    student = create_fresh_student(device)
    output_dir = "/tmp/sk_resnet_bkd"
    os.makedirs(output_dir, exist_ok=True)

    layer_pattern = r"timm_model\.layer[1-4]$"
    alignments = create_alignments(
        teacher_model=teacher,
        student_model=student,
        modules=layer_pattern,
        loss_function="mse",
        auto_projector=True,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        run_name="sk_bkd",
        max_steps=num_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        lr_scheduler_type="linear",
        warmup_ratio=warmup_ratio,
        max_grad_norm=1.0,
        bf16=False,
        fp16=False,
        report_to="none",
        logging_steps=9999,
        save_strategy="no",
        eval_strategy="no",
        disable_tqdm=True,
        dataloader_num_workers=4,
    )

    def prepare_teacher_inputs(inputs):
        return {"pixel_values": inputs["pixel_values"]}

    trainer = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        train_dataset=train_ds,
        data_collator=DefaultDataCollator(),
        args=training_args,
        prepare_teacher_inputs=prepare_teacher_inputs,
        student_models={student.name_or_path: student},
    )

    elapsed, peak_mem = _run_silverspoon_trainer(trainer, device)
    return student, elapsed, peak_mem


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="torchdistill vs silverspoon-kd: ResNet-50→18 on CIFAR-100")
    parser.add_argument("--num_steps", type=int, default=5000, help="Training steps per run")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Steps: {args.num_steps}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"Temperature: {args.temperature}, Alpha: {args.alpha}")
    print()

    print("Loading teacher model (ResNet-50)...")
    teacher = load_teacher(device)
    for p in teacher.parameters():
        p.requires_grad = False

    print("Loading CIFAR-100 dataset...")
    train_ds, eval_ds = load_cifar100_data(args.batch_size)
    eval_loader = DataLoader(
        eval_ds, batch_size=64, shuffle=False,
        collate_fn=DefaultDataCollator(), num_workers=4, pin_memory=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=DefaultDataCollator(), num_workers=4, pin_memory=True,
    )

    all_results = {}

    def run_and_report(label, run_fn, **kwargs):
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")
        student, elapsed, peak_mem = run_fn(**kwargs)
        total_samples = args.num_steps * args.batch_size
        accuracy = evaluate_accuracy(student, eval_loader, device)
        result = {
            "toolkit": label,
            "test_accuracy_pct": round(accuracy, 2),
            "wall_clock_sec": round(elapsed, 2),
            "sec_per_step": round(elapsed / args.num_steps, 4),
            "samples_per_sec": round(total_samples / elapsed, 1),
            "peak_gpu_mb": round(peak_mem, 0),
        }
        print(json.dumps(result, indent=2))
        del student
        gc.collect()
        torch.cuda.empty_cache()
        return result

    # 1. torchdistill baseline
    all_results["torchdistill_baseline"] = run_and_report(
        "torchdistill baseline (CE only)",
        run_torchdistill_baseline,
        teacher=teacher, train_loader=train_loader,
        num_steps=args.num_steps, lr=args.lr,
        warmup_ratio=args.warmup_ratio, device=device,
    )

    # 2. torchdistill KD
    all_results["torchdistill_kd"] = run_and_report(
        "torchdistill KD (T=2, α=0.5)",
        run_torchdistill_kd,
        teacher=teacher, train_loader=train_loader,
        num_steps=args.num_steps, lr=args.lr,
        warmup_ratio=args.warmup_ratio, device=device,
        temperature=args.temperature, alpha=args.alpha,
    )

    # 3. torchdistill AT
    all_results["torchdistill_at"] = run_and_report(
        "torchdistill AT (attention transfer + CE)",
        run_torchdistill_at,
        teacher=teacher, train_loader=train_loader,
        num_steps=args.num_steps, lr=args.lr,
        warmup_ratio=args.warmup_ratio, device=device,
    )

    # 4. torchdistill baseline (bf16)
    all_results["torchdistill_baseline_bf16"] = run_and_report(
        "torchdistill baseline (CE, bf16)",
        run_torchdistill_baseline,
        teacher=teacher, train_loader=train_loader,
        num_steps=args.num_steps, lr=args.lr,
        warmup_ratio=args.warmup_ratio, device=device,
        bf16=True,
    )

    # 5. torchdistill KD (bf16)
    all_results["torchdistill_kd_bf16"] = run_and_report(
        "torchdistill KD (T=2, α=0.5, bf16)",
        run_torchdistill_kd,
        teacher=teacher, train_loader=train_loader,
        num_steps=args.num_steps, lr=args.lr,
        warmup_ratio=args.warmup_ratio, device=device,
        temperature=args.temperature, alpha=args.alpha,
        bf16=True,
    )

    # 6. torchdistill AT (bf16)
    all_results["torchdistill_at_bf16"] = run_and_report(
        "torchdistill AT (AT + CE, bf16)",
        run_torchdistill_at,
        teacher=teacher, train_loader=train_loader,
        num_steps=args.num_steps, lr=args.lr,
        warmup_ratio=args.warmup_ratio, device=device,
        bf16=True,
    )

    # 7. silverspoon-kd baseline (fp32)
    all_results["silverspoon_baseline"] = run_and_report(
        "silverspoon-kd baseline (CE only)",
        run_silverspoon_baseline,
        teacher=teacher, train_ds=train_ds,
        num_steps=args.num_steps, lr=args.lr,
        warmup_ratio=args.warmup_ratio, batch_size=args.batch_size, device=device,
    )

    # 8. silverspoon-kd response-based KD (fp32, vanilla — apples-to-apples with torchdistill)
    all_results["silverspoon_reskd"] = run_and_report(
        "sk response-based KD (fp32, vanilla)",
        run_silverspoon_reskd,
        teacher=teacher, train_ds=train_ds,
        num_steps=args.num_steps, lr=args.lr,
        warmup_ratio=args.warmup_ratio, batch_size=args.batch_size, device=device,
        temperature=args.temperature, alpha=args.alpha,
    )

    # 9. silverspoon-kd response-based KD (fp32, fused optimizer)
    all_results["silverspoon_reskd_fused"] = run_and_report(
        "sk response-based KD (fp32, fused optim)",
        run_silverspoon_reskd,
        teacher=teacher, train_ds=train_ds,
        num_steps=args.num_steps, lr=args.lr,
        warmup_ratio=args.warmup_ratio, batch_size=args.batch_size, device=device,
        temperature=args.temperature, alpha=args.alpha,
        fused_optim=True,
    )

    # 10. silverspoon-kd response-based KD (bf16, fused optimizer)
    all_results["silverspoon_reskd_bf16_fused"] = run_and_report(
        "sk response-based KD (bf16, fused optim)",
        run_silverspoon_reskd,
        teacher=teacher, train_ds=train_ds,
        num_steps=args.num_steps, lr=args.lr,
        warmup_ratio=args.warmup_ratio, batch_size=args.batch_size, device=device,
        temperature=args.temperature, alpha=args.alpha,
        bf16=True, fused_optim=True,
    )

    # 11. silverspoon-kd BKD
    all_results["silverspoon_bkd"] = run_and_report(
        "silverspoon-kd BKD (blockwise)",
        run_silverspoon_bkd,
        teacher=teacher, train_ds=train_ds,
        num_steps=args.num_steps, lr=args.lr,
        warmup_ratio=args.warmup_ratio, batch_size=args.batch_size, device=device,
    )

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print(f"  COMPARISON SUMMARY ({args.num_steps} steps, batch={args.batch_size})")
    print(f"{'=' * 90}")
    print(f"{'Configuration':<42} {'Acc (%)':>8} {'Time (s)':>10} {'Samp/sec':>10} {'GPU (MB)':>10}")
    print("-" * 82)
    for key, r in all_results.items():
        if key == "config":
            continue
        print(
            f"{r['toolkit']:<42} {r['test_accuracy_pct']:>8.2f} "
            f"{r['wall_clock_sec']:>10.1f} {r['samples_per_sec']:>10.1f} "
            f"{r['peak_gpu_mb']:>10.0f}"
        )

    all_results["config"] = vars(args)
    out_path = args.output or str(PROJECT_DIR / "results" / "torchdistill_resnet_cifar100_comparison.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
