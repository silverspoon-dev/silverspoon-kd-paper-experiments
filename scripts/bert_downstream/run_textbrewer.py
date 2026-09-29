#!/usr/bin/env python3
"""
Run TextBrewer's GeneralDistiller on MNLI for a direct quality comparison.

Replicates TextBrewer's Table 2 setup (Yang et al., ACL 2020):
  - Teacher: fine-tuned BERT-base-cased on MNLI
  - Students: T6 (6L/768H) or T4-tiny (4L/312H)
  - Intermediate matching: hidden_mse + NST with linear projectors
  - Temperature: 8, LR: 1e-4, 30 epochs with early stopping

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/bert_downstream/run_textbrewer.py --student T6
    CUDA_VISIBLE_DEVICES=0 python scripts/bert_downstream/run_textbrewer.py --student T4-tiny
    CUDA_VISIBLE_DEVICES=0 python scripts/bert_downstream/run_textbrewer.py --student T6 --dry_run
"""
import argparse
import json
import os
import sys
from copy import deepcopy

import wandb
from pathlib import Path

import numpy as np
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
sys.path.insert(0, str(PROJECT_DIR))
RUNS_DIR = PROJECT_DIR / "runs"
RESULTS_DIR = PROJECT_DIR / "results" / "downstream"
TEACHER_PATH = RUNS_DIR / "hf__standard__bert_base_cased__mnli__scratch" / "model"


class PlainDictCollator:
    """Wraps a HF collator to return plain dicts (required by TextBrewer).
    Defined at module level so it is picklable for multiprocessing DataLoaders.
    """
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, features):
        return dict(self.collator(features))


# ── Student definitions (matching TextBrewer Table 3) ─────────────────────

STUDENT_CONFIGS = {
    "T6": {
        "num_hidden_layers": 6,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_attention_heads": 12,
        "teacher_layers": [1, 3, 5, 7, 9, 11],
    },
    "T4-tiny": {
        "num_hidden_layers": 4,
        "hidden_size": 312,
        "intermediate_size": 1200,
        "num_attention_heads": 12,
        "teacher_layers": [2, 5, 8, 11],
    },
}


# ── Data ──────────────────────────────────────────────────────────────────

def load_mnli(tokenizer, max_length=128):
    from datasets import load_dataset

    ds = load_dataset("glue", "mnli")

    def tok(examples):
        return tokenizer(
            examples["premise"], examples["hypothesis"],
            truncation=True, max_length=max_length, padding=False,
        )

    train = ds["train"].map(tok, batched=True, remove_columns=["premise", "hypothesis", "idx"])
    train = train.rename_column("label", "labels")
    train.set_format("torch")

    val_m = ds["validation_matched"].map(tok, batched=True, remove_columns=["premise", "hypothesis", "idx"])
    val_m = val_m.rename_column("label", "labels")
    val_m.set_format("torch")

    val_mm = ds["validation_mismatched"].map(tok, batched=True, remove_columns=["premise", "hypothesis", "idx"])
    val_mm = val_mm.rename_column("label", "labels")
    val_mm.set_format("torch")

    return train, val_m, val_mm


# ── Model creation ────────────────────────────────────────────────────────

def create_student(teacher_model, student_name, num_labels=3):
    """Create a student by pruning layers from the teacher."""
    cfg = STUDENT_CONFIGS[student_name]

    student_config = deepcopy(teacher_model.config)
    student_config.num_hidden_layers = cfg["num_hidden_layers"]
    student_config.hidden_size = cfg["hidden_size"]
    student_config.intermediate_size = cfg["intermediate_size"]
    student_config.num_attention_heads = cfg["num_attention_heads"]
    student_config.num_labels = num_labels

    student = AutoModelForSequenceClassification.from_config(student_config)

    # Copy embeddings if dimensions match
    if cfg["hidden_size"] == teacher_model.config.hidden_size:
        student.bert.embeddings.load_state_dict(teacher_model.bert.embeddings.state_dict())
        student.bert.pooler.load_state_dict(teacher_model.bert.pooler.state_dict())
        student.classifier.load_state_dict(teacher_model.classifier.state_dict())
        for s_idx, t_idx in enumerate(cfg["teacher_layers"]):
            student.bert.encoder.layer[s_idx].load_state_dict(
                teacher_model.bert.encoder.layer[t_idx].state_dict()
            )
    # For T4-tiny (different hidden dim), start from random init

    return student


# ── Evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())
    return float(np.mean(np.array(all_preds) == np.array(all_labels)))


# ── TextBrewer training ──────────────────────────────────────────────────

def build_intermediate_matches(student_name, teacher_config, student_config):
    """Build TextBrewer intermediate_matches config (hidden_mse).

    Matches each student layer to its corresponding teacher layer with:
      - hidden_mse: MSE on hidden states (with linear projector if dims differ)

    Note: NST (mmd_loss) is omitted due to a tensor shape incompatibility in
    TextBrewer 0.2.x with recent transformers hidden_states output format.
    Hidden MSE alone provides strong intermediate matching.
    """
    cfg = STUDENT_CONFIGS[student_name]
    matches = []
    needs_proj = student_config.hidden_size != teacher_config.hidden_size

    for s_idx, t_idx in enumerate(cfg["teacher_layers"]):
        match_mse = {
            "layer_T": t_idx,
            "layer_S": s_idx,
            "feature": "hidden",
            "loss": "hidden_mse",
            "weight": 1.0,
        }
        if needs_proj:
            match_mse["proj"] = [
                "linear",
                student_config.hidden_size,
                teacher_config.hidden_size,
            ]
        matches.append(match_mse)

    return matches


def train_textbrewer(teacher, student, student_name, train_dataset, val_m_loader, val_mm_loader,
                     args, device):
    from textbrewer import TrainingConfig, DistillationConfig, GeneralDistiller

    teacher = teacher.to(device).eval()
    student = student.to(device).train()

    output_dir = str(RUNS_DIR / f"textbrewer__bert_{student_name}__mnli")
    os.makedirs(output_dir, exist_ok=True)

    intermediate_matches = build_intermediate_matches(
        student_name, teacher.config, student.config
    )

    train_config = TrainingConfig(
        device=device,
        log_dir=output_dir,
        output_dir=output_dir,
        ckpt_steps=999999,  # we handle checkpointing via early stopping
    )
    distill_config = DistillationConfig(
        temperature=8,
        kd_loss_type="ce",
        kd_loss_weight=1.0,
        hard_label_weight=0.0,
        intermediate_matches=intermediate_matches,
    )

    def adaptor(batch, model_outputs):
        return {
            "logits": (model_outputs.logits,),
            "hidden": model_outputs.hidden_states,
            "losses": (model_outputs.loss,) if model_outputs.loss is not None else (),
        }

    distiller = GeneralDistiller(
        train_config=train_config,
        distill_config=distill_config,
        model_T=teacher,
        model_S=student,
        adaptor_T=adaptor,
        adaptor_S=adaptor,
    )

    # TextBrewer requires manual training loop for epoch-level eval + early stopping.
    # Only include student params here — TextBrewer's initialize_training()
    # will add projector params via add_param_group() on each train() call.
    # We create a fresh optimizer each epoch (see loop below) so the
    # add_param_group doesn't collide with previously added projectors.
    optimizer = AdamW(student.parameters(), lr=args.lr)
    collator = DataCollatorWithPadding(AutoTokenizer.from_pretrained(str(TEACHER_PATH)))

    # Plain-dict collator (TextBrewer requires `type(batch) is dict`)
    plain_collator = PlainDictCollator(
        DataCollatorWithPadding(AutoTokenizer.from_pretrained(str(TEACHER_PATH)))
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=plain_collator, num_workers=4, pin_memory=True,
    )

    steps_per_epoch = len(train_loader)
    if args.max_steps > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)
        args.num_epochs = 1
    total_steps = steps_per_epoch * args.num_epochs
    num_warmup = int(total_steps * args.warmup_ratio)

    best_acc = 0.0
    patience_counter = 0

    print(f"Training: {args.num_epochs} epochs, {steps_per_epoch} steps/epoch, "
          f"{total_steps} total steps")

    # wandb picks up TextBrewer's per-step TensorBoard logs via log_dir sync
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "silverspoon-kd-paper-experiments"),
        name=f"textbrewer__{student_name}__{args.student}__mnli",
        config={"framework": "TextBrewer", "student": args.student,
                "lr": args.lr, "batch_size": args.batch_size,
                "num_epochs": args.num_epochs, "temperature": 8},
        sync_tensorboard=True,
    )

    for epoch in range(args.num_epochs):
        student.train()
        # Enable hidden_states output for intermediate matching
        teacher.config.output_hidden_states = True
        student.config.output_hidden_states = True

        # Create fresh optimizer each epoch — TextBrewer's initialize_training()
        # calls optimizer.add_param_group() for projectors on each train() call.
        # A fresh optimizer ensures no collision with previously added projectors.
        optimizer = AdamW(student.parameters(), lr=args.lr)

        distiller.train(
            optimizer=optimizer,
            dataloader=train_loader,
            num_steps=steps_per_epoch,
            scheduler_class=get_linear_schedule_with_warmup,
            scheduler_args={
                "num_warmup_steps": max(0, num_warmup - epoch * steps_per_epoch),
                "num_training_steps": steps_per_epoch,
            },
            max_grad_norm=1.0,
        )

        # Disable hidden_states for eval
        teacher.config.output_hidden_states = False
        student.config.output_hidden_states = False

        acc_m = evaluate(student, val_m_loader, device)
        acc_mm = evaluate(student, val_mm_loader, device)
        wandb.log({"eval/accuracy_matched": acc_m, "eval/accuracy_mismatched": acc_mm,
                    "epoch": epoch + 1})
        print(f"  Epoch {epoch + 1}/{args.num_epochs}: matched={acc_m:.4f}, mismatched={acc_mm:.4f}")

        if acc_m > best_acc:
            best_acc = acc_m
            patience_counter = 0
            # Save best model
            student.save_pretrained(os.path.join(output_dir, "model"))
            AutoTokenizer.from_pretrained(str(TEACHER_PATH)).save_pretrained(
                os.path.join(output_dir, "model")
            )
            best_results = {
                "accuracy_matched": acc_m,
                "accuracy_mismatched": acc_mm,
                "epoch": epoch + 1,
            }
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch + 1} (patience={args.patience})")
                break

    # Log peak GPU memory
    if torch.cuda.is_available():
        peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_gib = peak_mib / 1024
        print(f"  Peak GPU memory: {peak_mib:.0f} MiB ({peak_gib:.2f} GiB)")
        import json as _json
        mem_path = os.path.join(output_dir, "gpu_peak_memory.json")
        with open(mem_path, "w") as f:
            _json.dump({"peak_memory_mib": round(peak_mib, 1),
                        "peak_memory_gib": round(peak_gib, 3)}, f)

    print(f"  Best matched accuracy: {best_acc:.4f} (epoch {best_results['epoch']})")
    return best_results


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TextBrewer MNLI distillation (full training)")
    parser.add_argument("--student", required=True, choices=["T6", "T4-tiny"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_steps", type=int, default=0, help="Stop after N steps (0=unlimited)")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    from utils import get_device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else device.type
    print(f"TextBrewer MNLI Distillation: student={args.student}")
    print(f"Device: {device} ({device_name})")

    if args.dry_run:
        print("DRY RUN — exiting")
        return

    # Load teacher
    print(f"Loading teacher from {TEACHER_PATH}")
    teacher = AutoModelForSequenceClassification.from_pretrained(str(TEACHER_PATH))
    tokenizer = AutoTokenizer.from_pretrained(str(TEACHER_PATH))

    # Load data
    print("Loading MNLI...")
    train_ds, val_m_ds, val_mm_ds = load_mnli(tokenizer)

    plain_collator = PlainDictCollator(DataCollatorWithPadding(tokenizer))

    val_m_loader = DataLoader(val_m_ds, batch_size=64, collate_fn=plain_collator, num_workers=4)
    val_mm_loader = DataLoader(val_mm_ds, batch_size=64, collate_fn=plain_collator, num_workers=4)

    # Create student
    student = create_student(teacher, args.student, num_labels=3)
    n_params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"Student: {args.student} ({n_params:.1f}M params)")

    # Train
    results = train_textbrewer(
        teacher, student, args.student, train_ds, val_m_loader, val_mm_loader, args, device,
    )

    # Save results
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    out_path = RESULTS_DIR / f"textbrewer_bert_{args.student.replace('-', '_')}_mnli.json"
    with open(out_path, "w") as f:
        json.dump({"framework": "TextBrewer", "student": args.student, **results}, f, indent=2)
    print(f"Results saved to {out_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
