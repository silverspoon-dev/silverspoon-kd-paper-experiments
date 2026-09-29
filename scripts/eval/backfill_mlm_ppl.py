#!/usr/bin/env python3
"""Backfill MLM perplexity for runs that didn't log e2e eval loss.

Loads saved student models and computes MLM loss on the textbook eval set,
then patches the trainer_state.json log_history with the missing metric.

Usage:
    python scripts/eval/backfill_mlm_ppl.py                 # dry-run
    python scripts/eval/backfill_mlm_ppl.py --write          # patch trainer_state
    python scripts/eval/backfill_mlm_ppl.py --pattern '*reskd*bert*'
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def compute_mlm_loss(model_dir, tokenizer_name="google-bert/bert-base-uncased",
                     max_samples=5000, batch_size=32, max_length=128):
    """Compute MLM eval loss for a saved model."""
    from datasets import load_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForMaskedLM.from_pretrained(model_dir).to(device)
    model.eval()

    # Use same eval set as training (textbook)
    ds = load_dataset("open-phi/textbooks", split="train[:5000]")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length,
                         padding="max_length", return_special_tokens_mask=True)

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    ds.set_format("torch")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collator)

    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="*reskd*bert*scratch",
                        help="Glob pattern for run dirs")
    parser.add_argument("--write", action="store_true",
                        help="Actually patch trainer_state.json")
    parser.add_argument("--runs-dir", default=None)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir) if args.runs_dir else project_root / "runs"

    for run_dir in sorted(runs_dir.glob(args.pattern)):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        model_dir = run_dir / "student_model"
        if not model_dir.exists():
            continue

        # Check if already has e2e loss
        ckpts = sorted(run_dir.glob("checkpoint-*/trainer_state.json"),
                       key=lambda p: int(p.parent.name.split("-")[1]))
        if not ckpts:
            continue
        ts = json.load(open(ckpts[-1]))
        has_e2e = any("eval_loss/e2e" in e for e in ts.get("log_history", []))
        if has_e2e:
            print(f"SKIP (already has e2e): {name}")
            continue

        print(f"Computing MLM loss: {name} ...")
        try:
            loss, ppl = compute_mlm_loss(str(model_dir))
            print(f"  MLM loss={loss:.4f}  PPL={ppl:.2f}")

            if args.write:
                # Append to last checkpoint's log_history
                ts["log_history"].append({
                    "eval_loss/e2e": loss,
                    "eval_loss/e2e_ppl": ppl,
                    "step": ts.get("global_step"),
                    "backfilled": True,
                })
                with open(ckpts[-1], "w") as f:
                    json.dump(ts, f, indent=2)
                print(f"  Patched {ckpts[-1]}")
            else:
                print("  (dry-run, use --write to patch)")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
