#!/usr/bin/env python3
"""Evaluate MLM perplexity of a BERT model on WikiText-2.

Usage:
    python scripts/eval/eval_mlm_ppl.py --model google-bert/bert-base-uncased
    python scripts/eval/eval_mlm_ppl.py --model path/to/checkpoint --output results.json
"""

import argparse
import json
import math

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling


def eval_mlm_ppl(model_name_or_path, batch_size=32, max_samples=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Filter empty lines and tokenize
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

    def tokenize(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=512,
            padding="max_length", return_special_tokens_mask=True,
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    dataset.set_format("torch")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # Only count masked tokens (labels != -100)
            mask = batch["labels"] != -100
            n_tokens = mask.sum().item()
            if n_tokens > 0:
                total_loss += outputs.loss.item() * n_tokens
                total_tokens += n_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    return {"model": model_name_or_path, "mlm_loss": avg_loss, "mlm_ppl": ppl, "n_tokens": total_tokens}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = eval_mlm_ppl(args.model, args.batch_size, args.max_samples)
    print(f"MLM Loss: {result['mlm_loss']:.4f}")
    print(f"MLM PPL:  {result['mlm_ppl']:.2f}")
    print(f"Tokens:   {result['n_tokens']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
