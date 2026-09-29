#!/usr/bin/env python3
"""Task-specific evaluation for downstream fine-tuned models.

Usage:
    python evaluate.py --model_path runs/.../model --task mnli
    python evaluate.py --model_path runs/.../model --task mnli --split validation_mismatched
    python evaluate.py --model_path runs/.../model --task squad
    python evaluate.py --model_path runs/.../model --task conll2003
    python evaluate.py --model_path runs/.../student_model --task mnli
"""

import argparse
import json
import logging
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForQuestionAnswering,
                          AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer,
                          DefaultDataCollator, DataCollatorWithPadding)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONLL_LABEL_LIST = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def evaluate_mnli(model_path, split="validation_matched", batch_size=64, device=None):
    """Evaluate a sequence classification model on MNLI."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # num_labels=3 for MNLI so transformers builds a matching head when
    # loading from a checkpoint that might have a different num_labels
    # (e.g. the default BERT-base has num_labels=2).  ignore_mismatched_sizes
    # allows transformers to discard a saved classifier head whose shape
    # doesn't fit; ours is trained during distillation so the shape is
    # always 3 here and nothing actually gets discarded.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=3, ignore_mismatched_sizes=True,
    ).to(device)
    model.eval()

    dataset = load_dataset("glue", "mnli", split=split)

    def tokenize_fn(examples):
        return tokenizer(
            examples["premise"], examples["hypothesis"],
            truncation=True, max_length=128, padding=False,
        )

    cols = [c for c in dataset.column_names if c not in ("label",)]
    tok_ds = dataset.map(tokenize_fn, batched=True, remove_columns=cols)
    tok_ds = tok_ds.rename_column("label", "labels")
    tok_ds.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(tok_ds, batch_size=batch_size, collate_fn=collator)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"MNLI ({split})"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return {"accuracy": float(accuracy), "split": split, "n_samples": len(all_labels)}


def evaluate_squad(model_path, split="validation", batch_size=32, device=None):
    """Evaluate a QA model on SQuAD v1.1."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(device)
    model.eval()

    dataset = load_dataset("squad", split=split)
    max_length = 384
    doc_stride = 128

    def prepare_features(examples):
        tokenized = tokenizer(
            examples["question"], examples["context"],
            truncation="only_second", max_length=max_length,
            stride=doc_stride, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding="max_length",
        )
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        tokenized["example_id"] = [examples["id"][s] for s in sample_mapping]
        return tokenized

    cols = dataset.column_names
    features = dataset.map(prepare_features, batched=True, remove_columns=cols)

    # Keep offset_mapping and example_id for post-processing
    offset_mapping_list = features["offset_mapping"]
    example_ids = features["example_id"]
    eval_features = features.remove_columns(["offset_mapping", "example_id"])
    eval_features.set_format("torch")

    from transformers import DataCollatorWithPadding
    collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(eval_features, batch_size=batch_size, collate_fn=collator)

    all_start_logits, all_end_logits = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="SQuAD"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())

    all_start_logits = np.concatenate(all_start_logits)
    all_end_logits = np.concatenate(all_end_logits)

    # Post-process predictions
    example_to_features = defaultdict(list)
    for feat_idx, eid in enumerate(example_ids):
        example_to_features[eid].append(feat_idx)

    predictions = {}
    for example in dataset:
        eid = example["id"]
        context = example["context"]
        feat_indices = example_to_features[eid]

        best_score = -float("inf")
        best_answer = ""

        for feat_idx in feat_indices:
            start_logits = all_start_logits[feat_idx]
            end_logits = all_end_logits[feat_idx]
            offsets = offset_mapping_list[feat_idx]

            # Get top-k start and end indices
            n_best = 20
            start_indices = np.argsort(start_logits)[-n_best:][::-1]
            end_indices = np.argsort(end_logits)[-n_best:][::-1]

            for start_idx in start_indices:
                for end_idx in end_indices:
                    if start_idx >= len(offsets) or end_idx >= len(offsets):
                        continue
                    if offsets[start_idx] is None or offsets[end_idx] is None:
                        continue
                    if offsets[start_idx][0] == 0 and offsets[start_idx][1] == 0:
                        continue
                    if end_idx < start_idx:
                        continue
                    if end_idx - start_idx + 1 > 30:
                        continue

                    score = start_logits[start_idx] + end_logits[end_idx]
                    if score > best_score:
                        best_score = score
                        start_char = offsets[start_idx][0]
                        end_char = offsets[end_idx][1]
                        best_answer = context[start_char:end_char]

        predictions[eid] = best_answer

    # Compute metrics
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in dataset]
    exact_match, f1_total = 0, 0.0
    for ref in references:
        pred = predictions.get(ref["id"], "")
        gold_answers = ref["answers"]["text"]
        em = max(_exact_match(pred, g) for g in gold_answers) if gold_answers else 0
        f1 = max(_f1_score(pred, g) for g in gold_answers) if gold_answers else 0.0
        exact_match += em
        f1_total += f1

    n = len(references)
    return {"exact_match": exact_match / n * 100, "f1": f1_total / n * 100, "n_samples": n}


def _normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    import re
    import string
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    s = ' '.join(s.split())
    return s


def _exact_match(pred, gold):
    return int(_normalize_answer(pred) == _normalize_answer(gold))


def _f1_score(pred, gold):
    pred_tokens = _normalize_answer(pred).split()
    gold_tokens = _normalize_answer(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    num_same = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
    precision = num_same / len(pred_tokens) if pred_tokens else 0
    recall = num_same / len(gold_tokens) if gold_tokens else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_conll(model_path, split="validation", batch_size=64, device=None):
    """Evaluate a token classification model on CoNLL-2003."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
    model.eval()

    dataset = load_dataset("BramVanroy/conll2003", split=split)

    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"], is_split_into_words=True,
            truncation=True, max_length=128, padding="max_length",
        )
        all_labels = []
        for i, ner_tags in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            labels = []
            prev = None
            for wid in word_ids:
                if wid is None:
                    labels.append(-100)
                elif wid != prev:
                    labels.append(ner_tags[wid])
                else:
                    labels.append(-100)
                prev = wid
            all_labels.append(labels)
        tokenized["labels"] = all_labels
        return tokenized

    cols = dataset.column_names
    tok_ds = dataset.map(tokenize_and_align, batched=True, remove_columns=cols)
    tok_ds.set_format("torch")

    collator = DefaultDataCollator()
    loader = DataLoader(tok_ds, batch_size=batch_size, collate_fn=collator)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="CoNLL-2003"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            for pred_row, label_row in zip(preds, labels):
                row_preds, row_labels = [], []
                for p, l in zip(pred_row, label_row):
                    if l == -100:
                        continue
                    row_labels.append(CONLL_LABEL_LIST[l])
                    row_preds.append(CONLL_LABEL_LIST[p] if p < len(CONLL_LABEL_LIST) else "O")
                all_preds.append(row_preds)
                all_labels.append(row_labels)

    try:
        from seqeval.metrics import classification_report, f1_score
        f1 = f1_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds)
        logger.info("\n%s", report)
    except ImportError:
        logger.warning("seqeval not installed, computing token-level accuracy instead")
        correct = sum(p == l for ps, ls in zip(all_preds, all_labels) for p, l in zip(ps, ls))
        total = sum(len(ls) for ls in all_labels)
        f1 = correct / total if total > 0 else 0.0

    return {"f1": float(f1) * 100, "n_samples": len(all_labels)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate downstream task models")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--task", required=True, choices=["mnli", "squad", "conll2003"],
                        help="Task to evaluate on")
    parser.add_argument("--split", default=None, help="Dataset split (default: task-specific)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default=None, help="Device (e.g., cuda:0)")
    parser.add_argument("--output_json", default=None, help="Path to save results as JSON")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None

    if args.task == "mnli":
        split = args.split or "validation_matched"
        results = evaluate_mnli(args.model_path, split=split, batch_size=args.batch_size, device=device)
        # Also evaluate mismatched if on matched
        if split == "validation_matched":
            mm_results = evaluate_mnli(args.model_path, split="validation_mismatched",
                                        batch_size=args.batch_size, device=device)
            results["accuracy_matched"] = results["accuracy"]
            results["accuracy_mismatched"] = mm_results["accuracy"]

    elif args.task == "squad":
        split = args.split or "validation"
        results = evaluate_squad(args.model_path, split=split, batch_size=args.batch_size, device=device)

    elif args.task == "conll2003":
        split = args.split or "validation"
        results = evaluate_conll(args.model_path, split=split, batch_size=args.batch_size, device=device)

    # Print results
    print(f"\n{'='*60}")
    print(f"Task: {args.task} | Model: {args.model_path}")
    print(f"{'='*60}")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    if args.output_json:
        results["model_path"] = args.model_path
        results["task"] = args.task
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", args.output_json)

    return results


if __name__ == "__main__":
    main()
