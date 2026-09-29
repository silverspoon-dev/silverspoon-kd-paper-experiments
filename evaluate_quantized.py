"""Evaluate a model with real quantization applied to weights.

Loads a saved model, applies the same symmetric per-channel quantization
used during QAT (baking quantization error into the weights permanently),
then runs lm_eval benchmarks.

Usage:
    python evaluate_quantized.py --model runs/.../model --quant int8
    python evaluate_quantized.py --model Qwen/Qwen3-0.6B --quant int4  # PTQ baseline
"""

import argparse
import json
import os
import logging

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import lm_eval
from lm_eval.models.huggingface import HFLM
from quantization import make_quantize_fn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DEFAULT_TASKS = "hellaswag,arc_easy,arc_challenge,winogrande,mmlu,truthfulqa_mc2,wikitext"


def quantize_model_weights(model, spec):
    """Apply real quantization to all Linear layer weights (in-place)."""
    quantize_fn = make_quantize_fn(spec)
    count = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data = quantize_fn(module.weight.data)
                count += 1
    logger.info("Quantized %d Linear layers with %s", count, spec)
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate model with real quantization")
    parser.add_argument("--model", required=True, help="Model path or HuggingFace model name")
    parser.add_argument("--quant", required=True, help="Quantization spec (e.g. int4, int8)")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer path (defaults to --model, falls back to Qwen/Qwen3-0.6B)")
    parser.add_argument("--tasks", default=DEFAULT_TASKS, help="Comma-separated lm_eval tasks")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--output", default=None, help="Path to save results JSON")
    args = parser.parse_args()

    # Resolve tokenizer
    tokenizer_path = args.tokenizer or args.model
    is_local_tok = os.path.exists(tokenizer_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=is_local_tok
        )
    except Exception:
        logger.info("Tokenizer not found at %s, falling back to Qwen/Qwen3-0.6B", tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    # Load model
    is_local = os.path.exists(args.model)
    logger.info("Loading model from %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        local_files_only=is_local,
    )

    # Apply real quantization
    quantize_model_weights(model, args.quant)

    # Wrap for lm_eval
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    # Run evaluation
    tasks = [t.strip() for t in args.tasks.split(",")]
    logger.info("Evaluating on: %s", tasks)
    results = lm_eval.simple_evaluate(model=lm, tasks=tasks)

    # Print results table
    if "results" in results:
        print("\n" + lm_eval.utils.make_table(results))

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        # Filter to serializable data
        out = {task: metrics for task, metrics in results["results"].items()}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
