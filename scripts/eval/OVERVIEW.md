# Evaluation Scripts

## Purpose

Provides a shared evaluation entrypoint using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to benchmark trained models on standard NLP tasks.

## Usage

```bash
DEVICES=0 ./run_eval.sh <run_name> [tasks] [extra_args]
```

Default tasks: `hellaswag,arc_easy,arc_challenge,winogrande,mmlu,truthfulqa_mc2,wikitext`

Used to evaluate all decoder-based experiments (GPT-2, Qwen3, linearized attention, quantization-aware) on a consistent set of benchmarks.
