# Exp 8: Qwen3 Quantization — Quantization-Aware Training with Knowledge Distillation

## Purpose

Demonstrates that silverspoon-kd integrates with **Quantization-Aware Training (QAT)**, combining simulated low-precision training with knowledge distillation to recover accuracy lost from quantization. This follows the approach of Polino et al. ("Model Compression via Distillation and Quantization", 2018), showing that distillation and quantization are complementary compression techniques.

## Setup

- **Teacher**: Qwen3-1.7B (bf16, full precision)
- **Student**: Qwen3-1.7B pretrained (same architecture and weights, simulated INT4/INT5/INT6 quantization-aware training; recipes also exist for INT7 and INT8)
- **Dataset**: Dolma
- **Training**: 20,000 steps, LR 2e-5, adamw_8bit; batch 4 for the baseline and BKD, batch 2 for HKD and ReSKD

INT4 is the focus because INT8 QAT is nearly lossless; INT4 is where distillation has the most impact.

## Experiments

One recipe per (bit width, method), named `int<bits>_<method>.sh`:

| Script | Method | Description |
|--------|--------|-------------|
| `int{4,5,6,7,8}_baseline.sh` | None | QAT without a teacher |
| `int{4,5,6,7,8}_bkd.sh` | BKD | QAT with block-wise distillation from the bf16 teacher |
| `int{4,5,6,7,8}_hkd.sh` | HKD | QAT with holistic distillation from the bf16 teacher |
| `int{4,5,6,7,8}_reskd.sh` | ReSKD | QAT with response-based distillation from the bf16 teacher |
| `bf16_pretrained_eval.sh` | Eval only | Reference ceiling: the pretrained bf16 model, no training |
| `eval_qat.sh [BITS...]` | Eval | Evaluates the PTQ floor and every QAT run for the given bit widths (default `4 6 8`); `METHODS` narrows the set |

`slurm/qat.sh` trains the four methods for every bit width in `BITS` (default `4 5 6`, the paper's sweep) and then runs both evaluation scripts.

## Results (paper, table "Qwen3-1.7B INT4/INT5/INT6 QAT with distillation")

HKD dominates at INT4: MMLU 48.26 against 41.50 for QAT alone (+6.76pp) and ARC-Easy +9.42pp, with WikiText perplexity 29.98 against 36.15. At INT5 and INT6 the paradigms converge, because post-training quantization alone already approaches the bf16 ceiling (MMLU 55.55).

## What This Shows

- silverspoon-kd is **compatible with quantization workflows**, a key model compression technique orthogonal to distillation.
- Distillation from the full-precision teacher recovers accuracy lost to aggressive quantization, with holistic KD the most effective paradigm at INT4.
- The experiment compares QAT alone vs. QAT + distillation vs. post-training quantization, showing the value of combining compression techniques.
- This demonstrates a practical deployment scenario: producing smaller, faster models through simultaneous distillation and quantization.
