# Exp 5: GPT-2 Compression — Multi-Workflow Knowledge Distillation

## Purpose

Demonstrates that silverspoon-kd supports **multiple KD workflows on autoregressive decoder transformers**, showing that the full range of distillation paradigms -- BKD, HKD, Response-based KD, and multi-stage pipelines including LoRA fine-tuning -- apply cleanly to causal language models.

## Setup

- **Teacher**: GPT-2 Small (124M parameters)
- **Student**: GPT-2 96M (custom smaller configuration)
- **Dataset**: Dolma
- **Training**: LR 1e-3 at a Chinchilla-scale token budget: batch 32 for 60,000 steps (baseline, BKD, and the LoRA stage) or batch 16 for 117,000 steps (HKD, ReSKD, BKD → HKD)

## Experiments

| Script | Method | Description |
|--------|--------|-------------|
| `baseline.sh` | None | GPT-2 96M trained from scratch |
| `bkd.sh` | BKD | Block-wise distillation with MSE loss |
| `hkd.sh` | HKD | Holistic distillation (MSE on hidden states) |
| `reskd.sh` | Response-based KD | Response-based (logit) distillation |
| `bkd_hkd.sh` | BKD → HKD | Two-stage pipeline: BKD then holistic refinement |
| `bkd_hkd_lora.sh` | BKD → HKD → LoRA | Three-stage pipeline |
| `eval_all.sh` | Eval | Evaluate all checkpoints (wikitext, LAMBADA, HellaSwag) |

## Results (paper, table "GPT-2 Compression")

Standalone BKD diverges (WikiText PPL 245k), while the three-stage BKD → HKD → LoRA pipeline rescues it (PPL 93.39) and gives the best LAMBADA accuracy (19.56%, vs 15.25% for the no-teacher baseline at PPL 107.15). Standalone HKD reaches PPL 97.69 and ReSKD 133.99; the teacher sits at PPL 37.37.

## What This Shows

- silverspoon-kd applies to **decoder-only transformers** for causal language modeling.
- The toolkit supports **multi-stage distillation pipelines** where the output of one stage initializes the next (BKD → HKD → LoRA), managed via the `from_run` mechanism.
