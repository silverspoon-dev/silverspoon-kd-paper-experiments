# Exp 6: Cross-Architecture — MHA → MLA Distillation

## Purpose

Demonstrates silverspoon-kd's ability to distill knowledge **across fundamentally different model architectures**. The teacher (GPT-2 Small) uses standard multi-head attention with learned positional embeddings and GELU activations. The student uses DeepSeek-V3's architecture: Multi-head Latent Attention (MLA) with compressed KV projections, Rotary Position Embeddings (RoPE), and SiLU activations. This is the most architecturally diverse distillation in the paper — unlike the linearization experiments (same model family, different attention kernel), here the teacher and student share no architectural lineage.

## Setup

- **Teacher**: GPT-2 Small (124M, 12 layers, standard MHA, learned positional embeddings, GELU)
- **Student**: DeepSeek-V3-architecture 96M (6 layers, MLA with kv_lora_rank=96, RoPE, SiLU, dense/no MoE)
- **Dataset**: Dolma
- **Training**: LR 1e-3. HKD, ReSKD, and the baseline: batch 32 for 60,000 steps; BKD and BKD → HKD: batch 16 for 20,000 steps. The fine-tune stages (`hkd_ft.sh`, `reskd_ft.sh`): LR 5e-4, batch 32, 60,000 steps
- **Layer mapping**: 6 student layers aligned to every-other teacher layer via `teacher_layers: [0, 2, 4, 6, 8, 10]`

## Experiments

| Script | Method | Description |
|--------|--------|-------------|
| `baseline.sh` | None | DeepSeek-V3-arch 96M trained from scratch |
| `hkd.sh` | HKD | Holistic distillation (MSE) |
| `reskd.sh` | ReSKD | Output logit matching |
| `hkd_ft.sh` | HKD → FT | Stage 2: CE fine-tune from the HKD checkpoint |
| `reskd_ft.sh` | ReSKD → FT | Stage 2: CE fine-tune from the ReSKD checkpoint |
| `bkd.sh` | BKD | Block-wise distillation (MSE); exploratory, not in the paper table |
| `bkd_hkd.sh` | BKD → HKD | Two-stage pipeline; exploratory, not in the paper table |
| `eval_all.sh` | Eval | Evaluate all checkpoints (wikitext, LAMBADA, HellaSwag) |

## Results (paper, table "Cross-Architecture Distillation")

Standalone KD degrades LAMBADA well below the no-teacher baseline (HKD 0.87%, ReSKD 6.02%, baseline 16.34%). A fine-tuning stage brings both back to near-baseline quality: HKD → FT reaches PPL 118.49 / LAMBADA 15.06% and ReSKD → FT PPL 117.28 / 14.96%, against the baseline's PPL 117.78. The teacher sits at PPL 37.37.

## What This Shows

- silverspoon-kd can distill across **completely different model families**, not just within the same architecture with different sizes or attention kernels.
- The `teacher_layers` mechanism correctly handles **cross-architecture layer mapping** when teacher and student use different module naming conventions (`transformer.h.{i}` vs `model.layers.{i}`).
- Both hidden_size=768 are shared, so BKD aligns block outputs **without projectors** — isolating the architectural variable (MHA vs MLA, learned positions vs RoPE, GELU vs SiLU).
- Comparing results against the same-architecture GPT-2 → GPT-2 96M experiments directly quantifies the cost of cross-architecture transfer.
