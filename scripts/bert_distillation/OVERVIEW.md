# Exp 3: BERT Pre-Training — Masked Language Model Distillation

## Purpose

Demonstrates that silverspoon-kd works on **encoder-only transformer architectures** for Masked Language Model pre-training. This is an architecture-agnostic proof that the toolkit handles bidirectional encoders, complementing the decoder experiments (GPT-2, Qwen3) and CNN experiments (VGG16).

## Setup

- **Teacher**: BERT-base-uncased (110M parameters)
- **Students**: bert_T6 (6L/768H, mild compression), bert_T4_tiny (4L/312H, aggressive compression)
- **Dataset**: Dolma (pre-training corpus), max sequence length 512
- **Task**: Masked Language Modeling (pre-training)
- **Training**: 20,000 steps, float32

## Experiments

| Script | Method | Description |
|--------|--------|-------------|
| `baseline.sh` | None | Student pre-trained from scratch on Dolma |
| `bkd.sh` | BKD | Block-wise distillation from BERT-base |
| `hkd.sh` | HKD | Holistic (end-to-end) distillation |
| `reskd.sh` | Response-based KD | Response-based (logit) distillation |
| `eval_glue.sh` | Eval | Fine-tunes distilled checkpoints on GLUE (MNLI, SST-2) |
| `run_all.sh` | All | Runs all 2 students × 4 methods (8 experiments) |

## Results (paper, table "BERT distillation")

HKD wins downstream MNLI at both compression levels: T6 80.33% (+3.04pp over the 77.29% baseline) and T4-tiny 70.44% (+3.52pp over 66.92%). The spread between paradigms fans out from about 1.3pp at T6 to 5.2pp at T4-tiny, where standalone BKD falls below the baseline (66.10%). The BERT-base teacher reaches 83.59%.

## What This Shows

- silverspoon-kd supports **encoder architectures** (BERT) with bidirectional attention, not just causal decoders and CNNs.
- All three KD paradigms (BKD, HKD, Response-based KD) work for Masked LM pre-training.
- The same toolkit interface handles encoders and decoders without architecture-specific code.
- Two student sizes (mild + aggressive compression) demonstrate the approach scales across compression ratios.
