# Exp 4: BERT Downstream — Head-to-Head vs TextBrewer on MNLI

## Purpose

Compares SilverSpoon-KD directly against **TextBrewer** (Yang et al., ACL 2020) on MNLI, TextBrewer's primary benchmark, with the same teacher, students, data, and training budget. Holistic KD is configured to mirror TextBrewer's `GeneralDistiller` recipe so the two toolkits are compared like for like; blockwise KD is included because it is a paradigm TextBrewer does not offer.

## Setup

- **Teacher**: BERT-base-cased (12 layers, 768 hidden, 108M params), fine-tuned on MNLI by `finetune_teachers.sh`
- **Students**:
  - **T6**: 6 layers, 768 hidden, ~65M params (~59% of the teacher); teacher layers {1, 3, 5, 7, 9, 11}
  - **T4-tiny**: 4 layers, 312 hidden, ~14M params (~13%); teacher layers {2, 5, 8, 11} with hidden-dimension narrowing
- **Task**: MNLI (3-way natural language inference), matched and mismatched dev accuracy
- **Training** (`configs/training/downstream.yaml`): 30 epochs, batch 128, AdamW, LR 1e-4, linear schedule with 10% warmup, weight decay 0.01, max grad norm 1.0, float32. TextBrewer additionally uses early stopping with patience 5; the SilverSpoon-KD runs train for the full epoch budget.

## Methods

| Method | Framework | Description |
|--------|-----------|-------------|
| **Baseline** | — | Student fine-tuned on MNLI without a teacher |
| **TextBrewer** | TextBrewer | `GeneralDistiller` with `hidden_mse` intermediate matching and CE-soft KD on logits at T=8 |
| **HKD → FT** | SilverSpoon-KD | Single-stage holistic KD combining hidden-state alignment with a response-based loss at T=8 (response weight 1/T² so the effective KD weight matches TextBrewer's), followed by fine-tuning |
| **BKD → FT** | SilverSpoon-KD | Blockwise KD, a paradigm TextBrewer does not support, followed by fine-tuning |

## Experiments

1. **`finetune_teachers.sh`** — fine-tune BERT-base-cased on MNLI
2. **`run_downstream.sh`** — for each student (T6, T4-tiny): TextBrewer distillation via `run_textbrewer.py`, then the SilverSpoon-KD baseline, HKD, and BKD runs (`STUDENT=bert_T6` restricts it to one student)
3. **`evaluate_downstream.sh`** — evaluate every model and produce the comparison table

`slurm/bert_downstream.sh` runs the whole pipeline: the teacher once, then distillation and evaluation per seed.

## Results (paper, table "BERT Downstream MNLI")

| Student | Method | MNLI-m (%) | MNLI-mm (%) |
|---------|--------|-----------:|------------:|
| — | Teacher (BERT-base) | 83.59 | 83.76 |
| T6 | Baseline | 78.47 | 78.99 |
| T6 | TextBrewer | 80.09 | 80.48 |
| T6 | SilverSpoon-KD HKD → FT | **83.47** | **83.93** |
| T4-tiny | Baseline | 31.82 | 31.82 |
| T4-tiny | TextBrewer | **81.20** | **81.42** |
| T4-tiny | SilverSpoon-KD HKD → FT | 80.87 | — |
| T4-tiny | SilverSpoon-KD BKD → FT | 66.01 | 65.79 |

At T6, holistic KD matches the teacher within 0.12pp and beats TextBrewer by 3.38pp on TextBrewer's own benchmark. At T4-tiny, TextBrewer narrowly leads (81.20% vs 80.87%), and blockwise KD followed by fine-tuning reaches only 66.01%: blockwise alignment alone struggles when the structural gap to the teacher is this wide, consistent with the pre-training results of Exp 3.

## What This Shows

1. SilverSpoon-KD reaches TextBrewer-level quality on TextBrewer's benchmark when the same paradigm is used, and exceeds it at mild compression.
2. Blockwise KD is available for encoder tasks, which TextBrewer does not offer. It does not beat holistic KD here, so its value is in widening the design space rather than in headline accuracy.
3. Both toolkits see the same teacher checkpoint, students, data, and epoch budget, so the comparison isolates the toolkit rather than the training setup.
