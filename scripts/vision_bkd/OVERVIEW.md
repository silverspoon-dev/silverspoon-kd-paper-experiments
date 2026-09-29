# Exp 1: VGG Compression — Block-Wise Knowledge Distillation on Convolutional Networks

## Purpose

Demonstrates that silverspoon-kd supports **Block-Wise Knowledge Distillation (BKD)** on classical CNN architectures, as described by Wang et al. ("Progressive Blockwise Knowledge Distillation for Neural Network Acceleration", 2018) and Koratana et al. ("LIT: Block-wise Intermediate Representation Training for Model Compression", 2018).

## Setup

- **Teacher**: VGG16 (standard convolutions) on CIFAR-10
- **Student**: VGG16-DS (depthwise-separable convolutions) on CIFAR-10
- **Dataset**: CIFAR-10
- **Training**: 100 epochs (60 for the fine-tune stage), batch 128, LR 1e-3

## Experiments

| Script | Method | Description |
|--------|--------|-------------|
| `teacher.sh` | — | Fine-tune the VGG16 teacher on CIFAR-10 |
| `baseline.sh` | None | Student trained from scratch (no teacher) |
| `bkd.sh` | BKD | Block-wise distillation from VGG16 to VGG16-DS |
| `bkd_finetune.sh` | BKD → FT | Stage 2: supervised fine-tune from the BKD checkpoint |
| `bkd_hkd.sh` | BKD → HKD | Stage 2: holistic refinement from the BKD checkpoint |
| `bkd_reskd.sh` | BKD → ResKD | Stage 2: response-based KD from the BKD checkpoint |
| `reskd.sh` | ResKD | Response-based (logit) distillation |

`slurm/vgg16_cnn.sh` runs all of the above and harvests the best test accuracy of each run.

## Results (paper, table "VGG Compression on CIFAR-10")

BKD → ResKD reaches 90.42% test accuracy, 3.16pp above the VGG-16 teacher (87.26%); BKD → FT gives 88.92% and BKD → HKD 87.97%. Standalone ResKD reaches 84.35% and the no-teacher baseline 82.69%, while standalone BKD stops at 72.49%: blockwise alignment on its own is a strong initialisation but needs a second stage to become a good classifier.

## What This Shows

- silverspoon-kd can perform BKD on **non-transformer** architectures (CNNs), showing the toolkit is architecture-agnostic.
- The block-wise approach works with convolutional feature maps, not just hidden states from transformer layers.
- Automatic projectors handle the dimension mismatch between standard and depthwise-separable convolutions.
