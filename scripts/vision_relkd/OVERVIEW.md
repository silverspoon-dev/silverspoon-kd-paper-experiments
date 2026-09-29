# Exp 2: VGG Relational KD — Relational Knowledge Distillation

## Purpose

Replicates the experimental setup from **Park et al. ("Relational Knowledge Distillation", CVPR 2019), Table 4** to demonstrate that silverspoon-kd supports easy-to-configure Relational Knowledge Distillation (RKD). The original paper distills a ResNet50 teacher into a VGG11-BN student on CIFAR-100 using relational loss functions that transfer structural knowledge (inter-sample relationships) rather than individual activations.

## Setup

- **Teacher**: ResNet50 on CIFAR-100 (paper reports 77.76% accuracy)
- **Student**: VGG11-BN on CIFAR-100 (paper reports 71.26% baseline)
- **Dataset**: CIFAR-100
- **Training**: 200 epochs, batch 128, float32, SGD (momentum 0.9), LR 0.1 decayed ×0.2 at epochs 60/120/160, weight decay 5e-4. The joint recipes fold the CE and relational losses as Park et al.'s L_CE + 25·L_RKD (α = 0.5, loss weight 25) and use LR 0.2 and weight decay 2.5e-4 so that the effective values match. The two-stage alternative trains features only in stage 1 and then fine-tunes the classifier with CE for 100 epochs (LR 0.01, milestones 30/60/80).

## Experiments

| Script | Method | Description | Paper Reference |
|--------|--------|-------------|-----------------|
| `teacher.sh` | Standard | Train ResNet50 teacher | 77.76% reported |
| `baseline.sh` | None | VGG11-BN trained from scratch | 71.26% reported |
| `reskd.sh` | Response-based KD | Response-based distillation (Hinton KD, T=16) | 74.26% reported |
| `hkd_mse.sh` | HKD (MSE) | Feature-based with MSE loss (silverspoon-kd comparison) | N/A |
| `hkd_mse_ft.sh` | HKD (MSE) → FT | Stage 2 CE fine-tune of the MSE run | N/A |
| `hkd_relkd_distance_joint.sh` | HKD (relkd-D), joint | Relational KD with distance loss, single stage (L_CE + 25·L_RKD-D) | 72.27% reported |
| `hkd_relkd_angle_joint.sh` | HKD (relkd-A), joint | Relational KD with angle loss, single stage | -- |
| `hkd_relkd_da_joint.sh` | HKD (relkd-DA), joint | Combined distance + angle loss, single stage | 72.97% reported |
| `hkd_relkd_{distance,angle,da}.sh` | HKD (relkd-*), stage 1 | Two-stage alternative: relational loss on features only | -- |
| `hkd_relkd_{distance,angle,da}_ft.sh` | → FT | Two-stage alternative: stage 2 CE fine-tune | -- |

`slurm/vision_relkd.sh` runs the teacher, baseline, ResKD, HKD (MSE) and the two-stage relational recipes; `slurm/vision_relkd_joint.sh` runs the three joint relational recipes.

The paper's relation-based rows come from the joint recipes. The two-stage stage-1 recipes disable the head-level KL term (`response_loss_weight=0`) and add no cross-entropy, so their classifier stays untrained until the `_ft.sh` stage; the joint recipes train it through the CE term (α = 0.5) in the same 200-epoch run. HKD (MSE) keeps the default head-level KL alignment to the teacher's logits, which is what trains its classifier.

## Results (paper, table "VGG Relational KD on CIFAR-100")

| Method | Ours (%) | Park et al. (%) |
|--------|---------:|----------------:|
| Teacher (ResNet-50) | 77.81 | — |
| Baseline (VGG-11-BN, no teacher) | 65.16 | 71.26 |
| ResKD (T = 16) | **73.65** | 74.26 |
| HKD (MSE) | 72.78 | — |
| HKD (RelKD distance) | 70.40 | 72.27 |
| HKD (RelKD angle) | 71.01 | — |
| HKD (RelKD distance + angle) | 71.61 | 72.97 |

Every distillation variant improves substantially over the no-teacher baseline. ResKD lands within 0.61pp of Park et al.'s number despite a weaker baseline (65.16% vs 71.26%); the relation-based variants trail both ResKD and Park et al.'s reported values by about 1–2pp.

## What This Shows

- silverspoon-kd provides **built-in Relation-based KD loss functions** (distance, angle, combined DA) that can reproduce published results with minimal configuration.
- The toolkit supports **cross-architecture distillation** (ResNet50 to VGG11-BN) on vision tasks.
- Users can swap loss functions (MSE, relkd-D, relkd-A, relkd-DA) via a single config parameter, demonstrating the modularity of silverspoon-kd's loss system.
