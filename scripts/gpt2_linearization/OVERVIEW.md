# Exp 7: GPT-2 Attention Linearization — Softmax → LoLCATs Hybrid Attention

## Purpose

Distills GPT-2 Small's softmax attention into a LoLCATs hybrid-attention student of the same model family (Zhang et al., "LoLCATs: On Low-Rank Linearizing of Large Language Models", 2024). Every attention layer becomes sliding-window softmax over the most recent 64 tokens plus low-rank linear attention over older tokens, while all other weights are copied from the teacher and frozen, so only the learned feature maps (about 1M parameters) train. The experiment exercises SilverSpoon-KD's sub-module targeting: blockwise KD aligns the attention modules alone (`transformer.h.<i>.attn`) rather than whole blocks.

## Setup

- **Teacher**: GPT-2 Small (124M, 12 layers, softmax attention)
- **Student**: GPT-2 Small with LoLCATs hybrid attention (window 64), teacher weights copied and frozen — `configs/student/gpt2_small_lolcats.yaml`
- **Architecture file**: `compiled/lolcats_gpt2.py` is the LoLCATs variant of the transformers GPT-2 implementation (generated against transformers 5.3). `models.py` builds the student from it, and `eval_all.sh` registers it with `AutoModelForCausalLM` so saved students reload with their feature maps. The `flash-linear-attention` kernels are used when installed and fall back to pure PyTorch otherwise.
- **Dataset**: Dolma
- **Training**: 76,000 steps. Standalone KD runs use LR 1e-2; stage-2 runs and the baseline use LR 1e-3. Batch 32 for BKD, the fine-tune stage and the baseline; batch 16 for HKD and ReSKD.
- **Evaluation**: WikiText perplexity, LAMBADA, HellaSwag (`eval_all.sh`)

## Experiments

| Script | Method | Description |
|--------|--------|-------------|
| `lolcats_baseline.sh` | None | Teacher-initialised LoLCATs student, CE fine-tune only (the no-KD floor) |
| `lolcats_bkd.sh` | BKD | Attention-level alignment of hybrid attention to softmax attention (stage 1) |
| `lolcats_hkd.sh` | HKD | Holistic distillation with attention-level alignment |
| `lolcats_reskd.sh` | ReSKD | Logit-level distillation |
| `lolcats_bkd_ft.sh` | BKD → FT | Stage 2: CE fine-tune from the BKD checkpoint |
| `lolcats_bkd_hkd.sh` | BKD → HKD | Stage 2: holistic refinement from the BKD checkpoint |
| `lolcats_bkd_reskd.sh` | BKD → ReSKD | Stage 2: logit distillation from the BKD checkpoint |
| `eval_all.sh` | Eval | Evaluate every LoLCATs run (WikiText, LAMBADA, HellaSwag) |

`slurm/linearized_attention.sh` runs the seven training scripts in dependency order and then the evaluation.

## Results (paper, table "Attention Linearization")

Only BKD → FT (WikiText PPL 52.01) beats the no-KD LoLCATs floor (53.49). The standalone KD runs push perplexity above 230, and the KD-only chains (BKD → HKD, BKD → ReSKD) stay in the 250–290 range. In this setting, where the student keeps the teacher's weights and only the attention mechanism changes, distillation needs a fine-tuning stage to pay off.
