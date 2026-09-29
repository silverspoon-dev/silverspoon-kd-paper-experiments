# SilverSpoon-KD: Paper Experiments

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Library](https://img.shields.io/badge/library-silverspoon--kd-3f51b5.svg)](https://github.com/silverspoon-dev/silverspoon-kd)
[![Docs](https://img.shields.io/badge/docs-kd.silverspoon.dev-3f51b5.svg)](https://kd.silverspoon.dev)

Experiment configs, training entrypoints, and SLURM job scripts behind every
table and figure in *SilverSpoon-KD: A General-Purpose Toolkit for Knowledge
Distillation* (Xaver R. Davey and David Broman, KTH Royal Institute of
Technology). The toolkit itself lives in
[silverspoon-dev/silverspoon-kd](https://github.com/silverspoon-dev/silverspoon-kd)
and is documented at [kd.silverspoon.dev](https://kd.silverspoon.dev).

The paper validates SilverSpoon-KD across vision, masked-language-model, and
causal-language-model workloads, covering homogeneous, heterogeneous, and
quantization-aware distillation with blockwise, holistic, and response-based
paradigms, and compares it against DistillKit, TextBrewer, torchdistill, and
torchtune. Each experiment group below is one appendix section of the paper.

## Repository layout

```
.
├── train.py                     # Hydra entrypoint for every training run
├── evaluate.py                  # Standalone evaluator (HF + lm-evaluation-harness)
├── models.py, trainers.py       # Model construction and trainer wiring shared by all groups
├── data.py, utils.py            # Dataset preparation and run naming
├── configs/                     # Hydra configs: data, distiller, loss, student, teacher, training
├── compiled/                    # Generated LoLCATs GPT-2 architecture used by Exp 7
├── scripts/
│   ├── vision_bkd/              # Exp 1 — VGG-16 → VGG-16-DS blockwise KD on CIFAR-10
│   ├── vision_relkd/            # Exp 2 — ResNet-50 → VGG-11-BN relational KD on CIFAR-100
│   ├── bert_distillation/       # Exp 3 — BERT MLM pre-training distillation
│   ├── bert_downstream/         # Exp 4 — BERT MNLI head-to-head vs TextBrewer
│   ├── gpt2_distillation/       # Exp 5 — GPT-2 multi-workflow KD (96M student)
│   ├── gpt2_cross_arch/         # Exp 6 — Cross-architecture KD (GPT-2 → DeepSeek-V3 layout)
│   ├── gpt2_linearization/      # Exp 7 — Softmax → LoLCATs hybrid attention
│   ├── qwen3_quantization/      # Exp 8 — Qwen3-1.7B quantization-aware training + KD
│   ├── toolkit_comparison/      # Framework comparison (throughput, memory, placement)
│   ├── gpt2_distillation_112M/  # Companion to Exp 5 with a 112M student (not in the paper)
│   ├── eval/                    # Shared evaluation and result summarisation
│   └── smoke_test.sh            # Two-step run of every pipeline for a quick local check
└── slurm/                       # SLURM drivers, one per experiment group
```

Each `scripts/<group>/OVERVIEW.md` describes that group's purpose, setup,
recipes, and the paper's results for it.

## Installation

```bash
pip install silverspoon-kd
pip install -r requirements.txt
```

The lower bounds in `requirements.txt` are the versions used to produce the
reported numbers (PyTorch 2.11, transformers 5.3, CUDA 13). Two optional
packages speed things up on CUDA GPUs and are not needed for correctness:
`liger-kernel` (fused losses, exercised in the framework comparison) and
`flash-linear-attention` (Triton kernels for the LoLCATs student in Exp 7).

## Running experiments locally

Every recipe under `scripts/<group>/` runs directly, without SLURM:

```bash
DEVICES=0 bash scripts/vision_bkd/bkd.sh
DEVICES=0 bash scripts/gpt2_distillation/bkd_hkd.sh
```

The shared driver `scripts/_common.sh` honours:

| Variable        | Default                            | Purpose                                          |
| --------------- | ---------------------------------- | ------------------------------------------------ |
| `DEVICES`       | `0`                                | `CUDA_VISIBLE_DEVICES` for the run               |
| `DRY_RUN`       | `false`                            | Print the commands instead of executing them     |
| `WANDB_PROJECT` | `silverspoon-kd-paper-experiments` | Weights & Biases project (`disabled` to skip)    |
| `RUNS_DIR`      | `./runs`                           | Where checkpoints, logs, and eval results go     |

A recipe is skipped when its output directory already exists, so re-running
a script after a crash continues where it left off. Multi-stage recipes
(`bkd_hkd.sh`, `*_ft.sh`, ...) locate their stage-1 checkpoint by run name,
so run the stages in the order the group's SLURM driver uses.

`bash scripts/smoke_test.sh` runs every pipeline for two steps at batch size 1
and is a quick way to confirm the environment before a cluster submission.

## Running on a SLURM cluster

The `slurm/` scripts are written for any SLURM cluster. Each header carries
two placeholders, which the `sbatch` command line can override:

```
#SBATCH --account=REPLACE_WITH_YOUR_ALLOCATION
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
```

```bash
export SLURM_CONDA_ENV=/path/to/conda/env   # holds the packages from requirements.txt
export STORAGE_DIR=/path/to/scratch         # optional; defaults to the repo root
sbatch --account=<allocation> --partition=<gpu-partition> slurm/bert_downstream.sh
```

`slurm/_common.sh` loads Lmod modules for Miniforge and CUDA; set
`MOD_MINIFORGE` and `MOD_CUDA` if your cluster names them differently. Set
`SEEDS=N` to repeat a driver over seeds 1..N (the default is a single run with
seed 42; the paper's variance figures use `SEEDS=3`).

## Reproducing the paper

| Paper section                              | Group                          | Driver                                                 |
| ------------------------------------------ | ------------------------------ | ------------------------------------------------------ |
| Exp 1 — VGG compression, CIFAR-10          | `scripts/vision_bkd/`          | `slurm/vgg16_cnn.sh`                                   |
| Exp 2 — VGG relational KD, CIFAR-100       | `scripts/vision_relkd/`        | `slurm/vision_relkd.sh`, `slurm/vision_relkd_joint.sh` |
| Exp 3 — BERT pre-training distillation     | `scripts/bert_distillation/`   | `slurm/bert_encoder.sh`                                |
| Exp 4 — BERT downstream vs TextBrewer      | `scripts/bert_downstream/`     | `slurm/bert_downstream.sh`                             |
| Exp 5 — GPT-2 compression                  | `scripts/gpt2_distillation/`   | `slurm/gpt2_decoder.sh`                                |
| Exp 6 — Cross-architecture distillation    | `scripts/gpt2_cross_arch/`     | `slurm/cross_architecture.sh`                          |
| Exp 7 — GPT-2 attention linearization      | `scripts/gpt2_linearization/`  | `slurm/linearized_attention.sh`                        |
| Exp 8 — Qwen3 quantization-aware training  | `scripts/qwen3_quantization/`  | `slurm/qat.sh`                                         |
| Framework comparison                       | `scripts/toolkit_comparison/`  | `slurm/compare_*.sh`                                   |

Language-model runs are evaluated with each group's `eval_all.sh`
(lm-evaluation-harness), classification runs with

```bash
python evaluate.py --model_path runs/<run_name>/student_model --task mnli
```

and the result tables are rebuilt from the contents of `runs/` with

```bash
python scripts/eval/summarize_results.py            # all groups
python scripts/eval/summarize_results.py --latex    # LaTeX output
python scripts/eval/summarize_results.py --group <group_name>
```

The peak-memory and throughput columns come from Weights & Biases
(`summarize_results.py --wandb`, `scripts/eval/backfill_gpu_peak.py`) and need
`WANDB_PROJECT_PATH=<entity>/<project>` pointing at your own logs.

## Data

Datasets download automatically through the `datasets` and `torchvision`
libraries on first use, into the Hugging Face cache (`~/.cache/huggingface` by
default). No proprietary or access-restricted data is used.

## Compute and checkpoints

- Trained checkpoints are not part of this repository (the full `runs/` tree
  is about 7 GB). Every recipe retrains from public models and data.
- Recipes default to seed 42; `SEEDS=N` repeats a driver over seeds 1..N.
- Rough single-GPU wall-clock: about 7 days on one A40 for the full BERT
  downstream group, 3 days for the GPT-2 groups, and a day for the vision
  groups. The Qwen3 QAT runs need an A100 (40 GB).

## License

Apache License 2.0, see [LICENSE](LICENSE). Third-party packages keep their
own licenses.

## Citation

```bibtex
@software{silverspoon_kd,
  author  = {Davey, Xaver R.},
  title   = {SilverSpoon-KD: A General-Purpose Toolkit for Knowledge Distillation},
  year    = {2026},
  url     = {https://github.com/silverspoon-dev/silverspoon-kd},
  version = {0.1.0},
  license = {Apache-2.0},
}
```
