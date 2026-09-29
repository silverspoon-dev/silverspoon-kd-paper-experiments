#!/usr/bin/env python3
"""Unified training entry point with Hydra configuration.

Usage:
    # Distillation
    python train.py teacher=qwen3_0.6B student=qwen3_420M distiller=bkd data=dolma
    python train.py teacher=qwen3_0.6B student=qwen3_420M distiller=hkd \
        init.method=from_experiment init.from_experiment=silverspoon__bkd__qwen3_0.6B__qwen3_420M__scratch \
        init.checkpoint_step=latest

    # LoRA (no teacher)
    python train.py student=qwen3_420M distiller=lora data=dolma

    # Standard training (no teacher)
    python train.py student=qwen3_420M distiller=standard data=dolma

    # Print experiment name only
    python train.py student=qwen3_420M distiller=lora run.name_only=true
"""

import logging
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer

from data import load_datasets
from models import (create_student_model, get_auto_model_class, get_model_type,
                    load_teacher_model, resolve_checkpoint_path, _load_from_checkpoint,
                    adapt_vgg16_for_cifar, apply_depthwise_separable_replacement,
                    _create_linearized_student, _make_pruning_example_inputs)
from quantization import apply_quantization_simulation
from trainers import (create_alignments, get_training_args,
                      train_distillation, train_lora, train_standard)
from utils import (get_experiment_name, get_model_precision, setup_device)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _build_compute_metrics(cfg):
    """Build a compute_metrics callback for downstream tasks."""
    model_type = get_model_type(cfg)
    data_type = cfg.data.type

    if model_type == "sequence_classification" or data_type == "glue_mnli":
        import numpy as np
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {"accuracy": (preds == labels).mean().item()}
        return compute_metrics

    if model_type == "image_classification":
        import numpy as np
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {"accuracy": (preds == labels).mean().item()}
        return compute_metrics

    if model_type == "question_answering" or data_type == "squad_v1":
        return None  # QA eval is done in evaluate.py (requires post-processing)

    if model_type == "token_classification" or data_type == "conll2003":
        import numpy as np
        from data import CONLL_LABEL_LIST
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            true_labels = []
            true_preds = []
            for pred_row, label_row in zip(preds, labels):
                row_labels = []
                row_preds = []
                for p, l in zip(pred_row, label_row):
                    if l == -100:
                        continue
                    row_labels.append(CONLL_LABEL_LIST[l])
                    row_preds.append(CONLL_LABEL_LIST[p] if p < len(CONLL_LABEL_LIST) else "O")
                true_labels.append(row_labels)
                true_preds.append(row_preds)
            try:
                from seqeval.metrics import f1_score
                return {"f1": f1_score(true_labels, true_preds)}
            except ImportError:
                correct = sum(p == l for ps, ls in zip(true_preds, true_labels) for p, l in zip(ps, ls))
                total = sum(len(ls) for ls in true_labels)
                return {"token_accuracy": correct / total if total > 0 else 0.0}
        return compute_metrics

    return None


def _train_without_teacher(cfg, experiment_name, device):
    """LoRA or standard training pipeline (no teacher model needed)."""
    student_quant, student_compute_dtype = get_model_precision(cfg, "student")
    model_type = cfg.student.get("model_type", "causal_lm")
    model_dtype = student_compute_dtype

    # Load tokenizer / image processor
    if model_type == "image_classification":
        tokenizer = AutoImageProcessor.from_pretrained(cfg.student.base_model, use_fast=True)
        auto_cls = get_auto_model_class(model_type)
        model_kwargs = {}
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.student.base_model, trust_remote_code=cfg.student.get("trust_remote_code", False))
        auto_cls = get_auto_model_class(model_type)
        trust = cfg.student.get("trust_remote_code", False)
        model_kwargs = {"trust_remote_code": trust}
        if not trust:
            model_kwargs["attn_implementation"] = getattr(cfg.training, "attn_implementation", "flash_attention_2")

    # Forward num_labels for downstream tasks
    num_labels = cfg.data.get("num_labels") if cfg.get("data") else None
    if num_labels is not None:
        model_kwargs["num_labels"] = num_labels
    # When loading an MLM checkpoint as a classification model (e.g.
    # eval_glue.sh fine-tuning a distilled BERT encoder on MNLI), the
    # saved config may carry a default num_labels (2) while we want 3.
    # ``ignore_mismatched_sizes`` lets transformers discard the stale
    # classifier head and reinitialize it with the correct shape.
    if model_type == "sequence_classification":
        model_kwargs["ignore_mismatched_sizes"] = True

    # Load student model
    arch = cfg.student.architecture
    if arch == "from_pretrained":
        model = auto_cls.from_pretrained(cfg.student.pretrained_path, **model_kwargs).to(device)
    elif arch == "from_config":
        # Random init from config — only load config, not full pretrained weights.
        # This avoids ignore_mismatched_sizes errors when num_labels differs
        # (e.g. CIFAR-100 num_labels=100 vs ImageNet num_labels=1000).
        from copy import deepcopy
        student_config = deepcopy(AutoConfig.from_pretrained(cfg.student.base_model))
        if cfg.student.get("config_overrides"):
            config_diff = OmegaConf.to_container(cfg.student.config_overrides, resolve=True)
        elif cfg.student.get("reconfig") and cfg.student.reconfig.get("config_diff"):
            config_diff = OmegaConf.to_container(cfg.student.reconfig.config_diff, resolve=True)
        else:
            config_diff = {}
        _num_labels = cfg.data.get("num_labels") if cfg.get("data") else None
        if _num_labels is not None:
            config_diff["num_labels"] = _num_labels
        for key, value in config_diff.items():
            setattr(student_config, key, value)
        model = auto_cls.from_config(student_config).to(device)
        logger.info("Created student from config (random init): %s", config_diff)
    else:
        base_model = auto_cls.from_pretrained(cfg.student.base_model, torch_dtype=model_dtype, **model_kwargs).to(device)
        if arch == "reconfig":
            from silverspoon_kd.utils import reconfig_model
            model = reconfig_model(
                base_model, base_model.name_or_path.replace("0.6B", cfg.student.reconfig.name_suffix),
                OmegaConf.to_container(cfg.student.reconfig.config_diff, resolve=True),
                copy_matching_weights=True, freeze_copied_weights=True).to(device)
            del base_model
        elif arch == "pruned":
            from silverspoon_kd.utils import prune_model
            example_inputs = _make_pruning_example_inputs(cfg, base_model)
            config_diff = OmegaConf.to_container(cfg.student.reconfig.config_diff, resolve=True)
            _num_labels = cfg.data.get("num_labels") if cfg.get("data") else None
            if _num_labels is not None:
                config_diff["num_labels"] = _num_labels
            model = prune_model(
                base_model, base_model.name_or_path.replace("0.6B", cfg.student.reconfig.name_suffix),
                config_diff,
                example_inputs=example_inputs,
                output_transform=lambda out: out.logits if hasattr(out, "logits") else (out.start_logits if hasattr(out, "start_logits") else out),
                round_to=cfg.student.reconfig.get("round_to", None),
                freeze_copied_weights=cfg.student.reconfig.get("freeze_copied_weights", True)).to(device)
            del base_model
        elif arch == "linearized":
            from silverspoon_kd.alignments.utils import load_student_weights_from_checkpoint
            model = _create_linearized_student(cfg, base_model, model_dtype)
            if cfg.init.method == "from_experiment":
                ckpt_path = resolve_checkpoint_path(cfg)
                load_student_weights_from_checkpoint(
                    student_model=model, checkpoint_dir=ckpt_path,
                    student_model_name=model.name_or_path, strict=False)
            # Set linearized attention mode — handle both Llama-style (.model.layers)
            # and GPT-2-style (.transformer.h) architectures
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                layers = model.model.layers
                attn_attr = "self_attn"
            elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                layers = model.transformer.h
                attn_attr = "attn"
            else:
                raise AttributeError(
                    f"Cannot find layers in {type(model).__name__}: "
                    "expected .model.layers or .transformer.h"
                )
            for layer in layers:
                getattr(layer, attn_attr).mode = "linear"
            # Freeze backbone, only train feature maps (and optional window factors)
            for p in model.parameters():
                p.requires_grad = False
            trainable = 0
            for name, p in model.named_parameters():
                if "feature_map" in name or "window_factor" in name:
                    p.requires_grad = True
                    trainable += 1
            logger.info("Linearized student: %d trainable parameter groups "
                        "(feature maps + window factors), backbone frozen", trainable)
            model = model.to(device)
            del base_model
        else:
            model = base_model

    # CIFAR / depthwise separable
    if cfg.student.get("cifar_head", False):
        head_type = cfg.student.get("cifar_head_type", "vgg")
        if head_type == "resnet":
            from models import adapt_resnet_for_cifar
            model = adapt_resnet_for_cifar(model, cfg.student.get("num_classes", 10), dtype=model_dtype)
        else:
            model = adapt_vgg16_for_cifar(model, cfg.student.get("num_classes", 10), dtype=model_dtype)
    if cfg.student.get("use_depthwise_separable", False):
        model = apply_depthwise_separable_replacement(model, dtype=model_dtype)

    # Load from prior experiment checkpoint (e.g., BKD → finetune).
    # Skip when architecture=from_pretrained — the model was already loaded
    # with the correct weights above; the init.from_experiment field is only
    # used for experiment naming in that case.
    if cfg.init.method == "from_experiment" and arch != "from_pretrained":
        ckpt_path = resolve_checkpoint_path(cfg)
        logger.info("Loading from checkpoint: %s", ckpt_path)
        result = _load_from_checkpoint(model, ckpt_path)
        logger.info("Missing: %d, Unexpected: %d",
                     len(result.get("missing_keys", [])), len(result.get("unexpected_keys", [])))

    # Quantization simulation
    if student_quant is not None:
        model = apply_quantization_simulation(model, student_quant)

    # Load datasets
    num_training_steps = None
    result = load_datasets(cfg, tokenizer)
    if len(result) == 3:
        train_dataset, eval_dataset, num_training_steps = result
    else:
        train_dataset, eval_dataset = result

    output_dir = str(Path(__file__).parent / "runs" / experiment_name)

    if cfg.distiller.type == "lora":
        return train_lora(cfg, model, train_dataset, eval_dataset, tokenizer, output_dir, experiment_name)
    compute_metrics = _build_compute_metrics(cfg)
    return train_standard(cfg, model, train_dataset, eval_dataset, tokenizer,
                          output_dir, experiment_name, num_training_steps=num_training_steps,
                          compute_metrics=compute_metrics)


def _train_with_distillation(cfg, experiment_name, device):
    """Distillation pipeline (requires teacher + student)."""
    teacher_quant, teacher_compute_dtype = get_model_precision(cfg, "teacher")
    student_quant, student_compute_dtype = get_model_precision(cfg, "student")
    model_type = get_model_type(cfg)

    # Multi-GPU placement
    teacher_placement_cfg = cfg.training.get("teacher_placement", None)
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if teacher_placement_cfg is not None:
        # Load teacher to CPU; we'll place it via strategies after loading.
        teacher_device = torch.device("cpu")
        student_device = device
        logger.info("Teacher placement configured; teacher loaded to CPU first")
    elif n_gpu > 1:
        teacher_device, student_device = torch.device("cuda:0"), torch.device("cuda:1")
        logger.info("Multi-GPU: teacher on %s, student on %s", teacher_device, student_device)
    else:
        teacher_device = student_device = device

    # Tokenizer / processor
    if model_type == "image_classification":
        # Use base_model for the processor — teacher.name may be a local
        # checkpoint path that doesn't contain preprocessor_config.json.
        processor_name = cfg.teacher.get("base_model", cfg.teacher.name)
        tokenizer_or_processor = AutoImageProcessor.from_pretrained(processor_name, use_fast=True)
    else:
        tokenizer_or_processor = AutoTokenizer.from_pretrained(cfg.teacher.name)
        if tokenizer_or_processor.pad_token is None:
            tokenizer_or_processor.pad_token = tokenizer_or_processor.eos_token

    # Models
    teacher_model = load_teacher_model(cfg, teacher_device, teacher_compute_dtype)
    student_model = create_student_model(cfg, teacher_model, student_device, student_compute_dtype)
    if student_quant is not None:
        student_model = apply_quantization_simulation(student_model, student_quant)

    # Disable KV cache on both models — distillation doesn't use incremental
    # decoding, and the cache bloats accelerate's fp32 conversion during eval.
    if hasattr(teacher_model, "config"):
        teacher_model.config.use_cache = False
    if hasattr(student_model, "config"):
        student_model.config.use_cache = False

    # LoLCATs: enable distill_mode on student attention modules so the forward
    # pass uses quadratic_attention (explicit L×L matrix, pure PyTorch gradients)
    # instead of the efficient causal kernel (fla chunk_linear_attn) whose
    # backward pass produces poor gradients for feature map training.
    _distill_mode_set = 0
    for module in student_model.modules():
        if hasattr(module, "_distill_mode") and hasattr(module, "feature_map_q"):
            module._distill_mode = True
            _distill_mode_set += 1
    if _distill_mode_set > 0:
        logger.info("LoLCATs: set _distill_mode=True on %d attention modules "
                     "(using quadratic_attention for training)", _distill_mode_set)

    # Apply explicit teacher placement (PP across dedicated GPUs)
    if teacher_placement_cfg is not None:
        from omegaconf import OmegaConf
        tp_dict = OmegaConf.to_container(teacher_placement_cfg, resolve=True)
        from silverspoon_kd.distributed.strategies import place_teacher_pp
        teacher_gpus = tp_dict["teacher_only_devices"]
        logger.info("Placing teacher via PP on GPUs %s", teacher_gpus)
        teacher_model = place_teacher_pp(teacher_model, teacher_gpus, "cuda")
        teacher_model.eval()

    # Datasets
    num_training_steps = None
    result = load_datasets(cfg, tokenizer_or_processor)
    if len(result) == 3:
        train_dataset, eval_dataset, num_training_steps = result
    else:
        train_dataset, eval_dataset = result

    # Alignments (only for layer-wise distillers)
    distiller_type = cfg.distiller.type
    if distiller_type in ("bkd", "bkd_attn", "hkd"):
        alignments = create_alignments(cfg, teacher_model, student_model, student_device, dtype=student_compute_dtype)
        # Optional per-alignment loss weight (e.g. mse_factor=1000 for LoLCATs)
        alignment_loss_weight = cfg.distiller.alignment.get("loss_weight", None)
        if alignment_loss_weight is not None:
            for a in alignments:
                a.loss_weight = float(alignment_loss_weight)
            logger.info("Set alignment loss_weight=%.1f on %d alignments", alignment_loss_weight, len(alignments))
        # Projector init override is applied after lazy creation (see _ReinitProjectorsCallback)
    else:
        alignments = None

    output_dir = str(Path(__file__).parent / "runs" / experiment_name)
    training_args = get_training_args(cfg, output_dir, experiment_name, max_steps_override=num_training_steps)

    compute_metrics = _build_compute_metrics(cfg)
    # When teacher_placement_cfg is set, the teacher is already on the right
    # GPU(s) via place_teacher_pp. Pass teacher_device so train_distillation
    # wraps prepare_teacher_inputs to route inputs to the teacher's device.
    if teacher_placement_cfg is not None:
        teacher_device = next(teacher_model.parameters()).device
    return train_distillation(
        cfg, teacher_model, student_model, alignments,
        train_dataset, eval_dataset, tokenizer_or_processor, training_args,
        teacher_device=teacher_device, student_device=student_device,
        compute_metrics=compute_metrics)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.run.seed)
    experiment_name = get_experiment_name(cfg)

    if getattr(cfg.run, "name_only", False):
        print(experiment_name)
        return None

    logger.info("Experiment: %s", experiment_name)
    OmegaConf.set_struct(cfg, False)
    cfg.experiment.name = experiment_name
    OmegaConf.set_struct(cfg, True)

    # Log config
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if cfg.distiller.type in ("reskd", "lora", "standard") and "loss" in cfg_dict:
        del cfg_dict["loss"]
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg_dict))

    device = setup_device(cfg)
    output_dir = str(Path(__file__).parent / "runs" / experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if cfg.distiller.type in ("lora", "standard"):
        trainer = _train_without_teacher(cfg, experiment_name, device)
    else:
        trainer = _train_with_distillation(cfg, experiment_name, device)

    # Log peak GPU memory
    if torch.cuda.is_available():
        import json as _json
        peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_gib = peak_mib / 1024
        logger.info("Peak GPU memory: %.0f MiB (%.2f GiB)", peak_mib, peak_gib)
        mem_path = os.path.join(output_dir, "gpu_peak_memory.json")
        with open(mem_path, "w") as f:
            _json.dump({"peak_memory_mib": round(peak_mib, 1),
                        "peak_memory_gib": round(peak_gib, 3)}, f)

    logger.info("Training completed successfully!")
    return trainer


if __name__ == "__main__":
    main()
