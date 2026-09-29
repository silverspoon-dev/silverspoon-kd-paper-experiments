"""Teacher/student model loading and architecture factories."""

import logging
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForImageClassification,
                          AutoModelForMaskedLM, AutoModelForQuestionAnswering,
                          AutoModelForSequenceClassification,
                          AutoModelForTokenClassification)

logger = logging.getLogger(__name__)


# ── Model type helpers ───────────────────────────────────────────────────────

def get_model_type(cfg: DictConfig) -> str:
    """Determine the model type from distiller, teacher, or student config."""
    if cfg.distiller.get("model_type"):
        return cfg.distiller.model_type
    teacher = OmegaConf.select(cfg, "teacher")
    if teacher and teacher.get("model_type"):
        return teacher.model_type
    student = OmegaConf.select(cfg, "student")
    if student and student.get("model_type"):
        return student.model_type
    return "causal_lm"


_AUTO_MODEL_CLASSES = {
    "causal_lm": AutoModelForCausalLM,
    "mlm": AutoModelForMaskedLM,
    "image_classification": AutoModelForImageClassification,
    "sequence_classification": AutoModelForSequenceClassification,
    "question_answering": AutoModelForQuestionAnswering,
    "token_classification": AutoModelForTokenClassification,
}


def get_auto_model_class(model_type: str):
    cls = _AUTO_MODEL_CLASSES.get(model_type)
    if cls is None:
        raise ValueError(f"Unknown model type: {model_type}")
    return cls


# ── VGG/CIFAR helpers ────────────────────────────────────────────────────────

def adapt_vgg_for_cifar(model, num_classes=10, dtype=None):
    """Adapt timm VGG for CIFAR-style datasets (32x32 images).

    The standard CIFAR VGG adaptation (used in RepDistiller, RelKD, etc.)
    preserves more spatial resolution than the ImageNet VGG by modifying
    the pooling layers.  ImageNet VGG has 5 MaxPool2d(2,2) layers, giving
    32→16→8→4→2→1 on CIFAR.  The CIFAR adaptation:
    - Removes the 4th MaxPool (keeps 4x4 feature maps for the last 2 conv blocks)
    - Replaces the 5th MaxPool with AdaptiveAvgPool2d(1) for the classifier

    This gives 32→16→8→4→4→1, matching the spatial resolution used in
    standard KD papers (Park et al., 2019; Tian et al., 2020).
    """
    device = next(model.parameters()).device

    # Replace head
    model.timm_model.pre_logits = nn.Identity()
    model.timm_model.head.fc = nn.Linear(512, num_classes, device=device, dtype=dtype)
    model.config.num_labels = num_classes

    # Fix spatial resolution: find MaxPool2d layers in features
    features = model.timm_model.features
    pool_indices = [i for i, m in enumerate(features) if isinstance(m, nn.MaxPool2d)]
    if len(pool_indices) >= 5:
        # Remove 4th MaxPool → keep 4x4 feature maps for last conv blocks
        features[pool_indices[3]] = nn.Identity()
        # Replace 5th MaxPool → AdaptiveAvgPool for classifier
        features[pool_indices[4]] = nn.AdaptiveAvgPool2d(1)
    elif len(pool_indices) >= 4:
        # VGG variants with only 4 pools: replace last with AdaptiveAvgPool
        features[pool_indices[3]] = nn.AdaptiveAvgPool2d(1)

    return model


# Backwards-compatible alias
adapt_vgg16_for_cifar = adapt_vgg_for_cifar


def adapt_resnet_for_cifar(model, num_classes=100, dtype=None):
    """Adapt timm ResNet for CIFAR-style datasets (32x32 images).

    ImageNet ResNet uses 7x7 conv stride 2 + maxpool that destroys
    spatial resolution on 32x32 inputs (32→8 before layer1).  The
    standard CIFAR adaptation (He et al., 2016) replaces the stem:
    - conv1: 7x7 stride 2 → 3x3 stride 1
    - maxpool: removed (Identity)
    This preserves 32x32 resolution through layer1.
    """
    device = next(model.parameters()).device

    # Replace stem: 7x7 stride 2 → 3x3 stride 1
    old_conv1 = model.timm_model.conv1
    model.timm_model.conv1 = nn.Conv2d(
        3, old_conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=False,
        device=device, dtype=dtype,
    )
    # Remove maxpool (keeps 32x32 through layer1)
    model.timm_model.maxpool = nn.Identity()

    # Replace classifier head
    in_features = model.timm_model.fc.in_features
    model.timm_model.fc = nn.Linear(in_features, num_classes, device=device, dtype=dtype)
    model.config.num_labels = num_classes
    return model


def _make_ds_block(in_ch, out_ch, ks, device=None, dtype=None):
    """Create a single depthwise-separable block: DW(3x3) + PW(1x1) + ReLU.

    Initialization is chosen so that each DW+PW+ReLU block preserves
    activation variance (important for deep DS networks):
    - DW: normal(0, 1/k) makes the spatial convolution variance-neutral.
    - PW: kaiming_normal(fan_in, relu) compensates for the ReLU halving.
    """
    dw = nn.Conv2d(in_ch, in_ch, kernel_size=ks, stride=1, padding="same",
                   groups=in_ch, bias=False)
    pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
    k = ks[0] if isinstance(ks, tuple) else ks
    nn.init.normal_(dw.weight, 0, 1.0 / k)
    nn.init.kaiming_normal_(pw.weight, mode="fan_in", nonlinearity="relu")
    nn.init.zeros_(pw.bias)
    block = nn.Sequential(dw, pw, nn.ReLU(inplace=False))
    if device is not None or dtype is not None:
        block = block.to(device=device, dtype=dtype)
    return block


def apply_depthwise_separable_replacement(model, dtype=None):
    """Replace Conv2d layers in VGG16 features with depthwise separable convolutions.

    Matches the Parallel Blockwise KD paper (codestar12): each Conv2d (in_ch >= 16)
    is replaced with **2 stacked** ``SeparableConv2D(out_ch, 3x3, same) + ReLU``
    blocks.  The original ReLU following each replaced Conv2d is deleted (replaced
    with ``nn.Identity``) since the replacement already ends with ReLU.
    """
    device = next(model.parameters()).device
    replaced_indices = set()

    for i, layer in enumerate(model.timm_model.features):
        if not isinstance(layer, nn.Conv2d):
            continue
        in_ch = layer.in_channels
        if in_ch < 16:
            continue
        out_ch = layer.out_channels
        ks = layer.kernel_size

        # 2 stacked DS blocks (matching paper's add_layers(layers=2)):
        #   Block 1: in_ch → out_ch
        #   Block 2: out_ch → out_ch
        replacement = nn.Sequential(
            *_make_ds_block(in_ch, out_ch, ks),   # DW1 + PW1 + ReLU
            *_make_ds_block(out_ch, out_ch, ks),  # DW2 + PW2 + ReLU
        )
        model.timm_model.features[i] = replacement.to(device=device, dtype=dtype)
        replaced_indices.add(i)

    # Delete the original ReLU after each replaced Conv2d (paper deletes them;
    # keeping them creates a harmless but unnecessary double-ReLU).
    for i, layer in enumerate(model.timm_model.features):
        if isinstance(layer, nn.ReLU) and (i - 1) in replaced_indices:
            model.timm_model.features[i] = nn.Identity()

    return model


# ── Teacher loading ──────────────────────────────────────────────────────────

def load_teacher_model(cfg: DictConfig, device: torch.device, dtype: torch.dtype):
    """Load the teacher model based on model type."""
    model_type = get_model_type(cfg)
    model_name = cfg.teacher.name
    trust = cfg.teacher.get("trust_remote_code", False)
    logger.info("Loading teacher model: %s (type: %s)", model_name, model_type)

    # Honor the same attn_implementation as the student so the teacher uses
    # SDPA (or whatever's configured) and avoids materializing full attention
    # score tensors during forward. Eager attention can cause large per-layer
    # memory growth that impacts HKD eval where the teacher forward runs every
    # batch.
    extra_kwargs = {}
    if not trust:
        extra_kwargs["attn_implementation"] = getattr(
            cfg.training, "attn_implementation", "sdpa"
        )

    if model_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=trust, **extra_kwargs
        ).to(device)
    elif model_type == "mlm":
        model = AutoModelForMaskedLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=trust, **extra_kwargs
        ).to(device)
    elif model_type == "image_classification":
        cifar_head = cfg.teacher.get("cifar_head", False)
        base_model_name = cfg.teacher.get("base_model", model_name)
        if cifar_head and model_name != base_model_name:
            from safetensors.torch import load_file
            model = AutoModelForImageClassification.from_pretrained(
                base_model_name, torch_dtype=dtype, trust_remote_code=trust).to(device)
            num_classes = cfg.teacher.get("num_classes", 10)
            head_type = cfg.teacher.get("cifar_head_type", "vgg")
            if head_type == "resnet":
                model = adapt_resnet_for_cifar(model, num_classes=num_classes, dtype=dtype)
            else:
                model = adapt_vgg16_for_cifar(model, num_classes=num_classes, dtype=dtype)
            state_dict = load_file(str(Path(model_name) / "model.safetensors"))
            model.load_state_dict(state_dict)
        else:
            model = AutoModelForImageClassification.from_pretrained(
                model_name, torch_dtype=dtype, trust_remote_code=trust).to(device)
            if cifar_head:
                num_classes = cfg.teacher.get("num_classes", 10)
                head_type = cfg.teacher.get("cifar_head_type", "vgg")
                if head_type == "resnet":
                    model = adapt_resnet_for_cifar(model, num_classes=num_classes, dtype=dtype)
                else:
                    model = adapt_vgg16_for_cifar(model, num_classes=num_classes, dtype=dtype)
    else:
        auto_cls = get_auto_model_class(model_type)
        num_labels = cfg.data.get("num_labels") if cfg.get("data") else None
        load_kwargs = {"torch_dtype": dtype, "trust_remote_code": trust}
        if num_labels is not None:
            load_kwargs["num_labels"] = num_labels
        model = auto_cls.from_pretrained(model_name, **load_kwargs).to(device)

    if cfg.distiller.get("model_overrides"):
        for key, value in cfg.distiller.model_overrides.items():
            setattr(model.config, key, value)
    return model


# ── Checkpoint resolution ────────────────────────────────────────────────────

def resolve_checkpoint_path(cfg: DictConfig) -> Optional[str]:
    """Resolve the checkpoint path from init configuration."""
    if cfg.init.method != "from_experiment":
        return None
    parent = cfg.init.from_experiment
    step = str(cfg.init.checkpoint_step)
    runs_dir = Path(__file__).parent / "runs"

    if "checkpoint-" in step:
        full_path = Path(step)
        if not full_path.is_absolute():
            full_path = Path(__file__).parent / full_path
        return str(full_path)
    if step == "latest":
        parent_dir = runs_dir / parent
        checkpoints = sorted(parent_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {parent_dir}")
        return str(checkpoints[-1])
    return str(runs_dir / parent / f"checkpoint-{step}")


def _load_from_checkpoint(student_model, checkpoint_dir):
    """Load student weights from a checkpoint, handling both BKD and standard formats.

    BKD checkpoints save the full student model in a ``student_model_<name>/``
    subdirectory inside each checkpoint.  Standard (response-based KD/HKD) checkpoints save
    directly as ``model.safetensors``.
    """
    from safetensors.torch import load_file

    checkpoint_dir = Path(checkpoint_dir)

    # BKD checkpoint: look for student_model_* subdirectory first
    student_model_dirs = sorted(checkpoint_dir.glob("student_model_*"))
    if student_model_dirs:
        sf_path = student_model_dirs[0] / "model.safetensors"
        if sf_path.exists():
            logger.info("Loading BKD student model from: %s", sf_path)
            state_dict = load_file(str(sf_path))
            result = student_model.load_state_dict(state_dict, strict=False)
            return {"missing_keys": list(result.missing_keys),
                    "unexpected_keys": list(result.unexpected_keys)}

    # Standard checkpoint: model.safetensors at top level
    sf_path = checkpoint_dir / "model.safetensors"
    if sf_path.exists():
        logger.info("Loading standard checkpoint from: %s", sf_path)
        state_dict = load_file(str(sf_path))
        result = student_model.load_state_dict(state_dict, strict=False)
        return {"missing_keys": list(result.missing_keys),
                "unexpected_keys": list(result.unexpected_keys)}

    # Also check parent run directory for final student_model/
    run_dir = checkpoint_dir.parent
    final_model = run_dir / "student_model" / "model.safetensors"
    if final_model.exists():
        logger.info("Loading final student model from: %s", final_model)
        state_dict = load_file(str(final_model))
        result = student_model.load_state_dict(state_dict, strict=False)
        return {"missing_keys": list(result.missing_keys),
                "unexpected_keys": list(result.unexpected_keys)}

    raise FileNotFoundError(
        f"No student model found in {checkpoint_dir}. "
        f"Checked: student_model_*/model.safetensors, model.safetensors, "
        f"and ../student_model/model.safetensors")


# ── Linearized student ───────────────────────────────────────────────────────

def _load_generated_module(path: str, module_name: str):
    """Import a generated architecture file (``compiled/<name>.py``) as a module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _find_model_class(module, prefix: str, suffixes=("ForCausalLM", "LMHeadModel")) -> type:
    """Return the class in ``module`` named ``<prefix>*<suffix>``, e.g. ``LoLCATsGPT2LMHeadModel``."""
    for suffix in suffixes:
        for name in dir(module):
            if name.startswith(prefix) and name.endswith(suffix):
                return getattr(module, name)
    raise RuntimeError(f"No {prefix}*({'|'.join(suffixes)}) class found in {module.__file__}")


def _create_linearized_student(cfg, teacher_model, model_dtype):
    """Create a LoLCATs student from the teacher.

    The student class comes from the pre-generated architecture module
    ``compiled/lolcats_<teacher slug>.py``, in which every softmax attention
    layer is replaced by LoLCATs hybrid attention: sliding-window softmax over
    the most recent ``window_size`` tokens plus linear attention over older
    ones.  Every weight the student shares with the teacher is copied over and,
    by default, frozen, so that only the new feature-map parameters train.
    """
    method = cfg.student.get("linearization_method", "lolcats")
    if method != "lolcats":
        raise ValueError(f"Unsupported linearization method {method!r}; only 'lolcats' is available")
    if cfg.student.get("feature_map", None) is not None:
        raise ValueError("student.feature_map is fixed by the generated architecture and cannot be overridden")

    project_root = os.path.dirname(os.path.abspath(__file__))
    slug = teacher_model.name_or_path.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    arch_file = os.path.join(project_root, "compiled", f"{method}_{slug}.py")
    if not os.path.exists(arch_file):
        raise FileNotFoundError(
            f"No generated LoLCATs architecture for teacher {teacher_model.name_or_path!r}; "
            f"expected {arch_file}")
    module = _load_generated_module(arch_file, module_name=f"compiled.{method}_{slug}")
    cls = _find_model_class(module, prefix="LoLCATs")

    # The generated attention reads the window size from the config, so set it
    # before construction.  None selects pure linear attention.  Persisting it
    # in the config also lets from_pretrained rebuild the hybrid attention.
    config = deepcopy(teacher_model.config)
    window_size = cfg.student.get("window_size", None)
    config.lolcats_window_size = int(window_size) if window_size is not None else None
    student = cls(config)

    # Copy every weight the student shares with the teacher; the feature maps
    # keep their fresh initialisation.
    result = student.load_state_dict(teacher_model.state_dict(), strict=False)
    copied = set(student.state_dict()) - set(result.missing_keys)
    if cfg.student.get("freeze_copied", True):
        for name, param in student.named_parameters():
            if name in copied:
                param.requires_grad = False
    if model_dtype is not None:
        student = student.to(model_dtype)

    frozen = sum(1 for _, p in student.named_parameters() if not p.requires_grad)
    trainable = sum(1 for _, p in student.named_parameters() if p.requires_grad)
    logger.info("Copied teacher weights (frozen=%d, trainable/new=%d)", frozen, trainable)
    return student


# ── Pruning helper ───────────────────────────────────────────────────────────

def _make_pruning_example_inputs(cfg, model):
    model_type = get_model_type(cfg)
    device = next(model.parameters()).device
    if model_type in ("causal_lm", "mlm", "sequence_classification", "question_answering", "token_classification"):
        return torch.randint(0, model.config.vocab_size, (1, 64), device=device)
    elif model_type == "image_classification":
        return torch.randn(1, 3, 224, 224, device=device)
    raise ValueError(f"Unsupported model_type for pruning: {model_type}")


# ── Student creation (dispatch table) ────────────────────────────────────────

def create_student_model(cfg: DictConfig, teacher_model, device: torch.device, dtype: torch.dtype):
    """Create and initialize the student model.

    Architecture: student config (reconfig/pruned/from_pretrained/from_config/linearized).
    Initialization: init config (scratch/teacher/from_experiment/copy_matching).
    """
    from silverspoon_kd.utils import prune_model, reconfig_model

    model_type = get_model_type(cfg)
    auto_cls = get_auto_model_class(model_type)
    architecture = cfg.student.architecture
    logger.info("Creating student architecture: %s (type: %s)", architecture, model_type)

    # ── Architecture dispatch ────────────────────────────────────────────
    if architecture == "reconfig":
        config_diff = OmegaConf.to_container(cfg.student.reconfig.config_diff, resolve=True)
        student = reconfig_model(
            teacher_model,
            teacher_model.name_or_path.replace("0.6B", cfg.student.reconfig.name_suffix),
            config_diff, copy_matching_weights=True, freeze_copied_weights=True,
        ).to(dtype)

    elif architecture == "pruned":
        config_diff = OmegaConf.to_container(cfg.student.reconfig.config_diff, resolve=True)
        num_labels = cfg.data.get("num_labels") if cfg.get("data") else None
        if num_labels is not None:
            config_diff["num_labels"] = num_labels
        example_inputs = _make_pruning_example_inputs(cfg, teacher_model)
        student = prune_model(
            teacher_model,
            teacher_model.name_or_path.replace("0.6B", cfg.student.reconfig.name_suffix),
            config_diff, example_inputs=example_inputs,
            output_transform=lambda out: out.logits if hasattr(out, "logits") else (out.start_logits if hasattr(out, "start_logits") else out),
            round_to=cfg.student.reconfig.get("round_to", None),
            freeze_copied_weights=cfg.student.reconfig.get("freeze_copied_weights", True),
        ).to(dtype)

    elif architecture == "from_pretrained":
        load_kwargs = {"trust_remote_code": cfg.student.get("trust_remote_code", False)}
        num_labels = cfg.data.get("num_labels") if cfg.get("data") else None
        if num_labels is not None:
            load_kwargs["num_labels"] = num_labels
        student = auto_cls.from_pretrained(
            cfg.student.pretrained_path, **load_kwargs,
        ).to(dtype)

    elif architecture == "from_pretrained_reconfig":
        base = auto_cls.from_pretrained(
            cfg.student.pretrained_path, trust_remote_code=cfg.student.get("trust_remote_code", False))
        config_diff = OmegaConf.to_container(cfg.student.reconfig.config_diff, resolve=True)
        student = reconfig_model(base, cfg.student.reconfig.get("name", cfg.student.pretrained_path), config_diff).to(dtype)
        del base

    elif architecture == "from_config":
        model_type_id = cfg.student.get("model_type_id")
        student_base = cfg.student.get("base_model")
        if model_type_id:
            # Build config from scratch using AutoConfig.for_model() — avoids
            # inheriting stale fields from a full-size HF checkpoint.
            student_config = AutoConfig.for_model(model_type_id,
                **OmegaConf.to_container(cfg.student.get("config_overrides", {}), resolve=True))
        elif student_base and student_base != cfg.teacher.name:
            student_config = AutoConfig.from_pretrained(student_base)
        else:
            student_config = deepcopy(teacher_model.config)
        num_labels = cfg.data.get("num_labels") if cfg.get("data") else None
        if num_labels is not None:
            student_config.num_labels = num_labels
        if not model_type_id:
            for key, val in cfg.student.get("config_overrides", {}).items():
                setattr(student_config, key, val)
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            student = auto_cls.from_config(student_config)
        finally:
            torch.set_default_dtype(old_dtype)

    elif architecture == "linearized":
        student = _create_linearized_student(cfg, teacher_model, dtype)

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # ── CIFAR / depthwise separable ──────────────────────────────────────
    if cfg.student.get("cifar_head", False):
        num_classes = cfg.student.get("num_classes", 10)
        head_type = cfg.student.get("cifar_head_type", "vgg")
        if head_type == "resnet":
            student = adapt_resnet_for_cifar(student, num_classes=num_classes, dtype=dtype)
        else:
            student = adapt_vgg16_for_cifar(student, num_classes=num_classes, dtype=dtype)
    if cfg.student.get("use_depthwise_separable", False):
        student = apply_depthwise_separable_replacement(student, dtype=dtype)

    # ── Weight initialization dispatch ───────────────────────────────────
    init_method = cfg.init.method
    logger.info("Initializing weights: %s", init_method)

    if init_method == "from_experiment":
        ckpt_path = resolve_checkpoint_path(cfg)
        logger.info("Loading from checkpoint: %s", ckpt_path)
        result = _load_from_checkpoint(student, ckpt_path)
        logger.info("Missing: %d, Unexpected: %d",
                     len(result.get("missing_keys", [])), len(result.get("unexpected_keys", [])))
        # Optionally overlay teacher weights
        copy_cfg = cfg.init.copy_teacher_after_load
        if copy_cfg.embed_tokens:
            student.model.embed_tokens.weight = deepcopy(teacher_model.model.embed_tokens.weight)
        if copy_cfg.norm:
            student.model.norm.weight = deepcopy(teacher_model.model.norm.weight)
        if copy_cfg.lm_head:
            student.lm_head.weight = deepcopy(teacher_model.lm_head.weight)
        if copy_cfg.tie_weights:
            student.tie_weights()

    elif init_method == "teacher":
        student.model.embed_tokens.weight = deepcopy(teacher_model.model.embed_tokens.weight)
        student.model.norm.weight = deepcopy(teacher_model.model.norm.weight)
        student.lm_head.weight = deepcopy(teacher_model.lm_head.weight)
        student.tie_weights()

    elif init_method == "copy_matching":
        teacher_state = teacher_model.state_dict()
        student_state = student.state_dict()
        copied, skipped = [], []
        for key in student_state:
            if key in teacher_state and teacher_state[key].shape == student_state[key].shape:
                student_state[key] = teacher_state[key].clone()
                copied.append(key)
            else:
                skipped.append(key)
        student.load_state_dict(student_state)
        logger.info("Copied %d matching params, %d remain random", len(copied), len(skipped))
        if cfg.init.get("freeze_copied", True):
            from silverspoon_kd.utils import freeze_parameters
            freeze_parameters(student, [re.escape(n) + "$" for n in copied])

    elif init_method == "scratch":
        logger.info("Random initialization (scratch)")

    else:
        raise ValueError(f"Unknown init method: {init_method}")

    if cfg.distiller.get("model_overrides"):
        for key, value in cfg.distiller.model_overrides.items():
            setattr(student.config, key, value)

    return student.to(device)
