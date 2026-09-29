"""Shared utilities: device selection, dtype resolution, experiment naming."""

import hashlib
import logging
from pathlib import Path

import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# ── Dtype maps ───────────────────────────────────────────────────────────────

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}

SIMULATED_DTYPES = {"int2", "int3", "int4", "int5", "int6", "int7", "int8"}
FLOAT8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}


def is_simulated_dtype(dtype_str: str) -> bool:
    return dtype_str in SIMULATED_DTYPES or dtype_str in ("float8_e4m3fn", "float8_e5m2")


def needs_quantization_sim(dtype) -> bool:
    if isinstance(dtype, str):
        return is_simulated_dtype(dtype)
    return dtype in FLOAT8_DTYPES


def compute_dtype_for(dtype) -> torch.dtype:
    if isinstance(dtype, str) and dtype in SIMULATED_DTYPES:
        return torch.bfloat16
    if isinstance(dtype, torch.dtype) and dtype in FLOAT8_DTYPES:
        return torch.bfloat16
    if isinstance(dtype, str):
        return DTYPE_MAP.get(dtype, torch.bfloat16)
    return dtype


def resolve_dtype(cfg_value) -> tuple:
    """Resolve a dtype config value into (quantization_spec, compute_dtype)."""
    if cfg_value is None:
        return None, None
    if cfg_value in SIMULATED_DTYPES:
        return cfg_value, torch.bfloat16
    dtype = DTYPE_MAP.get(cfg_value)
    if dtype is not None and dtype in FLOAT8_DTYPES:
        return cfg_value, torch.bfloat16
    return None, DTYPE_MAP.get(cfg_value, torch.bfloat16)


def get_dtype(cfg: DictConfig) -> torch.dtype:
    return DTYPE_MAP.get(cfg.run.dtype, torch.bfloat16)


def get_model_precision(cfg: DictConfig, role: str) -> tuple:
    """Get (quant_spec, compute_dtype) for a model role ('teacher' or 'student')."""
    role_dtype = cfg.run.get(f"{role}_dtype")
    dtype_str = role_dtype if role_dtype is not None else cfg.run.dtype
    return resolve_dtype(dtype_str)


# ── Device selection ─────────────────────────────────────────────────────────

def get_least_used_gpu() -> int:
    if not torch.cuda.is_available():
        return -1
    device_count = torch.cuda.device_count()
    if device_count == 1:
        return 0
    best, max_free = 0, 0
    for gpu_id in range(device_count):
        free, _ = torch.cuda.mem_get_info(gpu_id)
        logger.info("GPU %d: %.2f GB free", gpu_id, free / 1024**3)
        if free > max_free:
            max_free, best = free, gpu_id
    return best


def get_device(use_cpu=False, use_metal=False, select=None) -> torch.device:
    if select is not None:
        device = torch.device(f"cuda:{select}")
        torch.cuda.set_device(device)
        return device
    if use_cpu:
        return torch.device("cpu")
    if use_metal:
        return torch.device("mps")
    best = get_least_used_gpu()
    if best != -1:
        device = torch.device(f"cuda:{best}")
        torch.cuda.set_device(device)
        logger.info("Selected GPU %d", best)
        return device
    return torch.device("cpu")


def setup_device(cfg: DictConfig) -> torch.device:
    if cfg.run.device == "auto":
        return get_device()
    return torch.device(cfg.run.device)


# ── Experiment naming ────────────────────────────────────────────────────────

def experiment_hash(experiment_name: str) -> str:
    return hashlib.sha256(experiment_name.encode()).hexdigest()[:8]


def parse_checkpoint_path(checkpoint_step: str) -> tuple:
    checkpoint_step = str(checkpoint_step)
    if "checkpoint-" in checkpoint_step:
        path = Path(checkpoint_step)
        step = path.name.split("checkpoint-")[-1]
        experiment_name = path.parent.name
        if experiment_name == "runs":
            experiment_name = None
        return experiment_name, step
    return None, checkpoint_step


def get_init_suffix(cfg: DictConfig) -> str:
    init_method = cfg.init.method

    if init_method == "from_experiment":
        parent_experiment = cfg.init.from_experiment
        checkpoint_step = cfg.init.checkpoint_step
        if not parent_experiment and checkpoint_step:
            inferred, checkpoint_step = parse_checkpoint_path(checkpoint_step)
            parent_experiment = inferred
        if not parent_experiment:
            return f"from-unknown.{checkpoint_step}"
        parts = parent_experiment.split("__")
        parent_distiller = parts[1] if len(parts) >= 2 else "unknown"
        # Detect loss variant in the parent name.  The experiment name format is:
        #   framework__distiller__teacher__student[__loss_suffix]__init_suffix
        # When a non-default loss is used (e.g. relkd_angle), it appears as an
        # extra segment before the init suffix.  Include it to avoid collisions
        # (e.g. from-hkd vs from-hkd-relkd_angle).
        loss_variant = None
        if len(parts) >= 6:
            # parts[4] could be loss suffix; parts[5] is init suffix
            candidate = parts[4]
            if candidate not in ("scratch", "teacher") and not candidate.startswith("from-"):
                loss_variant = candidate
        tag = f"{parent_distiller}-{loss_variant}" if loss_variant else parent_distiller
        # Include checkpoint step only when it's a specific step (not "latest")
        if checkpoint_step and checkpoint_step != "latest":
            return f"from-{tag}.step{checkpoint_step}"
        return f"from-{tag}"

    if init_method == "teacher":
        return "teacher"
    # copy_matching (used by BKD) is an implementation detail — treat as scratch
    if init_method == "copy_matching":
        return "scratch"
    return "scratch"


def get_experiment_name(cfg: DictConfig) -> str:
    if cfg.experiment.name is not None:
        return cfg.experiment.name

    framework = cfg.distiller.framework
    distiller_type = cfg.distiller.type
    student_name = cfg.student.short_name
    init_suffix = get_init_suffix(cfg)

    # Include data name for downstream tasks (not pre-training datasets)
    data_name = cfg.data.get("short_name", cfg.data.type) if cfg.get("data") else None
    is_downstream = data_name is not None and data_name not in ("dolma", "textbook", "cifar10", "cifar100", "imagenet",
                                                                  "dolmino_wiki", "dolmino_pes2o",
                                                                  "dolmino_flan", "dolmino_stackexchange",
                                                                  "tulu3_sft_personas_instruction_following",
                                                                  "tulu3_sft_mixture")

    if distiller_type in ("lora", "standard"):
        if is_downstream:
            name = f"{framework}__{distiller_type}__{student_name}__{data_name}__{init_suffix}"
        else:
            name = f"{framework}__{distiller_type}__{student_name}__{init_suffix}"
        if cfg.run.seed != 42:
            name += f"_seed{cfg.run.seed}"
        return name

    teacher_name = cfg.teacher.short_name

    # Include loss in name when non-default (for loss ablation experiments)
    loss_type = cfg.loss.type if cfg.get("loss") else None
    loss_suffix = f"__{loss_type}" if loss_type and loss_type != "mse" else ""

    if is_downstream:
        name = f"{framework}__{distiller_type}__{teacher_name}__{student_name}{loss_suffix}__{data_name}__{init_suffix}"
    else:
        name = f"{framework}__{distiller_type}__{teacher_name}__{student_name}{loss_suffix}__{init_suffix}"

    # Append seed suffix for multi-seed runs (non-default seed only)
    if cfg.run.seed != 42:
        name += f"_seed{cfg.run.seed}"
    return name
