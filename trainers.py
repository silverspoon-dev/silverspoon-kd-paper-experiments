"""Training functions: distillation, LoRA, and standard fine-tuning."""

import inspect
import logging
import os

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import (DataCollatorForLanguageModeling, DefaultDataCollator)
from transformers.trainer_callback import TrainerCallback

from models import get_model_type

logger = logging.getLogger(__name__)


class MultiStepLRCallback(TrainerCallback):
    """Multiply LR by gamma at specified epoch milestones (like PyTorch MultiStepLR).

    Use with ``lr_scheduler_type=constant`` so the Trainer's scheduler does not
    interfere with the manual LR adjustments.  The callback also updates the
    scheduler's ``base_lrs`` so the constant scheduler doesn't reset the LR
    back to its initial value on every step.
    """

    def __init__(self, milestones, gamma=0.1):
        self.milestones = set(milestones)
        self.gamma = gamma
        self._applied = set()

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        if epoch in self.milestones and epoch not in self._applied:
            optimizer = kwargs.get("optimizer")
            lr_scheduler = kwargs.get("lr_scheduler")
            if optimizer is None:
                return
            for param_group in optimizer.param_groups:
                param_group["lr"] *= self.gamma
            # Also update the scheduler's base_lrs so it doesn't reset
            # the LR back to the initial value on the next step.
            if lr_scheduler is not None and hasattr(lr_scheduler, "base_lrs"):
                lr_scheduler.base_lrs = [lr * self.gamma for lr in lr_scheduler.base_lrs]
            self._applied.add(epoch)
            logger.info("MultiStepLR: epoch %d, LR *= %.2f -> %.6f",
                        epoch, self.gamma, optimizer.param_groups[0]["lr"])


class StudentOnlyGradClipCallback(TrainerCallback):
    """Clip gradients on student model only, excluding projector parameters.

    TextBrewer clips only ``self.model_S.parameters()``, not projector params
    added via ``add_param_group()``.  This callback reproduces that behaviour
    by disabling the HF Trainer's built-in clipping and doing it manually.
    """

    def __init__(self, student_model, max_grad_norm=1.0):
        self.student_model = student_model
        self.max_grad_norm = max_grad_norm

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        import torch
        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.max_grad_norm)


class ReinitProjectorsCallback(TrainerCallback):
    """Re-initialize auto-created projectors with normal_(0, std) after lazy creation.

    Auto-projectors are created during the first forward pass (eval_on_start or
    first training step). This callback fires once on the first training step to
    re-initialize them with BERT-style normal_(0, std) instead of kaiming_uniform_.
    """

    def __init__(self, alignments, std=0.02):
        self.alignments = alignments
        self.std = std
        self._done = False

    def on_step_begin(self, args, state, control, **kwargs):
        if self._done:
            return
        self._done = True
        import torch.nn as nn
        count = 0
        for a in self.alignments:
            for proj in [a.input_projector, a.output_projector]:
                if proj is None:
                    continue
                for m in proj.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0.0, std=self.std)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                        count += 1
        if count > 0:
            logger.info("ReinitProjectors: re-initialized %d linear layers with normal_(0, %.4f)", count, self.std)


class PerEpochOptimizerResetCallback(TrainerCallback):
    """Reset optimizer state at each epoch boundary (à la TextBrewer).

    TextBrewer creates a fresh ``AdamW`` every epoch, discarding momentum
    and variance accumulators.  This callback reproduces that behaviour
    within the HF Trainer loop by zeroing the optimizer state dict.

    Pair with ``lr_scheduler_type=cosine_with_restarts`` (num_cycles =
    num_epochs) so the LR also resets at each epoch boundary.
    """

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        if epoch == 0:
            return  # nothing to reset on the first epoch
        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            return
        for group in optimizer.param_groups:
            for p in group["params"]:
                optimizer.state[p] = {}
        logger.info("PerEpochOptimizerReset: cleared optimizer state at epoch %d", epoch)


class TextBrewerScheduleCallback(TrainerCallback):
    """Per-epoch linear warmup+decay schedule matching TextBrewer exactly.

    TextBrewer creates a fresh ``get_linear_schedule_with_warmup`` each epoch
    with decreasing warmup steps::

        warmup_this_epoch = max(0, total_warmup - epoch * steps_per_epoch)
        num_training_steps = steps_per_epoch

    This produces a sawtooth LR: each epoch ramps from 0 to some fraction
    of lr_max (capped by remaining warmup) then linearly decays to 0.

    Also resets optimizer state each epoch (combines PerEpochOptimizerReset).

    Sets LR directly on optimizer param_groups at each step to avoid
    replacing the Trainer's scheduler (which causes pickling errors).
    """

    def __init__(self, base_lr, steps_per_epoch, total_warmup_steps, num_epochs):
        self.base_lr = base_lr
        self.steps_per_epoch = steps_per_epoch
        self.total_warmup_steps = total_warmup_steps
        self.num_epochs = num_epochs
        self._epoch_start_step = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            return

        self._epoch_start_step = state.global_step

        # Reset optimizer state (fresh AdamW per epoch)
        if epoch > 0:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    optimizer.state[p] = {}

        logger.info(
            "TextBrewerSchedule: epoch %d — warmup=%d/%d steps, base_lr=%.2e",
            epoch, max(0, self.total_warmup_steps - epoch * self.steps_per_epoch),
            self.steps_per_epoch, self.base_lr,
        )

    def on_step_begin(self, args, state, control, **kwargs):
        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            return

        epoch = int(state.epoch)
        step_in_epoch = state.global_step - self._epoch_start_step
        warmup_this_epoch = max(0, self.total_warmup_steps - epoch * self.steps_per_epoch)

        if warmup_this_epoch > 0 and step_in_epoch < warmup_this_epoch:
            # Warmup phase: linear ramp from 0 to lr_max (or fraction)
            lr = self.base_lr * (step_in_epoch / warmup_this_epoch)
        else:
            # Decay phase: linear decay from lr_max to 0
            decay_start = max(0, warmup_this_epoch)
            remaining = self.steps_per_epoch - decay_start
            if remaining > 0:
                progress = (step_in_epoch - decay_start) / remaining
                lr = self.base_lr * max(0.0, 1.0 - progress)
            else:
                lr = self.base_lr

        for group in optimizer.param_groups:
            group["lr"] = lr


def build_memory_leak_callback_from_env():
    """Construct a ``MemoryLeakDetectionCallback`` from environment variables.

    Env vars:
    - ``MEMORY_PROFILE=1``: enable the callback (otherwise returns ``None``)
    - ``TENSOR_DIFF_AT_BATCH=N,M``: diff live tensors between eval batch N and M
    - ``MEMORY_SNAPSHOT_AT_BATCH=N``: dump a ``torch.cuda.memory`` history
      pickle at eval batch N (path: ``/tmp/memsnap_batch_<N>.pickle``)

    Returns:
        A ``MemoryLeakDetectionCallback`` instance, or ``None`` if
        ``MEMORY_PROFILE`` is not set to ``1``.
    """
    if os.environ.get("MEMORY_PROFILE") != "1":
        return None
    from silverspoon_kd.engines import MemoryLeakDetectionCallback

    diff = None
    diff_env = os.environ.get("TENSOR_DIFF_AT_BATCH")
    if diff_env:
        try:
            parts = [int(x) for x in diff_env.split(",")]
            if len(parts) == 2 and parts[0] < parts[1]:
                diff = (parts[0], parts[1])
        except ValueError:
            logger.warning(
                "TENSOR_DIFF_AT_BATCH=%r is not a valid 'N,M' pair", diff_env
            )

    snapshot_at = None
    snapshot_path = None
    snap_env = os.environ.get("MEMORY_SNAPSHOT_AT_BATCH")
    if snap_env:
        try:
            snapshot_at = int(snap_env)
            snapshot_path = f"/tmp/memsnap_batch_{snapshot_at}.pickle"
        except ValueError:
            logger.warning(
                "MEMORY_SNAPSHOT_AT_BATCH=%r is not a valid int", snap_env
            )

    return MemoryLeakDetectionCallback(
        diff_between_batches=diff,
        snapshot_at_batch=snapshot_at,
        snapshot_path=snapshot_path,
        log_every_n_batches=5,
        log_first_n_batches=35,
    )


# ── Loss function ────────────────────────────────────────────────────────────

def get_loss_function(cfg: DictConfig, teacher_model=None):
    from silverspoon_kd import get_loss_function as _get_loss_function
    loss_type = cfg.loss.type
    loss_kwargs = {k: v for k, v in cfg.loss.items() if k != "type"}
    # Mahalanobis losses need the teacher's output projection weights
    if loss_type.startswith("mahal") and teacher_model is not None:
        lm_head = getattr(teacher_model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            loss_kwargs["weight_matrix"] = lm_head.weight
            logger.info("Mahalanobis: using teacher lm_head weights (%s)", lm_head.weight.shape)
    log_kwargs = {k: v for k, v in loss_kwargs.items() if k != "weight_matrix"}
    logger.info("Loss: %s%s", loss_type, f" ({log_kwargs})" if log_kwargs else "")
    return _get_loss_function(loss_type, **loss_kwargs)


# ── Alignments ───────────────────────────────────────────────────────────────

def create_alignments(cfg, teacher_model, student_model, device, dtype=None):
    from silverspoon_kd import create_alignments as _lib_create_alignments

    module_pattern = cfg.distiller.alignment.get("module_pattern")
    if module_pattern is None and cfg.teacher.get("layer_pattern"):
        module_pattern = cfg.teacher.layer_pattern
    if module_pattern is None:
        raise ValueError("No module_pattern in distiller.alignment or teacher.layer_pattern")

    loss_function = get_loss_function(cfg, teacher_model=teacher_model)
    adapter_index = None
    if cfg.distiller.alignment.adapter.type == "generic":
        adapter_index = cfg.distiller.alignment.adapter.get("index")

    import re
    student_layer_pattern = cfg.student.get("layer_pattern", None)
    teacher_layers = cfg.student.get("teacher_layers", None)

    # Build module specification depending on architecture match and layer mapping
    if teacher_layers is not None:
        teacher_layers = list(teacher_layers)
        # Use the appropriate pattern for each model
        student_pattern = student_layer_pattern if student_layer_pattern else module_pattern
        teacher_names = sorted(
            [n for n, _ in teacher_model.named_modules() if re.search(module_pattern, n)],
            key=lambda n: int(re.search(module_pattern, n).group(1))
        )
        student_names = sorted(
            [n for n, _ in student_model.named_modules() if re.search(student_pattern, n)],
            key=lambda n: int(re.search(student_pattern, n).group(1))
        )
        if len(teacher_layers) != len(student_names):
            raise ValueError(
                f"teacher_layers has {len(teacher_layers)} entries but student has "
                f"{len(student_names)} matching modules"
            )
        modules = {
            re.escape(teacher_names[i]) + "$": student_names[j]
            for j, i in enumerate(teacher_layers)
        }
        logger.info("teacher_layers mapping: %s", modules)
    elif student_layer_pattern is not None and student_layer_pattern != module_pattern:
        # Cross-architecture with matching layer counts (no explicit teacher_layers)
        modules = {module_pattern: student_layer_pattern}
        logger.info("Cross-architecture alignment: %s", modules)
    else:
        modules = module_pattern

    projector_init = cfg.distiller.alignment.get("projector_init", None)
    auto_projector = cfg.distiller.alignment.get("auto_projector", True)
    alignments = _lib_create_alignments(
        teacher_model=teacher_model, student_model=student_model,
        modules=modules, loss_function=loss_function, output_selector_index=adapter_index,
        auto_projector=auto_projector, projector_init=projector_init)

    # Enable auto device/dtype matching for cross-GPU placement
    for a in alignments:
        a.auto_device_match = True
        a.auto_dtype_match = True

    # Apply per-alignment loss_weight from config (e.g. RelKD lambda values)
    alignment_loss_weight = cfg.distiller.alignment.get("loss_weight", None)
    if alignment_loss_weight is not None:
        for a in alignments:
            a.loss_weight = float(alignment_loss_weight)

    return alignments


# ── Training arguments ───────────────────────────────────────────────────────

def _resolve_teacher_placement(cfg):
    """Convert Hydra teacher_placement config to silverspoon-kd format.

    When teacher_placement is set in config, train.py handles placement
    manually via place_teacher_pp() before distiller init. We always return
    "replicated" so the distiller's _setup_teacher_placement() sees the
    teacher already on CUDA and skips re-placement.
    """
    return "replicated"


def get_training_args(cfg, output_dir, run_name, max_steps_override=None):
    from silverspoon_kd import TrainingArguments

    distiller_type = cfg.distiller.type
    if max_steps_override is not None and cfg.training.max_steps <= 0:
        max_steps = max_steps_override
    else:
        max_steps = cfg.training.max_steps

    extra_kwargs = {}
    if distiller_type == "reskd":
        extra_kwargs = {
            "alpha": cfg.distiller.get("alpha", 0.0),
            "magnitude_aware_weighting": cfg.distiller.get("magnitude_aware_weighting", False),
        }
    elif distiller_type == "hkd":
        extra_kwargs = {
            "alpha": cfg.distiller.get("alpha", 0.0),
            "magnitude_aware_weighting": cfg.distiller.get("magnitude_aware_weighting", False),
        }

    return TrainingArguments(
        output_dir=output_dir, logging_dir=output_dir, run_name=run_name,
        logging_strategy=cfg.training.logging_strategy, logging_steps=cfg.training.logging_steps,
        report_to=cfg.training.report_to,
        eval_strategy=cfg.training.eval_strategy, eval_steps=cfg.training.eval_steps,
        eval_on_start=cfg.training.eval_on_start,
        save_strategy=cfg.training.save_strategy, save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        num_train_epochs=cfg.training.num_epochs, max_steps=max_steps,
        per_device_train_batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        lr_scheduler_kwargs=OmegaConf.to_container(cfg.training.lr_scheduler_kwargs, resolve=True) if cfg.training.get("lr_scheduler_kwargs") else None,
        warmup_ratio=cfg.training.warmup_ratio,
        warmup_steps=cfg.training.get("warmup_steps", 0),
        bf16=cfg.training.bf16, fp16=cfg.training.fp16,
        gradient_checkpointing=cfg.training.get("gradient_checkpointing", False),
        optim=cfg.training.get("optim", "adamw_torch_fused"),
        optim_args=cfg.training.get("optim_args", None),
        weight_decay=cfg.training.get("weight_decay", 0.0),
        max_grad_norm=cfg.training.get("max_grad_norm", 1.0),
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        dataloader_pin_memory=cfg.training.dataloader_pin_memory,
        dataloader_prefetch_factor=cfg.training.dataloader_prefetch_factor if cfg.training.dataloader_num_workers > 0 else None,
        use_weightwatcher=cfg.training.get("use_weightwatcher", get_model_type(cfg) != "image_classification"),
        enable_profiling=cfg.training.get("enable_profiling", False),
        profiling_output_dir=cfg.training.get("profiling_output_dir", None),
        profiling_wait=cfg.training.get("profiling_wait", 20),
        profiling_warmup=cfg.training.get("profiling_warmup", 3),
        profiling_active=cfg.training.get("profiling_active", 3),
        profiling_repeat=cfg.training.get("profiling_repeat", 1),
        profiling_with_stack=cfg.training.get("profiling_with_stack", False),
        load_best_model_at_end=True,
        metric_for_best_model=cfg.training.get("metric_for_best_model", "eval_loss"),
        greater_is_better=cfg.training.get("greater_is_better", False),
        e2e_eval_loss=cfg.training.get("e2e_eval_loss", "forward"),
        clip_projectors=cfg.training.get("clip_projectors", True),
        teacher_placement=_resolve_teacher_placement(cfg),
        **extra_kwargs,
    )


# ── Distillation training ───────────────────────────────────────────────────

_DISTILLER_CLASSES = {
    "bkd": "BlockwiseDistiller",
    "bkd_attn": "BlockwiseDistiller",
    "hkd": "HolisticDistiller",
    "reskd": "ResponseBasedDistiller",
}


def train_distillation(
    cfg, teacher_model, student_model, alignments,
    train_dataset, eval_dataset, tokenizer_or_processor,
    training_args, teacher_device=None, student_device=None,
    compute_metrics=None,
):
    """Train using silverspoon-kd distillers."""
    from silverspoon_kd import (BlockwiseDistiller, HolisticDistiller,
                                ResponseBasedDistiller)

    distiller_type = cfg.distiller.type
    model_type = get_model_type(cfg)

    # Data collator
    if model_type == "causal_lm":
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer_or_processor, mlm=False)
    elif model_type == "mlm":
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer_or_processor, mlm=True,
            mlm_probability=cfg.distiller.get("mlm_probability", 0.15))
    elif model_type == "image_classification":
        collator = DefaultDataCollator()
    elif model_type in ("sequence_classification", "question_answering"):
        from transformers import DataCollatorWithPadding
        collator = DataCollatorWithPadding(tokenizer_or_processor)
    elif model_type == "token_classification":
        from transformers import DataCollatorForTokenClassification
        collator = DataCollatorForTokenClassification(tokenizer_or_processor)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Teacher input preparation
    if model_type == "image_classification":
        def prepare_teacher_inputs(inputs):
            return {"pixel_values": inputs["pixel_values"]}
    elif model_type in ("sequence_classification", "question_answering", "token_classification"):
        def prepare_teacher_inputs(inputs):
            result = {"input_ids": inputs["input_ids"]}
            if "attention_mask" in inputs:
                result["attention_mask"] = inputs["attention_mask"]
            if "token_type_ids" in inputs:
                result["token_type_ids"] = inputs["token_type_ids"]
            return result
    else:
        _accepts_cache = "use_cache" in inspect.signature(teacher_model.forward).parameters
        def prepare_teacher_inputs(inputs):
            result = {"input_ids": inputs["input_ids"]}
            if "attention_mask" in inputs:
                result["attention_mask"] = inputs["attention_mask"]
            if _accepts_cache:
                result["use_cache"] = False
            return result

    multi_device = (teacher_device is not None and student_device is not None
                    and teacher_device != student_device)
    if multi_device:
        training_args._n_gpu = 1
        _sd = student_device
        type(training_args).device = property(lambda self, _d=_sd: _d)
        # Wrap prepare_teacher_inputs to move tensors to teacher device
        _orig_prep = prepare_teacher_inputs
        _td = teacher_device
        def prepare_teacher_inputs(inputs, _fn=_orig_prep, _dev=_td):
            return {k: v.to(_dev) if isinstance(v, torch.Tensor) else v
                    for k, v in _fn(inputs).items()}

    base_kwargs = {
        "teacher_model": teacher_model, "alignments": alignments,
        "train_dataset": train_dataset, "eval_dataset": eval_dataset,
        "data_collator": collator, "args": training_args,
        "prepare_teacher_inputs": prepare_teacher_inputs,
        "compute_metrics": compute_metrics,
    }

    # Distiller dispatch
    if distiller_type in ("bkd", "bkd_attn"):
        trainer = BlockwiseDistiller(
            **base_kwargs, student_models={student_model.name_or_path: student_model},
            auto_truncate=True)

    elif distiller_type == "hkd":
        hkd_kwargs = {k: v for k, v in base_kwargs.items()}
        # Add response-based alignment on the final output layer (LM head)
        # so HKD aligns both intermediate blocks AND the final predictions.
        # Even when the LM head is copied from the teacher, the KL loss on
        # output distributions provides gradient signal to earlier layers.
        response_weight = cfg.distiller.get("response_loss_weight",
                                                  len(alignments) / 3)
        if response_weight > 0 and model_type == "causal_lm":
            from silverspoon_kd.alignments.alignment import Alignment
            temperature = cfg.distiller.get("temperature", 1.0)
            chunk_size = cfg.distiller.get("chunk_size", 1024)

            # Try Liger fused linear KL: capture inputs to lm_head (i.e. the
            # final norm output) and apply FusedLinearKLDivLoss with the
            # lm_head weights via closure. This avoids materializing the
            # full vocab logit tensors at all, critical for large-vocab
            # models like Qwen3 (151K vocab).
            from silverspoon_kd.losses.liger import (
                LIGER_KERNEL_AVAILABLE, FusedLinearKLDivLoss,
            )
            use_liger_response = (
                LIGER_KERNEL_AVAILABLE
                and hasattr(student_model, "lm_head")
                and hasattr(teacher_model, "lm_head")
                and hasattr(student_model, "model")
                and hasattr(student_model.model, "norm")
                and hasattr(teacher_model, "model")
                and hasattr(teacher_model.model, "norm")
            )
            if use_liger_response:
                _student_lm_head = student_model.lm_head
                _teacher_lm_head = teacher_model.lm_head
                _fused_kl = FusedLinearKLDivLoss(
                    temperature=temperature, chunk_size=chunk_size,
                )
                def liger_response_loss(student_hidden, teacher_hidden):
                    # student_hidden / teacher_hidden are last-norm outputs
                    # (input to lm_head): shape (batch, seq, hidden).
                    # FusedLinearKLDivLoss expects 2D inputs (N, hidden),
                    # so flatten the batch and seq dims.
                    s_flat = student_hidden.reshape(-1, student_hidden.shape[-1])
                    t_flat = teacher_hidden.reshape(-1, teacher_hidden.shape[-1])
                    sw = _student_lm_head.weight
                    tw = _teacher_lm_head.weight
                    sb = getattr(_student_lm_head, "bias", None)
                    tb = getattr(_teacher_lm_head, "bias", None)
                    return _fused_kl(
                        s_flat, sw, t_flat, tw,
                        target=None, output_head_bias=sb,
                        teacher_output_head_bias=tb,
                    )
                lm_head_alignment = Alignment(
                    teacher_block=teacher_model.model.norm,
                    student_block=student_model.model.norm,
                    teacher_model_name=teacher_model.name_or_path,
                    student_model_name=getattr(student_model, "name_or_path", "student"),
                    teacher_module_name="model.norm",
                    student_module_name="model.norm",
                    loss_function=liger_response_loss,
                    auto_projector=False,
                )
                logger.info("HKD: added Liger fused linear KL response alignment "
                            "on model.norm (weight=%.2f, T=%.1f)",
                            response_weight, temperature)
            else:
                # Fallback: chunked KL on lm_head outputs (still materializes
                # full vocab logits but in chunks).
                from silverspoon_kd.losses.kl import kl_divergence_loss
                kl_loss = kl_divergence_loss(
                    temperature=temperature, chunk_size=chunk_size,
                )
                lm_head_alignment = Alignment(
                    teacher_block=teacher_model.lm_head,
                    student_block=student_model.lm_head,
                    teacher_model_name=teacher_model.name_or_path,
                    student_model_name=getattr(student_model, "name_or_path", "student"),
                    teacher_module_name="lm_head",
                    student_module_name="lm_head",
                    loss_function=kl_loss,
                    auto_projector=False,
                )
                logger.info("HKD: added chunked-KL response alignment on lm_head "
                            "(weight=%.2f, T=%.1f)", response_weight, temperature)
            lm_head_alignment.loss_weight = response_weight
            hkd_kwargs["alignments"] = list(alignments) + [lm_head_alignment]
        elif response_weight > 0 and model_type in ("sequence_classification", "image_classification"):
            # Small-output-dim models (e.g. BERT classifier, ViT): simple KL
            # on the classification head.  No chunking or Liger needed since
            # the output dimension is tiny (num_labels, e.g. 3 for MNLI).
            from silverspoon_kd.alignments.alignment import Alignment
            from silverspoon_kd.losses.kl import kl_divergence_loss
            temperature = cfg.distiller.get("temperature", 1.0)
            kl_loss = kl_divergence_loss(temperature=temperature)
            # Locate the classification head on teacher and student independently.
            # They may differ (e.g. ResNet50 uses timm_model.fc, VGG11-BN uses
            # timm_model.head.fc).
            _HEAD_PATHS = ("classifier", "score", "heads",
                           "timm_model.head.fc", "timm_model.fc")
            def _find_head(model, paths):
                for attr in paths:
                    obj = model
                    try:
                        for part in attr.split("."):
                            obj = getattr(obj, part)
                        return obj, attr
                    except AttributeError:
                        continue
                return None, None
            teacher_head, t_head_name = _find_head(teacher_model, _HEAD_PATHS)
            student_head, s_head_name = _find_head(student_model, _HEAD_PATHS)
            if teacher_head is not None and student_head is not None:
                head_alignment = Alignment(
                    teacher_block=teacher_head,
                    student_block=student_head,
                    teacher_model_name=teacher_model.name_or_path,
                    student_model_name=getattr(student_model, "name_or_path", "student"),
                    teacher_module_name=t_head_name,
                    student_module_name=s_head_name,
                    loss_function=kl_loss,
                    auto_projector=False,
                )
                head_alignment.loss_weight = response_weight
                hkd_kwargs["alignments"] = list(alignments) + [head_alignment]
                logger.info("HKD: added KL response alignment on teacher=%s / student=%s "
                            "(weight=%.2f, T=%.1f)", t_head_name, s_head_name,
                            response_weight, temperature)
            else:
                logger.warning("HKD: could not locate classification head on "
                               "teacher/student — no KL response alignment will "
                               "be applied. The student classifier will receive "
                               "no direct supervision.")
        # Strip labels from student inputs so the student forward doesn't
        # compute the full-vocab cross-entropy loss (which OOMs on large-vocab
        # models like Qwen3 151K). HKD only needs hidden states and logits
        # for alignment + KL losses, not the hard label loss.
        # ``use_cache=False`` prevents KV cache allocation (~4 GB for large
        # models) but is only valid for causal-LM transformers — vision
        # models like VGG don't accept that kwarg.
        def prepare_student_inputs_no_labels(inputs):
            out = {k: v for k, v in inputs.items() if k != "labels"}
            if model_type == "causal_lm":
                out["use_cache"] = False
            return out
        hkd_kwargs["prepare_student_inputs"] = prepare_student_inputs_no_labels
        trainer = HolisticDistiller(student_model, **hkd_kwargs)

        # Apply weight decay to ALL parameters (including bias/LayerNorm)
        # when decay_all_params is set. By default, HF Trainer excludes
        # bias and LayerNorm from weight decay. Some frameworks (e.g.
        # TextBrewer) apply decay uniformly to all parameters.
        if cfg.training.get("decay_all_params", False):
            def _decay_all(self_trainer, model):
                return [n for n, _ in model.named_parameters()]
            trainer.get_decay_parameter_names = lambda model: _decay_all(trainer, model)
            logger.info("decay_all_params: weight decay will apply to ALL parameters")

        memleak_cb = build_memory_leak_callback_from_env()
        if memleak_cb is not None:
            trainer.add_callback(memleak_cb)
            logger.info(
                "HKD: enabled MemoryLeakDetectionCallback "
                "(diff=%s, snapshot=%s)",
                memleak_cb._diff_between, memleak_cb._snapshot_at,
            )

    elif distiller_type == "reskd":
        if multi_device:
            training_args.auto_device_match = True
        training_args.auto_dtype_match = True
        from silverspoon_kd.losses.kl import kl_divergence_loss
        soft_loss_fn = kl_divergence_loss(
            temperature=cfg.distiller.get("temperature", 4.0),
            chunk_size=cfg.distiller.get("chunk_size", 1024),
        )
        reskd_kwargs = {}
        if cfg.distiller.get("use_liger_kernel", False):
            reskd_kwargs["use_liger_kernel"] = True
            reskd_kwargs["output_head_layer"] = cfg.distiller.get("output_head_layer", "lm_head")
        trainer = ResponseBasedDistiller(
            student_model=student_model, teacher_model=teacher_model,
            soft_loss_fn=soft_loss_fn,
            train_dataset=train_dataset, eval_dataset=eval_dataset,
            data_collator=collator, args=training_args,
            prepare_teacher_inputs=prepare_teacher_inputs,
            compute_metrics=compute_metrics, **reskd_kwargs)
    else:
        raise ValueError(f"Unknown distiller: {distiller_type}")

    if multi_device and distiller_type == "reskd":
        trainer.accelerator.device_placement = False

    # Early stopping
    if cfg.training.get("early_stopping_patience", 0) > 0:
        from transformers import EarlyStoppingCallback
        trainer.add_callback(EarlyStoppingCallback(
            early_stopping_patience=cfg.training.early_stopping_patience,
            early_stopping_threshold=cfg.training.get("early_stopping_threshold", 0.0)))

    # Per-epoch optimizer reset (TextBrewer-style)
    if cfg.training.get("textbrewer_schedule", False):
        # Full TextBrewer schedule: per-epoch linear warmup+decay + optimizer reset
        steps_per_epoch = len(train_dataset) // cfg.training.batch_size
        total_warmup = int(steps_per_epoch * cfg.training.num_epochs * cfg.training.warmup_ratio)
        trainer.add_callback(TextBrewerScheduleCallback(
            base_lr=cfg.training.learning_rate,
            steps_per_epoch=steps_per_epoch,
            total_warmup_steps=total_warmup,
            num_epochs=cfg.training.num_epochs,
        ))
        logger.info("TextBrewerSchedule: %d steps/epoch, %d total warmup", steps_per_epoch, total_warmup)
    elif cfg.training.get("reset_optimizer_each_epoch", False):
        trainer.add_callback(PerEpochOptimizerResetCallback())

    # Projector init override (e.g. normal_(0, 0.02) to match TextBrewer)
    proj_init_std = cfg.distiller.alignment.get("projector_init_std", None) if alignments else None
    if proj_init_std is not None:
        trainer.add_callback(ReinitProjectorsCallback(alignments, std=float(proj_init_std)))

    # Exclude projectors from gradient clipping (TextBrewer only clips student params)
    if cfg.distiller.get("clip_student_only", False) and alignments:
        original_max_grad_norm = training_args.max_grad_norm
        training_args.max_grad_norm = 0  # disable Trainer's clipping
        trainer.add_callback(StudentOnlyGradClipCallback(
            student_model, max_grad_norm=original_max_grad_norm))

    # Multi-step LR decay
    lr_milestones = cfg.training.get("lr_milestones", None)
    if lr_milestones is not None:
        trainer.add_callback(MultiStepLRCallback(
            milestones=list(lr_milestones),
            gamma=cfg.training.get("lr_milestone_gamma", 0.1)))

    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)
    student_model.save_pretrained(os.path.join(training_args.output_dir, "student_model"), safe_serialization=True)
    logger.info("Saved student model to %s/student_model", training_args.output_dir)
    return trainer


# ── LoRA training ────────────────────────────────────────────────────────────

def train_lora(cfg, model, train_dataset, eval_dataset, tokenizer, output_dir, run_name):
    """Train using LoRA (Parameter-Efficient Fine-Tuning)."""
    from peft import LoraConfig, TaskType, get_peft_model
    from silverspoon_kd import SilentProgressCallback
    from transformers import Trainer, TrainingArguments
    from transformers.trainer_callback import ProgressCallback

    lora_cfg = cfg.distiller.lora
    task_types = {"CAUSAL_LM": TaskType.CAUSAL_LM, "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
                  "TOKEN_CLS": TaskType.TOKEN_CLS, "SEQ_CLS": TaskType.SEQ_CLS}
    # Detect LoLCATs feature map modules BEFORE wrapping with PEFT so we can
    # add them to modules_to_save (persisted alongside the LoRA adapter).
    feature_map_modules = []
    for name, _ in model.named_modules():
        if "feature_map" in name:
            # PEFT wants the leaf module name relative to the base model,
            # e.g. "transformer.h.0.attn.feature_map_q"
            feature_map_modules.append(name)
    modules_to_save = list(lora_cfg.get("modules_to_save", []))
    if feature_map_modules:
        # Deduplicate: PEFT matches by suffix, so just use the unique
        # short names (e.g. "feature_map_q", "feature_map_k")
        short_names = sorted({n.split(".")[-1] for n in feature_map_modules
                              if "feature_map" in n.split(".")[-1]})
        modules_to_save.extend(short_names)
        logger.info("LoLCATs: adding %s to modules_to_save for PEFT", short_names)

    lora_config = LoraConfig(
        r=lora_cfg.r, lora_alpha=lora_cfg.lora_alpha,
        target_modules=list(lora_cfg.target_modules),
        lora_dropout=lora_cfg.lora_dropout, bias=lora_cfg.bias,
        task_type=task_types.get(lora_cfg.task_type, TaskType.CAUSAL_LM),
        modules_to_save=modules_to_save if modules_to_save else None)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_workers = cfg.training.dataloader_num_workers
    early_stopping = cfg.training.get("early_stopping_patience", 0) > 0
    args = TrainingArguments(
        output_dir=output_dir, logging_dir=output_dir, run_name=run_name,
        logging_strategy=cfg.training.logging_strategy, logging_steps=cfg.training.logging_steps,
        report_to=cfg.training.report_to, log_level=cfg.training.log_level,
        disable_tqdm=cfg.training.disable_tqdm,
        eval_strategy=cfg.training.eval_strategy, eval_steps=cfg.training.eval_steps,
        eval_on_start=cfg.training.eval_on_start,
        save_strategy=cfg.training.save_strategy, save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        num_train_epochs=cfg.training.num_epochs, max_steps=cfg.training.max_steps,
        per_device_train_batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate, lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio, bf16=cfg.training.bf16, fp16=cfg.training.fp16,
        dataloader_num_workers=num_workers, dataloader_pin_memory=cfg.training.dataloader_pin_memory,
        dataloader_prefetch_factor=cfg.training.dataloader_prefetch_factor if num_workers > 1 else None,
        torch_compile=getattr(cfg.training, "torch_compile", True),
        optim=cfg.training.get("optim", "adamw_torch_fused"),
        gradient_checkpointing=cfg.training.get("gradient_checkpointing", False),
        load_best_model_at_end=True,
        metric_for_best_model=cfg.training.get("metric_for_best_model", "eval_loss"),
        greater_is_better=cfg.training.get("greater_is_better", False))

    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
                      data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))

    if early_stopping:
        from transformers import EarlyStoppingCallback
        trainer.add_callback(EarlyStoppingCallback(
            early_stopping_patience=cfg.training.early_stopping_patience,
            early_stopping_threshold=cfg.training.get("early_stopping_threshold", 0.0)))
    if getattr(cfg.training, "disable_console_logs", False):
        trainer.remove_callback(ProgressCallback)
        trainer.add_callback(SilentProgressCallback)

    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)
    model.save_pretrained(os.path.join(output_dir, "lora_model"))
    logger.info("Saved LoRA model to %s/lora_model", output_dir)

    # When modules_to_save are present (e.g. LoLCATs feature maps), also save
    # a fully merged model so evaluation doesn't depend on PEFT loading the
    # modules_to_save correctly (lm_eval doesn't handle this reliably).
    if lora_config.modules_to_save:
        merged = model.merge_and_unload()
        merged_dir = os.path.join(output_dir, "model")
        merged.save_pretrained(merged_dir)
        logger.info("Saved merged model (LoRA + modules_to_save) to %s", merged_dir)

    return trainer


# ── Standard training ────────────────────────────────────────────────────────

def train_standard(cfg, model, train_dataset, eval_dataset, tokenizer, output_dir, run_name,
                   num_training_steps=None, compute_metrics=None):
    """Full fine-tuning of all model weights (no LoRA, no distillation)."""
    from silverspoon_kd import SilentProgressCallback
    from transformers import Trainer, TrainingArguments
    from transformers.trainer_callback import ProgressCallback

    model_type = get_model_type(cfg)
    is_vision = not hasattr(tokenizer, "pad_token")
    if not is_vision and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_workers = cfg.training.dataloader_num_workers
    early_stopping = cfg.training.get("early_stopping_patience", 0) > 0
    max_steps = cfg.training.max_steps if cfg.training.max_steps > 0 else (num_training_steps if num_training_steps else -1)

    args = TrainingArguments(
        output_dir=output_dir, logging_dir=output_dir, run_name=run_name,
        logging_strategy=cfg.training.logging_strategy, logging_steps=cfg.training.logging_steps,
        report_to=cfg.training.report_to, log_level=cfg.training.log_level,
        disable_tqdm=cfg.training.disable_tqdm,
        eval_strategy=cfg.training.eval_strategy, eval_steps=cfg.training.eval_steps,
        eval_on_start=cfg.training.eval_on_start,
        save_strategy=cfg.training.save_strategy, save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        num_train_epochs=cfg.training.num_epochs, max_steps=max_steps,
        per_device_train_batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate, lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio, bf16=cfg.training.bf16, fp16=cfg.training.fp16,
        dataloader_num_workers=num_workers, dataloader_pin_memory=cfg.training.dataloader_pin_memory,
        dataloader_prefetch_factor=cfg.training.dataloader_prefetch_factor if num_workers > 1 else None,
        torch_compile=getattr(cfg.training, "torch_compile", True),
        optim=cfg.training.get("optim", "adamw_torch_fused"),
        optim_args=cfg.training.get("optim_args", None),
        weight_decay=cfg.training.get("weight_decay", 0.0),
        max_grad_norm=cfg.training.get("max_grad_norm", 1.0),
        gradient_checkpointing=cfg.training.get("gradient_checkpointing", False),
        load_best_model_at_end=True,
        metric_for_best_model=cfg.training.get("metric_for_best_model", "eval_loss"),
        greater_is_better=cfg.training.get("greater_is_better", False))

    if is_vision:
        collator = DefaultDataCollator()
    elif model_type == "mlm":
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True,
            mlm_probability=cfg.distiller.get("mlm_probability", 0.15))
    elif model_type in ("sequence_classification", "question_answering"):
        from transformers import DataCollatorWithPadding
        collator = DataCollatorWithPadding(tokenizer)
    elif model_type == "token_classification":
        from transformers import DataCollatorForTokenClassification
        collator = DataCollatorForTokenClassification(tokenizer)
    else:
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # Use SafeTrainer to clone loss before inplace ops (needed for fla's L2Wrap)
    class SafeTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            if return_outputs:
                loss, outputs = result
                return loss.clone(), outputs
            return result.clone()

    trainer = SafeTrainer(model=model, args=args, train_dataset=train_dataset,
                          eval_dataset=eval_dataset, data_collator=collator,
                          compute_metrics=compute_metrics)

    if early_stopping:
        from transformers import EarlyStoppingCallback
        trainer.add_callback(EarlyStoppingCallback(
            early_stopping_patience=cfg.training.early_stopping_patience,
            early_stopping_threshold=cfg.training.get("early_stopping_threshold", 0.0)))
    if getattr(cfg.training, "disable_console_logs", False):
        trainer.remove_callback(ProgressCallback)
        trainer.add_callback(SilentProgressCallback)

    # Multi-step LR decay
    lr_milestones = cfg.training.get("lr_milestones", None)
    if lr_milestones is not None:
        trainer.add_callback(MultiStepLRCallback(
            milestones=list(lr_milestones),
            gamma=cfg.training.get("lr_milestone_gamma", 0.1)))

    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)
    model.save_pretrained(os.path.join(output_dir, "model"), safe_serialization=True)
    tokenizer.save_pretrained(os.path.join(output_dir, "model"))
    logger.info("Saved model to %s/model", output_dir)
    return trainer
