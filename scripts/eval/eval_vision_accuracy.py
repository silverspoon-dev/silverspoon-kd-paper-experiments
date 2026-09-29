#!/usr/bin/env python3
"""Evaluate CIFAR accuracy for vision student_model directories.

Loads each student_model from its saved directory and computes accuracy
on the CIFAR test set.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_vision_accuracy.py
    CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_vision_accuracy.py --pattern '*reskd*cifar100*'
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def _infer_student_base(run_name):
    """Infer the student's timm base model from the run name."""
    parts = run_name.split("__")
    student_part = parts[3] if len(parts) >= 4 and parts[0] == "silverspoon-kd" else (parts[2] if len(parts) >= 3 else run_name)
    if "vgg11bn" in student_part:
        return "timm/vgg11_bn.tv_in1k"
    if "vgg16" in student_part:
        return "timm/vgg16.tv_in1k"
    if "resnet" in student_part:
        return "timm/resnet50.a1_in1k"
    # Fallback on full name
    for tag, model in [("vgg11bn", "timm/vgg11_bn.tv_in1k"), ("vgg16", "timm/vgg16.tv_in1k"), ("resnet50", "timm/resnet50.a1_in1k")]:
        if tag in run_name:
            return model
    raise ValueError(f"Cannot infer base model from: {run_name}")


def load_vision_model(model_dir, run_name):
    """Load a vision student_model by reconstructing the CIFAR-adapted architecture.

    The saved config has stale ``num_features`` and missing structural changes
    (pre_logits replaced with Identity), so ``from_pretrained`` alone cannot
    reconstruct the correct model. Instead we:
    1. Load the base timm model and apply CIFAR adaptations (matching training)
    2. Load saved weights, adding the ``timm_model.`` prefix if needed
    3. Use ``nn.Module.load_state_dict`` directly to bypass the timm wrapper's
       strict shape checking (the wrapper raises even with strict=False)
    """
    from safetensors.torch import load_file
    from transformers import AutoModelForImageClassification
    from models import adapt_vgg16_for_cifar, adapt_resnet_for_cifar, apply_depthwise_separable_replacement
    import torch.nn as nn

    model_dir = Path(model_dir)
    num_classes = 100 if "cifar100" in run_name else 10
    is_depthwise = "depthwise_separable" in run_name

    parts = run_name.split("__")
    student_part = parts[3] if len(parts) >= 4 and parts[0] == "silverspoon-kd" else (parts[2] if len(parts) >= 3 else run_name)
    is_resnet = "resnet" in student_part and "vgg" not in student_part

    # 1. Load base timm model and apply CIFAR adaptations (same as training)
    base_model = _infer_student_base(run_name)
    model = AutoModelForImageClassification.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    if is_resnet:
        model = adapt_resnet_for_cifar(model, num_classes=num_classes, dtype=torch.bfloat16)
    else:
        model = adapt_vgg16_for_cifar(model, num_classes=num_classes, dtype=torch.bfloat16)
    if is_depthwise:
        model = apply_depthwise_separable_replacement(model, dtype=torch.bfloat16)

    # 2. Load saved weights with prefix adjustment
    safetensors_path = model_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"No model.safetensors in {model_dir}")
    state_dict = load_file(str(safetensors_path))

    model_keys = set(model.state_dict().keys())
    needs_prefix = any(k.startswith("timm_model.") for k in model_keys)
    saved_has_prefix = any(k.startswith("timm_model.") for k in state_dict)
    if needs_prefix and not saved_has_prefix:
        state_dict = {f"timm_model.{k}": v for k, v in state_dict.items()}
    elif not needs_prefix and saved_has_prefix:
        state_dict = {k.replace("timm_model.", "", 1): v for k, v in state_dict.items()}

    # 3. Bypass timm wrapper's strict load_state_dict — use nn.Module directly
    result = nn.Module.load_state_dict(model, state_dict, strict=False)
    if result.missing_keys:
        print(f"    Missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]}")
    if result.unexpected_keys:
        print(f"    Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]}")

    return model


def evaluate_model(model, dataset_name, batch_size=128):
    """Compute accuracy on CIFAR test set.

    Uses the same preprocessing as training (native 32x32 with CIFAR
    normalization), NOT the timm ImageNet processor.
    """
    from datasets import load_dataset
    from torchvision.transforms import Compose, Normalize, ToTensor

    device = next(model.parameters()).device

    if "cifar100" in dataset_name:
        ds = load_dataset("uoft-cs/cifar100", split="test")
        label_key = "fine_label"
        normalize = Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    else:
        ds = load_dataset("uoft-cs/cifar10", split="test")
        label_key = "label"
        normalize = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    val_tf = Compose([ToTensor(), normalize])

    correct = 0
    total = 0
    model.eval()
    for i in range(0, len(ds), batch_size):
        batch = ds[i:i + batch_size]
        images = batch["img"]
        labels = batch[label_key]
        pixel_values = torch.stack([val_tf(img.convert("RGB")) for img in images]).to(device)
        with torch.no_grad():
            logits = model(pixel_values=pixel_values).logits
        preds = logits.argmax(dim=-1).cpu().numpy()
        correct += (preds == np.array(labels)).sum()
        total += len(labels)

    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="*cifar*", help="Glob pattern for run dirs")
    parser.add_argument("--force", action="store_true", help="Re-evaluate even if result exists")
    args = parser.parse_args()

    runs_dir = project_root / "runs"
    results_dir = project_root / "results" / "vision"
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for run_dir in sorted(runs_dir.glob(args.pattern)):
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        if ".old" in run_name or "sweep" in run_name or "bench" in run_name or "__test" in run_name:
            continue

        model_dir = run_dir / "student_model"
        if not model_dir.exists():
            continue

        result_path = results_dir / f"{run_name}.json"
        if result_path.exists() and not args.force:
            print(f"SKIP (result exists): {run_name}")
            continue

        dataset_name = "cifar100" if "cifar100" in run_name else "cifar10"

        print(f"Evaluating: {run_name} on {dataset_name}...")
        try:
            model = load_vision_model(str(model_dir), run_name).to(device)
            acc = evaluate_model(model, dataset_name)
            out = {
                "run": run_name,
                "best_eval_accuracy": acc,
                "dataset": dataset_name,
                "evaluated_from": "student_model",
            }
            with open(result_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"  {acc * 100:.2f}%  {run_name}")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR: {run_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
