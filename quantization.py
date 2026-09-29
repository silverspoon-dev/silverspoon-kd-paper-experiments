"""Quantization-Aware Training (QAT) via Straight-Through Estimator."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class _QuantizeSTE(torch.autograd.Function):
    """STE: forward quantizes, backward passes gradients unchanged."""

    @staticmethod
    def forward(ctx, x, quantize_fn):
        return quantize_fn(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def _quantize_float8(x: torch.Tensor, float8_dtype: torch.dtype) -> torch.Tensor:
    return x.to(float8_dtype).to(x.dtype)


def _quantize_symmetric_int(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-channel integer quantization."""
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    if x.ndim >= 2:
        abs_max = x.detach().abs().amax(dim=1, keepdim=True)
    else:
        abs_max = x.detach().abs().amax()
    abs_max = abs_max.clamp(min=1e-8)
    scale = abs_max / qmax
    return x.div(scale).round_().clamp_(qmin, qmax).mul_(scale)


_QUANTIZE_DISPATCH = {
    "float8_e4m3fn": lambda x: _quantize_float8(x, torch.float8_e4m3fn),
    "float8_e5m2": lambda x: _quantize_float8(x, torch.float8_e5m2),
}


def make_quantize_fn(spec: str):
    """Create a quantization function from a spec string."""
    if spec in _QUANTIZE_DISPATCH:
        return _QUANTIZE_DISPATCH[spec]
    if spec.startswith("int"):
        bits = int(spec[3:])
        return lambda x: _quantize_symmetric_int(x, bits)
    raise ValueError(f"Unknown quantization spec: {spec}")


class QuantizedSimLinear(nn.Module):
    """Drop-in nn.Linear replacement that simulates reduced-precision weights via STE."""

    def __init__(self, original: nn.Linear, quantize_fn, spec: str):
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias
        self.quantize_fn = quantize_fn
        self.spec = spec
        self.in_features = original.in_features
        self.out_features = original.out_features

    def forward(self, x):
        w = _QuantizeSTE.apply(self.weight, self.quantize_fn)
        return nn.functional.linear(x, w, self.bias)

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, simulated={self.spec}")


def apply_quantization_simulation(model: nn.Module, spec: str) -> nn.Module:
    """Replace all nn.Linear layers with STE-simulated quantized versions."""
    quantize_fn = make_quantize_fn(spec)
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            replacements.append((name, module))
    for name, module in replacements:
        parts = name.rsplit(".", 1)
        parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
        setattr(parent, parts[-1], QuantizedSimLinear(module, quantize_fn, spec))
    logger.info("Quantization simulation (%s): replaced %d Linear layers", spec, len(replacements))
    return model
