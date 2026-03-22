"""Dynamic quantization utilities for mini-sglang.

This module provides dynamic int8 quantization for linear layers.
Weights are quantized to int8 during loading and dequantized on-the-fly during forward pass.
"""
from __future__ import annotations

import torch


def dynamic_quantize_int8_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform per-channel dynamic int8 quantization on weight tensor.

    Args:
        weight: 2D weight tensor of shape (out_features, in_features)

    Returns:
        qweight: int8 quantized weight
        scale: per-channel scale factor (out_features,)
    """
    assert weight.dim() == 2, f"Expected 2D weight, got {weight.dim()}D"

    # Compute per-channel (row-wise) min/max
    min_val = weight.min(dim=1, keepdim=True)[0]
    max_val = weight.max(dim=1, keepdim=True)[0]

    # Compute scale: scale = max(|min|, |max|) / 127
    eps = torch.finfo(torch.float32).eps
    max_abs = torch.maximum(-min_val, max_val)
    scale = max_abs / 127.0
    scale = torch.clamp(scale, min=eps)

    # Quantize: q = round(w / scale) and clamp to [-127, 127]
    qweight = torch.round(weight / scale)
    qweight = torch.clamp(qweight, -127, 127).to(torch.int8)

    # Return scale as 1D tensor
    scale = scale.squeeze(dim=1)

    return qweight, scale


def dequantize_int8_per_channel(
    qweight: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Dequantize int8 weight back to target dtype.

    Args:
        qweight: int8 quantized weight (out_features, in_features)
        scale: per-channel scale (out_features,) or scalar
        dtype: target dtype (e.g., torch.bfloat16)

    Returns:
        Dequantized weight tensor
    """
    # Reshape scale for broadcasting: (out_features, 1)
    if scale.numel() == 1:
        # Per-tensor quantization
        return (qweight.to(dtype) * scale.to(dtype))

    # Per-channel quantization
    if scale.dim() == 1:
        scale = scale.view(-1, 1)

    # Dequantize: w = q * scale
    return (qweight.to(dtype) * scale.to(dtype))
