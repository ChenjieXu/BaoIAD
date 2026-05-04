"""Shared score normalization utilities."""

from __future__ import annotations

import torch


def minmax_normalize(
    tensor: torch.Tensor,
    eps: float = 1e-6,
    dim: int | None = None,
) -> torch.Tensor:
    """Min-max normalize a tensor, clamping the denominator to avoid division by zero.

    Args:
        tensor: Input tensor to normalize.
        eps: Small value added to denominator for numerical stability.
        dim: Dimension along which to normalize. If None, normalize globally.

    Returns:
        Normalized tensor in [0, 1] range (per-dimension if dim is specified).
    """
    if dim is None:
        mins = tensor.min()
        maxs = tensor.max()
    else:
        mins = tensor.min(dim=dim, keepdim=True).values
        maxs = tensor.max(dim=dim, keepdim=True).values
    return (tensor - mins) / (maxs - mins + eps)


def normalize_class_name(class_name) -> str:
    """Normalize dataset class name for prompt generation (replace underscores/hyphens with spaces)."""
    return str(class_name).replace('_', ' ').replace('-', ' ')


def safe_l2_normalize(x: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    """Numerically stable L2 normalization."""
    norm = x.float().norm(dim=dim, keepdim=True).clamp_min(eps).to(dtype=x.dtype)
    return x / norm
