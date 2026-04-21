"""Guided-filter edge-aware mask refinement for ComfyUI MASK tensors."""

from __future__ import annotations

from numbers import Integral, Real

import torch
import torch.nn.functional as functional
from kornia.filters import gaussian_blur2d


def _validate_mask(mask: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor.")
    preserve_channel_dim = False
    if mask.ndim == 3:
        mask_bchw = mask.unsqueeze(1)
    elif mask.ndim == 4 and mask.shape[1] == 1:
        mask_bchw = mask
        preserve_channel_dim = True
    else:
        raise ValueError(f"mask must have shape (B,H,W) or (B,1,H,W), got {tuple(mask.shape)}.")
    if not torch.is_floating_point(mask_bchw) and mask_bchw.dtype is not torch.bool:
        raise ValueError(f"mask must be float or bool, got {mask_bchw.dtype}.")
    return mask_bchw.to(dtype=torch.float32).clamp(0.0, 1.0), preserve_channel_dim


def _validate_guide(guide: torch.Tensor, *, batch: int, height: int, width: int) -> torch.Tensor:
    if not isinstance(guide, torch.Tensor):
        raise TypeError("guide must be a torch.Tensor.")
    if guide.ndim != 4 or guide.shape[-1] != 3:
        raise ValueError(f"guide must have shape (B,H,W,3), got {tuple(guide.shape)}.")
    if guide.shape[0] not in (1, batch):
        raise ValueError(f"guide batch must be 1 or {batch}, got {guide.shape[0]}.")
    if guide.shape[1:3] != (height, width):
        raise ValueError(f"guide spatial shape must be {(height, width)}, got {guide.shape[1:3]}.")
    if not torch.is_floating_point(guide):
        raise ValueError(f"guide must be floating point, got {guide.dtype}.")
    prepared = guide.to(dtype=torch.float32).clamp(0.0, 1.0)
    if prepared.shape[0] == 1 and batch > 1:
        prepared = prepared.expand(batch, -1, -1, -1)
    return prepared


def _validate_int(name: str, value: int, *, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}].")
    value_int = int(value)
    if not lower <= value_int <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_int


def _validate_float(name: str, value: float, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    value_float = float(value)
    if not lower <= value_float <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_float


def _box_filter(x: torch.Tensor, radius: int) -> torch.Tensor:
    kernel_size = 2 * radius + 1
    return functional.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=radius)


def _guide_luma(guide_bhwc: torch.Tensor) -> torch.Tensor:
    weights = guide_bhwc.new_tensor([0.299, 0.587, 0.114])
    return (guide_bhwc * weights).sum(dim=-1, keepdim=True).permute(0, 3, 1, 2)


def edge_aware_refine(
    mask: torch.Tensor,
    guide: torch.Tensor,
    *,
    radius: int = 8,
    eps: float = 1e-3,
    feather_sigma: float = 0.0,
) -> torch.Tensor:
    """Refine a mask with a grayscale guided filter and optional Gaussian feather.

    Args:
        mask: ComfyUI mask tensor shaped ``(B,H,W)`` or ``(B,1,H,W)`` in ``[0,1]``.
        guide: RGB ComfyUI image tensor shaped ``(B,H,W,3)`` used as the edge guide.
        radius: Guided-filter window radius in pixels.
        eps: Regularization term controlling edge adherence.
        feather_sigma: Optional post-filter Gaussian sigma. ``0`` disables feathering.

    Returns:
        Refined mask in ``[0,1]`` preserving the input mask rank.

    Raises:
        TypeError: If inputs are not tensors.
        ValueError: If shapes, dtypes, or parameters are invalid.
    """

    mask_bchw, preserve_channel_dim = _validate_mask(mask)
    batch, _, height, width = mask_bchw.shape
    guide_prepared = _validate_guide(guide, batch=batch, height=height, width=width).to(
        device=mask_bchw.device
    )
    radius = _validate_int("radius", radius, lower=1, upper=32)
    eps = _validate_float("eps", eps, lower=1e-6, upper=1.0)
    feather_sigma = _validate_float("feather_sigma", feather_sigma, lower=0.0, upper=64.0)

    guide_luma = _guide_luma(guide_prepared)
    mean_i = _box_filter(guide_luma, radius)
    mean_p = _box_filter(mask_bchw, radius)
    corr_i = _box_filter(guide_luma * guide_luma, radius)
    corr_ip = _box_filter(guide_luma * mask_bchw, radius)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = _box_filter(a, radius)
    mean_b = _box_filter(b, radius)
    refined = (mean_a * guide_luma + mean_b).clamp(0.0, 1.0)

    if feather_sigma > 0.0:
        kernel_size = max(3, 2 * int(round(feather_sigma * 3.0)) + 1)
        refined = gaussian_blur2d(
            refined,
            kernel_size=(kernel_size, kernel_size),
            sigma=(feather_sigma, feather_sigma),
            border_type="replicate",
        ).clamp(0.0, 1.0)

    refined = refined.to(device=mask.device, dtype=torch.float32)
    return refined if preserve_channel_dim else refined.squeeze(1)
