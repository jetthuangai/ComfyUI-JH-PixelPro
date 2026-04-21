"""3-value trimap helpers for ComfyUI MASK tensors."""

from __future__ import annotations

from numbers import Integral, Real

import cv2
import numpy as np
import torch
from kornia.filters import gaussian_blur2d


def _validate_mask(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor.")
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (B,H,W), got {tuple(mask.shape)}.")
    if not torch.is_floating_point(mask) and mask.dtype is not torch.bool:
        raise ValueError(f"mask must be float or bool, got {mask.dtype}.")
    return mask.to(dtype=torch.float32).clamp(0.0, 1.0)


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


def _ellipse_kernel(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def validate_trimap(trimap: torch.Tensor, tolerance: float = 0.05) -> torch.Tensor:
    """Validate and return a 3-value trimap MASK tensor.

    Args:
        trimap: Tensor ``(B,H,W)`` encoded as ``0.0`` BG, ``0.5`` Unknown, and
            ``1.0`` FG.
        tolerance: Allowed absolute tolerance around each encoded value.

    Returns:
        Float32 trimap tensor clamped to ``[0,1]``.

    Raises:
        TypeError: If trimap is not a tensor.
        ValueError: If shape, dtype, tolerance, or value encoding is invalid.
    """

    prepared = _validate_mask(trimap)
    tolerance = _validate_float("tolerance", tolerance, lower=0.0, upper=0.25)
    bg = (prepared - 0.0).abs() <= tolerance
    unknown = (prepared - 0.5).abs() <= tolerance
    fg = (prepared - 1.0).abs() <= tolerance
    valid = bg | unknown | fg
    if not valid.all():
        raise ValueError("trimap must use 0.0 BG / 0.5 Unknown / 1.0 FG values.")
    return prepared


def build_trimap(
    mask: torch.Tensor,
    *,
    fg_radius: int = 4,
    bg_radius: int = 8,
    smoothing: float = 0.0,
) -> torch.Tensor:
    """Build a 3-value trimap from a binary or soft ComfyUI MASK tensor.

    Args:
        mask: Input mask ``(B,H,W)``. Values above ``0.5`` are foreground.
        fg_radius: Erosion radius for the definite foreground core.
        bg_radius: Erosion radius for the definite background core.
        smoothing: Optional Gaussian sigma before thresholding.

    Returns:
        Trimap MASK ``(B,H,W)`` with values ``0.0`` BG, ``0.5`` Unknown, ``1.0`` FG.

    Raises:
        TypeError: If mask is not a tensor.
        ValueError: If shape, dtype, or parameters are invalid.
    """

    prepared = _validate_mask(mask)
    fg_radius = _validate_int("fg_radius", fg_radius, lower=1, upper=64)
    bg_radius = _validate_int("bg_radius", bg_radius, lower=1, upper=64)
    smoothing = _validate_float("smoothing", smoothing, lower=0.0, upper=16.0)
    if smoothing > 0.0:
        kernel_size = max(3, 2 * int(round(smoothing * 3.0)) + 1)
        prepared = gaussian_blur2d(
            prepared.unsqueeze(1),
            kernel_size=(kernel_size, kernel_size),
            sigma=(smoothing, smoothing),
            border_type="replicate",
        ).squeeze(1)

    binary = (prepared > 0.5).detach().to("cpu", torch.uint8).numpy()
    trimaps = []
    fg_kernel = _ellipse_kernel(fg_radius)
    bg_kernel = _ellipse_kernel(bg_radius)
    for batch_mask in binary:
        fg = cv2.erode(batch_mask, fg_kernel, iterations=1).astype(bool)
        bg = cv2.erode((1 - batch_mask).astype(np.uint8), bg_kernel, iterations=1).astype(bool)
        unknown = ~(fg | bg)
        trimap = np.zeros_like(batch_mask, dtype=np.float32)
        trimap[unknown] = 0.5
        trimap[fg] = 1.0
        trimaps.append(torch.from_numpy(trimap))
    return torch.stack(trimaps, dim=0).to(device=mask.device, dtype=torch.float32)
