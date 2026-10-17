"""Edge-smoothing filters for ComfyUI MASK tensors."""

from __future__ import annotations

import logging
from numbers import Integral, Real

import cv2
import numpy as np
import torch

_BACKENDS = ("opencv", "gpu")
_LOGGER = logging.getLogger(__name__)


def _validate_mask(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor.")
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (B,H,W), got {tuple(mask.shape)}.")
    if not torch.is_floating_point(mask) and mask.dtype is not torch.bool:
        raise ValueError(f"mask must be float or bool, got {mask.dtype}.")
    return mask.to(dtype=torch.float32).clamp(0.0, 1.0)


def _validate_guide(guide: torch.Tensor | None) -> torch.Tensor | None:
    if guide is None:
        return None
    if not isinstance(guide, torch.Tensor):
        raise TypeError("guide must be a torch.Tensor.")
    if guide.ndim != 4 or guide.shape[-1] != 3:
        raise ValueError(f"guide must have shape (B,H,W,3), got {tuple(guide.shape)}.")
    if not torch.is_floating_point(guide):
        raise ValueError(f"guide must be a floating point tensor, got {guide.dtype}.")
    return guide.to(dtype=torch.float32).clamp(0.0, 1.0)


def _validate_float(name: str, value: float, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    value_float = float(value)
    if not lower <= value_float <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_float


def _validate_int(name: str, value: int, *, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}].")
    value_int = int(value)
    if not lower <= value_int <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_int


def _broadcast_guide(mask: torch.Tensor, guide: torch.Tensor | None) -> torch.Tensor | None:
    if guide is None:
        return None
    if guide.shape[1:3] != mask.shape[1:3]:
        raise ValueError("guide and mask spatial shapes must match.")
    if guide.shape[0] == mask.shape[0]:
        return guide
    if guide.shape[0] == 1:
        return guide.expand(mask.shape[0], -1, -1, -1)
    raise ValueError("guide batch must match mask batch or be 1.")


def _diameter(sigma_spatial: float) -> int:
    return max(3, 2 * int(round(sigma_spatial)) + 1)


def _joint_bilateral(
    source: np.ndarray,
    guide: np.ndarray | None,
    *,
    diameter: int,
    sigma_range: float,
    sigma_spatial: float,
) -> np.ndarray:
    joint_bilateral = getattr(getattr(cv2, "ximgproc", None), "jointBilateralFilter", None)
    if guide is not None and joint_bilateral is not None:
        try:
            return joint_bilateral(
                guide,
                source,
                diameter,
                sigma_range,
                sigma_spatial,
            )
        except (AttributeError, cv2.error):
            pass
    return cv2.bilateralFilter(source, diameter, sigma_range, sigma_spatial)


def _reflect101_index(size: int, pad: int, device: torch.device) -> torch.Tensor:
    """Indices that reflect-pad (edge pixel not repeated) a length-``size`` axis by ``pad``."""
    positions = torch.arange(-pad, size + pad, device=device)
    if size == 1:
        return torch.zeros_like(positions)
    period = 2 * (size - 1)
    folded = positions.abs() % period
    return torch.where(folded >= size, period - folded, folded)


def _gpu_pass(
    source: torch.Tensor,
    guide: torch.Tensor | None,
    *,
    kernel_size: int,
    sigma_range: float,
    sigma_spatial: float,
) -> torch.Tensor:
    """One (joint) bilateral pass on ``(B,1,H,W)`` with an optional ``(B,C,H,W)`` guide.

    Shift-and-accumulate over the ``k x k`` square window (L1 colour distance, reflect-101
    borders), numerically equal to ``kornia.filters.joint_bilateral_blur`` but with
    O(C*H*W) memory instead of O(C*H*W*k*k).
    """
    pad = kernel_size // 2
    height, width = source.shape[-2:]
    rows_idx = _reflect101_index(height, pad, source.device)
    cols_idx = _reflect101_index(width, pad, source.device)

    def reflect(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.index_select(2, rows_idx).index_select(3, cols_idx)

    padded_source = reflect(source)
    guide_ref = source if guide is None else guide
    padded_guide = padded_source if guide is None else reflect(guide)
    color_scale = -0.5 / (sigma_range * sigma_range)
    space_scale = -0.5 / (sigma_spatial * sigma_spatial)
    numerator = torch.zeros_like(source)
    denominator = torch.zeros_like(source)
    weight = torch.empty_like(source)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            window = (slice(None), slice(None), slice(dy, dy + height), slice(dx, dx + width))
            distance = (padded_guide[window] - guide_ref).abs().sum(1, keepdim=True)
            torch.mul(distance, distance, out=weight)
            weight.mul_(color_scale).add_(space_scale * ((dy - pad) ** 2 + (dx - pad) ** 2))
            weight.exp_()
            denominator.add_(weight)
            numerator.addcmul_(weight, padded_source[window])
    return numerator / denominator


def _smooth_gpu(
    mask: torch.Tensor,
    guide: torch.Tensor | None,
    *,
    diameter: int,
    sigma_range: float,
    sigma_spatial: float,
    iterations: int,
) -> torch.Tensor:
    original_device = mask.device
    device = original_device
    if device.type != "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            _LOGGER.warning("mask_edge_smooth: gpu backend is running on CPU and will be slow.")
    smoothed = mask.unsqueeze(1).to(device)
    gpu_guide = None if guide is None else guide.permute(0, 3, 1, 2).to(device)
    for _ in range(iterations):
        smoothed = _gpu_pass(
            smoothed,
            gpu_guide,
            kernel_size=diameter,
            sigma_range=sigma_range,
            sigma_spatial=sigma_spatial,
        )
    return smoothed[:, 0].clamp(0.0, 1.0).to(device=original_device, dtype=torch.float32)


def mask_edge_smooth(
    mask: torch.Tensor,
    guide: torch.Tensor | None = None,
    *,
    sigma_spatial: float = 4.0,
    sigma_range: float = 0.1,
    iterations: int = 1,
    backend: str = "opencv",
) -> torch.Tensor:
    """Smooth MASK edges with bilateral or guide-aware joint bilateral filtering.

    Args:
        mask: Input MASK tensor ``(B,H,W)``.
        guide: Optional IMAGE tensor ``(B,H,W,3)`` used by OpenCV ximgproc joint
            bilateral filtering when available.
        sigma_spatial: Spatial bilateral sigma.
        sigma_range: Range/color bilateral sigma for normalized ``[0,1]`` inputs.
        iterations: Number of smoothing passes.
        backend: ``"opencv"`` (CPU; guide only honoured with opencv-contrib) or ``"gpu"``
            (pure torch on CUDA when available, else CPU with a warning; guide always
            honoured; memory O(C*H*W)).

    Returns:
        Smoothed MASK tensor ``(B,H,W)`` in ``[0,1]``.

    Raises:
        TypeError: If inputs are not tensors.
        ValueError: If shapes, dtypes, or parameters are invalid.
    """

    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}.")
    prepared_mask = _validate_mask(mask)
    prepared_guide = _broadcast_guide(prepared_mask, _validate_guide(guide))
    if prepared_guide is not None:
        prepared_guide = prepared_guide.to(device=prepared_mask.device)
    sigma_spatial = _validate_float("sigma_spatial", sigma_spatial, lower=0.1, upper=64.0)
    sigma_range = _validate_float("sigma_range", sigma_range, lower=0.001, upper=1.0)
    iterations = _validate_int("iterations", iterations, lower=1, upper=5)
    diameter = _diameter(sigma_spatial)

    if backend == "gpu":
        return _smooth_gpu(
            prepared_mask.detach(),
            None if prepared_guide is None else prepared_guide.detach(),
            diameter=diameter,
            sigma_range=sigma_range,
            sigma_spatial=sigma_spatial,
            iterations=iterations,
        )

    mask_np = prepared_mask.detach().to("cpu").numpy()
    guide_np = None
    if prepared_guide is not None:
        guide_np = prepared_guide.detach().to("cpu").numpy()

    outputs = []
    for batch_index, batch_mask in enumerate(mask_np):
        smoothed = np.ascontiguousarray(batch_mask.astype(np.float32, copy=False))
        batch_guide = None
        if guide_np is not None:
            batch_guide = np.ascontiguousarray(guide_np[batch_index].astype(np.float32, copy=False))
        for _ in range(iterations):
            smoothed = _joint_bilateral(
                smoothed,
                batch_guide,
                diameter=diameter,
                sigma_range=sigma_range,
                sigma_spatial=sigma_spatial,
            )
        outputs.append(torch.from_numpy(np.clip(smoothed, 0.0, 1.0).astype(np.float32)))
    return torch.stack(outputs, dim=0).to(device=mask.device, dtype=torch.float32)
