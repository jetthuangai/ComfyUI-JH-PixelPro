"""Edge-smoothing filters for ComfyUI MASK tensors."""

from __future__ import annotations

from numbers import Integral, Real

import cv2
import numpy as np
import torch


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


def mask_edge_smooth(
    mask: torch.Tensor,
    guide: torch.Tensor | None = None,
    *,
    sigma_spatial: float = 4.0,
    sigma_range: float = 0.1,
    iterations: int = 1,
) -> torch.Tensor:
    """Smooth MASK edges with bilateral or guide-aware joint bilateral filtering.

    Args:
        mask: Input MASK tensor ``(B,H,W)``.
        guide: Optional IMAGE tensor ``(B,H,W,3)`` used by OpenCV ximgproc joint
            bilateral filtering when available.
        sigma_spatial: Spatial bilateral sigma.
        sigma_range: Range/color bilateral sigma for normalized ``[0,1]`` inputs.
        iterations: Number of smoothing passes.

    Returns:
        Smoothed MASK tensor ``(B,H,W)`` in ``[0,1]``.

    Raises:
        TypeError: If inputs are not tensors.
        ValueError: If shapes, dtypes, or parameters are invalid.
    """

    prepared_mask = _validate_mask(mask)
    prepared_guide = _broadcast_guide(prepared_mask, _validate_guide(guide))
    if prepared_guide is not None:
        prepared_guide = prepared_guide.to(device=prepared_mask.device)
    sigma_spatial = _validate_float("sigma_spatial", sigma_spatial, lower=0.1, upper=64.0)
    sigma_range = _validate_float("sigma_range", sigma_range, lower=0.001, upper=1.0)
    iterations = _validate_int("iterations", iterations, lower=1, upper=5)
    diameter = _diameter(sigma_spatial)

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
