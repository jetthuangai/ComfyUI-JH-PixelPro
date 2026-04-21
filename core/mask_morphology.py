"""Classical morphology operations for ComfyUI MASK tensors."""

from __future__ import annotations

from numbers import Integral

import cv2
import numpy as np
import torch

MORPHOLOGY_OPERATIONS = (
    "dilate",
    "erode",
    "open",
    "close",
    "gradient",
    "tophat",
    "blackhat",
)

_CV2_OPS = {
    "open": cv2.MORPH_OPEN,
    "close": cv2.MORPH_CLOSE,
    "gradient": cv2.MORPH_GRADIENT,
    "tophat": cv2.MORPH_TOPHAT,
    "blackhat": cv2.MORPH_BLACKHAT,
}


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


def _validate_operation(operation: str) -> str:
    if operation not in MORPHOLOGY_OPERATIONS:
        allowed = ", ".join(MORPHOLOGY_OPERATIONS)
        raise ValueError(f"operation must be one of: {allowed}.")
    return operation


def _ellipse_kernel(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def mask_morphology(
    mask: torch.Tensor,
    *,
    operation: str = "dilate",
    radius: int = 3,
    iterations: int = 1,
) -> torch.Tensor:
    """Apply a classical morphology operation to a ComfyUI MASK tensor.

    Args:
        mask: Input MASK tensor ``(B,H,W)``.
        operation: One of ``dilate``, ``erode``, ``open``, ``close``, ``gradient``,
            ``tophat``, or ``blackhat``.
        radius: Elliptical structuring element radius.
        iterations: Number of morphology passes.

    Returns:
        Refined MASK tensor ``(B,H,W)`` in ``[0,1]``.

    Raises:
        TypeError: If mask is not a tensor.
        ValueError: If shape, dtype, operation, radius, or iterations is invalid.
    """

    prepared = _validate_mask(mask)
    operation = _validate_operation(operation)
    radius = _validate_int("radius", radius, lower=1, upper=64)
    iterations = _validate_int("iterations", iterations, lower=1, upper=10)

    kernel = _ellipse_kernel(radius)
    outputs = []
    for batch_mask in prepared.detach().to("cpu").numpy():
        source = np.ascontiguousarray(batch_mask.astype(np.float32, copy=False))
        if operation == "dilate":
            morphed = cv2.dilate(source, kernel, iterations=iterations)
        elif operation == "erode":
            morphed = cv2.erode(source, kernel, iterations=iterations)
        else:
            morphed = cv2.morphologyEx(
                source,
                _CV2_OPS[operation],
                kernel,
                iterations=iterations,
            )
        outputs.append(torch.from_numpy(np.clip(morphed, 0.0, 1.0).astype(np.float32)))
    return torch.stack(outputs, dim=0).to(device=mask.device, dtype=torch.float32)
