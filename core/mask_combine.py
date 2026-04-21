"""MASK combine operations for ComfyUI tensors."""

from __future__ import annotations

from numbers import Real

import torch
from kornia.filters import gaussian_blur2d

MASK_COMBINE_OPERATIONS = (
    "add",
    "subtract",
    "intersect",
    "union",
    "difference",
    "xor",
    "multiply",
)
MASK_COMBINE_BLEND_MODES = ("hard", "soft_feather")


def _validate_mask(name: str, mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if mask.ndim != 3:
        raise ValueError(f"{name} must have shape (B,H,W), got {tuple(mask.shape)}.")
    if not torch.is_floating_point(mask) and mask.dtype is not torch.bool:
        raise ValueError(f"{name} must be float or bool, got {mask.dtype}.")
    return mask.to(dtype=torch.float32).clamp(0.0, 1.0)


def _validate_float(name: str, value: float, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    value_float = float(value)
    if not lower <= value_float <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_float


def _validate_choice(name: str, value: str, choices: tuple[str, ...]) -> str:
    if value not in choices:
        allowed = ", ".join(choices)
        raise ValueError(f"{name} must be one of: {allowed}.")
    return value


def _broadcast_masks(
    mask_a: torch.Tensor, mask_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask_a.shape[1:] != mask_b.shape[1:]:
        raise ValueError("mask_a and mask_b spatial shapes must match.")
    if mask_a.shape[0] == mask_b.shape[0]:
        return mask_a, mask_b
    if mask_a.shape[0] == 1:
        return mask_a.expand(mask_b.shape[0], -1, -1), mask_b
    if mask_b.shape[0] == 1:
        return mask_a, mask_b.expand(mask_a.shape[0], -1, -1)
    raise ValueError("mask_a and mask_b batch sizes must match or one batch must be 1.")


def _soft_feather(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return mask
    kernel_size = max(3, 2 * int(round(sigma * 3.0)) + 1)
    return gaussian_blur2d(
        mask.unsqueeze(1),
        kernel_size=(kernel_size, kernel_size),
        sigma=(sigma, sigma),
        border_type="replicate",
    ).squeeze(1)


def _combine(mask_a: torch.Tensor, mask_b: torch.Tensor, operation: str) -> torch.Tensor:
    if operation == "add":
        return mask_a + mask_b
    if operation == "subtract":
        return mask_a - mask_b
    if operation == "intersect":
        return torch.minimum(mask_a, mask_b)
    if operation == "union":
        return torch.maximum(mask_a, mask_b)
    if operation == "difference":
        return mask_a * (1.0 - mask_b)
    if operation == "xor":
        return mask_a + mask_b - 2.0 * mask_a * mask_b
    if operation == "multiply":
        return mask_a * mask_b
    raise AssertionError(f"unreachable operation: {operation}")


def combine_masks(
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    *,
    operation: str = "union",
    blend_mode: str = "hard",
    opacity: float = 1.0,
    feather_sigma: float = 0.0,
) -> torch.Tensor:
    """Combine two ComfyUI MASK tensors with Photoshop-style mask ops.

    Args:
        mask_a: Base MASK tensor ``(B,H,W)``.
        mask_b: Secondary MASK tensor ``(B,H,W)``.
        operation: Combine operation.
        blend_mode: ``hard`` uses masks directly; ``soft_feather`` blurs both masks
            before combining.
        opacity: Blend factor from ``mask_a`` to the combined result.
        feather_sigma: Gaussian sigma used by ``soft_feather``.

    Returns:
        Combined MASK tensor ``(B,H,W)`` in ``[0,1]``.

    Raises:
        TypeError: If masks are not tensors.
        ValueError: If shapes, dtypes, operation, mode, or parameters are invalid.
    """

    prepared_a = _validate_mask("mask_a", mask_a)
    prepared_b = _validate_mask("mask_b", mask_b).to(device=prepared_a.device)
    prepared_a, prepared_b = _broadcast_masks(prepared_a, prepared_b)
    operation = _validate_choice("operation", operation, MASK_COMBINE_OPERATIONS)
    blend_mode = _validate_choice("blend_mode", blend_mode, MASK_COMBINE_BLEND_MODES)
    opacity = _validate_float("opacity", opacity, lower=0.0, upper=1.0)
    feather_sigma = _validate_float("feather_sigma", feather_sigma, lower=0.0, upper=64.0)

    if blend_mode == "soft_feather":
        prepared_a = _soft_feather(prepared_a, feather_sigma)
        prepared_b = _soft_feather(prepared_b, feather_sigma)

    combined = _combine(prepared_a, prepared_b, operation).clamp(0.0, 1.0)
    return (prepared_a * (1.0 - opacity) + combined * opacity).clamp(0.0, 1.0)
