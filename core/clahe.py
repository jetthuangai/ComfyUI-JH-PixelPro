"""CLAHE local contrast on the L* channel of CIE LAB (colour-preserving)."""

from __future__ import annotations

from numbers import Integral, Real

import torch
from kornia.color import lab_to_rgb, rgb_to_lab
from kornia.enhance import equalize_clahe


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


def _validate_image(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"image must have shape (B,H,W,3), got {tuple(image.shape)}.")
    if not torch.is_floating_point(image):
        raise ValueError(f"image must be a floating point tensor, got {image.dtype}.")
    return image.to(dtype=torch.float32)


def _validate_mask(mask: torch.Tensor | None, image: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor.")
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (B,H,W), got {tuple(mask.shape)}.")
    if not torch.is_floating_point(mask) and mask.dtype is not torch.bool:
        raise ValueError(f"mask must be float or bool, got {mask.dtype}.")
    if mask.shape[1:] != image.shape[1:3]:
        raise ValueError("mask and image spatial shapes must match.")
    if mask.shape[0] not in (1, image.shape[0]):
        raise ValueError("mask batch must match image batch or be 1.")
    return mask.to(device=image.device, dtype=torch.float32).clamp(0.0, 1.0)


# Kornia derives the clip threshold with integer maths on a 256-bin histogram; small tiles
# shift even a flat image (up to 0.14 below ~47 px, up to ~0.018 at 52-76 px; from 80 px on
# the drift stays within one 8-bit quantisation step pair, <= 0.01; measured, clip_limit 2).
_MIN_TILE_PX = 80


def _effective_grid(grid_size: int, height: int, width: int) -> int:
    """Largest grid <= ``grid_size`` whose tiles are at least ``_MIN_TILE_PX``; 0 = too small."""
    return min(grid_size, min(height, width) // _MIN_TILE_PX)


def clahe_local_contrast(
    image: torch.Tensor,
    *,
    clip_limit: float = 2.0,
    grid_size: int = 8,
    strength: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply CLAHE to L* only, keeping a*/b* (hue and chroma) untouched.

    The image is converted to CIE LAB, ``L/100`` is equalised with
    ``kornia.enhance.equalize_clahe`` on a ``grid_size x grid_size`` tile grid, and the
    result is rebuilt with the original a*/b* before the final clamp to ``[0,1]``.
    Kornia works on 8-bit histograms, so L* is quantised to 256 levels. Its clip threshold
    biases small tiles (a flat image drifts brighter), so the grid is reduced
    automatically until every tile is at least ``_MIN_TILE_PX`` wide; images smaller than
    one such tile are returned unchanged.

    Args:
        image: IMAGE tensor ``(B,H,W,3)`` RGB in ``[0,1]``.
        clip_limit: Histogram clip limit in ``[0.1, 40.0]`` (higher = stronger contrast).
        grid_size: Tiles per axis, integer in ``[2, 32]``; reduced on small images (see above).
        strength: Blend between original and enhanced in ``[0, 1]``.
        mask: Optional MASK ``(B,H,W)`` (batch 1 broadcasts) limiting the effect:
            ``out = lerp(original, enhanced, strength * mask)``.

    Returns:
        Float32 IMAGE tensor ``(B,H,W,3)`` in ``[0,1]`` on the input device.

    Raises:
        TypeError: If ``image`` or ``mask`` is not a tensor.
        ValueError: If shapes, dtypes, or parameters are invalid.
    """

    prepared = _validate_image(image)
    clip_limit = _validate_float("clip_limit", clip_limit, lower=0.1, upper=40.0)
    grid_size = _validate_int("grid_size", grid_size, lower=2, upper=32)
    strength = _validate_float("strength", strength, lower=0.0, upper=1.0)
    prepared_mask = _validate_mask(mask, prepared)
    height, width = prepared.shape[1:3]
    grid_size = _effective_grid(grid_size, height, width)

    original = prepared.clamp(0.0, 1.0)
    if strength == 0.0 or prepared.shape[0] == 0 or grid_size == 0:
        return original.clone()

    rgb = original.permute(0, 3, 1, 2).contiguous()
    lab = rgb_to_lab(rgb)
    lightness = (lab[:, 0:1] / 100.0).clamp(0.0, 1.0)
    equalised = equalize_clahe(lightness, clip_limit, (grid_size, grid_size))
    new_lab = torch.cat([equalised * 100.0, lab[:, 1:3]], dim=1)
    enhanced = lab_to_rgb(new_lab).clamp(0.0, 1.0).permute(0, 2, 3, 1)

    weight = strength
    if prepared_mask is not None:
        weight = strength * prepared_mask.unsqueeze(-1)
    out = original + (enhanced - original) * weight
    return out.clamp(0.0, 1.0).to(dtype=torch.float32)
