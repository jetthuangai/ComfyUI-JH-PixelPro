"""Unwrap aligned face edits back to the original canvas via inverse affine composite."""

from __future__ import annotations

import logging
import math
from numbers import Real

import torch
from kornia.filters import gaussian_blur2d
from kornia.geometry.transform import warp_affine

logger = logging.getLogger(__name__)


def _prepare_image(name: str, image_bchw: torch.Tensor) -> torch.Tensor:
    if not isinstance(image_bchw, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")

    if image_bchw.ndim != 4:
        raise ValueError(f"Expected {name} BCHW tensor, got shape {tuple(image_bchw.shape)}.")

    if image_bchw.shape[1] != 3:
        raise ValueError(f"Expected {name} 3-channel RGB image, got C={image_bchw.shape[1]}.")

    if image_bchw.dtype != torch.float32:
        raise ValueError(f"Expected {name} float32 image tensor, got {image_bchw.dtype}.")

    value_min, value_max = torch.aminmax(image_bchw.detach())
    if value_min.item() < 0.0 or value_max.item() > 1.0:
        logger.warning("%s input values outside [0,1]; clamped to [0,1].", name)
        return image_bchw.clamp(0.0, 1.0)

    return image_bchw


def _validate_feather_radius(feather_radius: float) -> float:
    if isinstance(feather_radius, bool) or not isinstance(feather_radius, Real):
        raise ValueError("feather_radius must be >= 0.")
    feather = float(feather_radius)
    if feather < 0.0:
        raise ValueError("feather_radius must be >= 0.")
    return feather


def _expand_batch(tensor: torch.Tensor, *, batch: int, name: str) -> torch.Tensor:
    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch, *tensor.shape[1:])
    raise ValueError(
        f"{name} batch ({tensor.shape[0]}) must be 1 or equal to target batch ({batch})."
    )


def _prepare_inverse_matrix(
    inverse_matrix: torch.Tensor, *, batch: int, device: torch.device
) -> torch.Tensor:
    if not isinstance(inverse_matrix, torch.Tensor):
        raise TypeError("inverse_matrix must be a torch.Tensor.")

    if inverse_matrix.ndim != 3:
        raise ValueError(
            f"Expected inverse_matrix Bx3x3 or Bx2x3, got {tuple(inverse_matrix.shape)}."
        )

    if inverse_matrix.shape[-2:] not in {(2, 3), (3, 3)}:
        raise ValueError(
            f"Expected inverse_matrix Bx3x3 or Bx2x3, got {tuple(inverse_matrix.shape)}."
        )

    matrix = inverse_matrix.to(device=device, dtype=torch.float32)
    matrix = _expand_batch(matrix, batch=batch, name="inverse_matrix")
    if matrix.shape[-2:] == (3, 3):
        return matrix[:, :2, :]
    return matrix


def _prepare_mask_override(
    mask_override: torch.Tensor | None,
    *,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor | None:
    if mask_override is None:
        return None

    if not isinstance(mask_override, torch.Tensor):
        raise TypeError("mask_override must be a torch.Tensor or None.")

    if mask_override.ndim == 3:
        mask = mask_override.unsqueeze(1)
    elif mask_override.ndim == 4 and mask_override.shape[1] == 1:
        mask = mask_override
    else:
        raise ValueError(
            "mask_override must have shape BHW or Bx1xH xW compatible with original canvas."
        )

    if mask.dtype == torch.bool:
        logger.warning("bool mask_override cast to float32.")
        mask = mask.to(dtype=torch.float32)
    elif not torch.is_floating_point(mask):
        raise ValueError(f"Expected mask_override float tensor in [0,1], got {mask.dtype}.")
    elif mask.dtype != torch.float32:
        logger.warning("mask_override cast to float32.")
        mask = mask.to(dtype=torch.float32)

    if mask.shape[-2:] != (height, width):
        raise ValueError(
            "mask_override HxW "
            f"{tuple(mask.shape[-2:])} must match original canvas ({height}, {width})."
        )

    mask = _expand_batch(mask.to(device=device), batch=batch, name="mask_override")
    value_min, value_max = torch.aminmax(mask.detach())
    if value_min.item() < 0.0 or value_max.item() > 1.0:
        logger.warning("mask_override values outside [0,1]; clamped to [0,1].")
        mask = mask.clamp(0.0, 1.0)
    return mask


def _gaussian_kernel_size(sigma: float) -> int:
    return max(3, 2 * math.ceil(3.0 * sigma) + 1)


def unwrap_face(
    edited_aligned_bchw: torch.Tensor,
    original_bchw: torch.Tensor,
    inverse_matrix: torch.Tensor,
    *,
    feather_radius: float = 16.0,
    mask_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp an edited aligned crop back to the original canvas and alpha-composite it."""

    edited = _prepare_image("edited_aligned_bchw", edited_aligned_bchw)
    original = _prepare_image("original_bchw", original_bchw)
    feather_radius = _validate_feather_radius(feather_radius)

    batch = max(edited.shape[0], original.shape[0])
    edited = _expand_batch(edited, batch=batch, name="edited_aligned_bchw")
    original = _expand_batch(original, batch=batch, name="original_bchw")
    affine = _prepare_inverse_matrix(inverse_matrix, batch=batch, device=edited.device)

    if original.device != edited.device:
        original = original.to(device=edited.device)

    _, _, aligned_height, aligned_width = edited.shape
    _, _, original_height, original_width = original.shape
    mask = _prepare_mask_override(
        mask_override,
        batch=batch,
        height=original_height,
        width=original_width,
        device=edited.device,
    )

    warped = warp_affine(
        edited,
        affine,
        dsize=(original_height, original_width),
        padding_mode="zeros",
        align_corners=False,
    ).clamp(0.0, 1.0)

    if mask is None:
        indicator = torch.ones(
            (batch, 1, aligned_height, aligned_width),
            dtype=torch.float32,
            device=edited.device,
        )
        warped_indicator = warp_affine(
            indicator,
            affine,
            dsize=(original_height, original_width),
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        ).clamp(0.0, 1.0)
        if feather_radius > 0.0:
            kernel_size = _gaussian_kernel_size(feather_radius)
            mask = gaussian_blur2d(
                warped_indicator,
                kernel_size=(kernel_size, kernel_size),
                sigma=(feather_radius, feather_radius),
                border_type="constant",
            ).clamp(0.0, 1.0)
        else:
            mask = warped_indicator

    composited = ((mask * warped) + ((1.0 - mask) * original)).clamp(0.0, 1.0)
    return composited, mask.squeeze(1)
