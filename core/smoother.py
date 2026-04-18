"""Edge-aware smoothing core math for BCHW RGB tensors."""

from __future__ import annotations

import logging
import math
from numbers import Real

import torch
from kornia.filters import bilateral_blur

logger = logging.getLogger(__name__)


def _validate_strength(strength: float) -> float:
    if isinstance(strength, bool) or not isinstance(strength, Real):
        raise ValueError("strength must be in [0.0, 1.0].")

    strength_float = float(strength)
    if not 0.0 <= strength_float <= 1.0:
        raise ValueError("strength must be in [0.0, 1.0].")
    return strength_float


def _validate_positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be > 0.")

    value_float = float(value)
    if value_float <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return value_float


def _prepare_image(image_bchw: torch.Tensor) -> torch.Tensor:
    if not isinstance(image_bchw, torch.Tensor):
        raise TypeError("image_bchw must be a torch.Tensor.")

    if image_bchw.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(image_bchw.shape)}.")

    if image_bchw.shape[1] != 3:
        raise ValueError(f"Expected 3-channel RGB image, got C={image_bchw.shape[1]}.")

    if image_bchw.dtype != torch.float32:
        raise ValueError(f"Expected float32 image tensor, got {image_bchw.dtype}.")

    value_min, value_max = torch.aminmax(image_bchw.detach())
    if value_min.item() < 0.0 or value_max.item() > 1.0:
        logger.warning("image input values outside [0,1]; clamped to [0,1].")
        return image_bchw.clamp(0.0, 1.0)

    return image_bchw


def _prepare_mask(mask_bchw: torch.Tensor | None, image_bchw: torch.Tensor) -> torch.Tensor | None:
    if mask_bchw is None:
        return None

    if not isinstance(mask_bchw, torch.Tensor):
        raise TypeError("mask_bchw must be a torch.Tensor or None.")

    if mask_bchw.ndim != 4:
        raise ValueError(f"Expected BC1HW mask tensor, got shape {tuple(mask_bchw.shape)}.")

    if mask_bchw.shape[1] != 1:
        raise ValueError(f"Expected single-channel mask, got C={mask_bchw.shape[1]}.")

    image_batch, _, image_height, image_width = image_bchw.shape
    mask_batch, _, mask_height, mask_width = mask_bchw.shape

    if (mask_height, mask_width) != (image_height, image_width):
        raise ValueError(
            f"mask HxW ({mask_height}, {mask_width}) != image HxW ({image_height}, {image_width})."
        )

    if mask_batch not in (1, image_batch):
        raise ValueError(
            f"mask batch ({mask_batch}) must be 1 or equal to image batch ({image_batch})."
        )

    if mask_bchw.dtype == torch.bool:
        logger.warning("bool mask input cast to float32.")
        mask = mask_bchw.to(dtype=torch.float32)
    elif not torch.is_floating_point(mask_bchw):
        raise ValueError(f"Expected float mask tensor in [0,1], got {mask_bchw.dtype}.")
    elif mask_bchw.dtype != torch.float32:
        logger.warning("mask input cast to float32.")
        mask = mask_bchw.to(dtype=torch.float32)
    else:
        mask = mask_bchw

    value_min, value_max = torch.aminmax(mask.detach())
    if value_min.item() < 0.0 or value_max.item() > 1.0:
        logger.warning("mask input values outside [0,1]; clamped to [0,1].")
        mask = mask.clamp(0.0, 1.0)

    return mask.to(device=image_bchw.device)


def edge_aware_smooth(
    image_bchw: torch.Tensor,
    strength: float = 0.4,
    sigma_color: float = 0.1,
    sigma_space: float = 6.0,
    mask_bchw: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply bilateral smoothing with optional strength blend and mask gating."""

    strength = _validate_strength(strength)
    sigma_color = _validate_positive("sigma_color", sigma_color)
    sigma_space = _validate_positive("sigma_space", sigma_space)
    image = _prepare_image(image_bchw)
    mask = _prepare_mask(mask_bchw, image)

    if sigma_color > 1.0:
        logger.warning("sigma_color scale is [0,1]; values > 1 will smooth across most edges.")

    if strength == 0.0:
        return image.clone()

    if mask is not None and mask.sum().item() == 0.0:
        return image.clone()

    kernel_size = 2 * math.ceil(3.0 * sigma_space) + 1
    smoothed = bilateral_blur(
        image,
        kernel_size=(kernel_size, kernel_size),
        sigma_color=sigma_color,
        sigma_space=(sigma_space, sigma_space),
        border_type="replicate",
        color_distance_type="l1",
    )
    blended = strength * smoothed + (1.0 - strength) * image

    output = blended if mask is None else mask * blended + (1.0 - mask) * image

    return output.clamp(0.0, 1.0)
