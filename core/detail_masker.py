"""High-frequency detail mask generation for BCHW RGB tensors."""

from __future__ import annotations

import logging
import math
from numbers import Real

import torch
from kornia.filters import gaussian_blur2d, laplacian, sobel

logger = logging.getLogger(__name__)

_VALID_KERNELS = {"laplacian", "sobel", "fs_gaussian"}
_VALID_THRESHOLDS = {"relative_percentile", "absolute"}


def _validate_sensitivity(sensitivity: float) -> float:
    if isinstance(sensitivity, bool) or not isinstance(sensitivity, Real):
        raise ValueError("sensitivity must be in [0.0, 1.0].")

    sensitivity_float = float(sensitivity)
    if not 0.0 <= sensitivity_float <= 1.0:
        raise ValueError("sensitivity must be in [0.0, 1.0].")
    return sensitivity_float


def _validate_choice(name: str, value: str, valid: set[str]) -> str:
    if not isinstance(value, str) or value not in valid:
        raise ValueError(f"{name} must be one of {tuple(sorted(valid))}.")
    return value


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

    if mask_bchw.ndim == 3:
        mask = mask_bchw.unsqueeze(1)
    elif mask_bchw.ndim == 4:
        if mask_bchw.shape[1] != 1:
            raise ValueError(f"Expected BHW or BC1HW mask, got C={mask_bchw.shape[1]}.")
        mask = mask_bchw
    else:
        raise ValueError(f"Expected BHW or BC1HW mask, got shape {tuple(mask_bchw.shape)}.")

    image_batch, _, image_height, image_width = image_bchw.shape
    mask_batch, _, mask_height, mask_width = mask.shape
    if (mask_height, mask_width) != (image_height, image_width):
        raise ValueError(
            f"mask HxW ({mask_height}, {mask_width}) != image HxW ({image_height}, {image_width})."
        )

    if mask_batch not in (1, image_batch):
        raise ValueError(
            f"mask batch ({mask_batch}) must be 1 or equal to image batch ({image_batch})."
        )

    if mask.dtype == torch.bool:
        logger.warning("bool mask input cast to float32.")
        mask = mask.to(dtype=torch.float32)
    elif not torch.is_floating_point(mask):
        raise ValueError(f"Expected float mask tensor in [0,1], got {mask.dtype}.")
    elif mask.dtype != torch.float32:
        logger.warning("mask input cast to float32.")
        mask = mask.to(dtype=torch.float32)

    value_min, value_max = torch.aminmax(mask.detach())
    if value_min.item() < 0.0 or value_max.item() > 1.0:
        logger.warning("mask input values outside [0,1]; clamped to [0,1].")
        mask = mask.clamp(0.0, 1.0)

    if mask_batch == 1 and image_batch > 1:
        mask = mask.expand(image_batch, -1, -1, -1)
    return mask


def _high_pass_luma(luma_bchw: torch.Tensor, kernel_type: str) -> torch.Tensor:
    if kernel_type == "laplacian":
        return laplacian(luma_bchw, kernel_size=3, border_type="replicate")
    if kernel_type == "sobel":
        return sobel(luma_bchw, normalized=True)

    low = gaussian_blur2d(
        luma_bchw,
        kernel_size=(5, 5),
        sigma=(1.0, 1.0),
        border_type="replicate",
    )
    return luma_bchw - low


def high_freq_detail_mask(
    image_bchw: torch.Tensor,
    *,
    sensitivity: float = 0.5,
    kernel_type: str = "laplacian",
    threshold_mode: str = "relative_percentile",
    mask_bchw: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate a binary detail-preservation mask from high-frequency energy."""

    sensitivity = _validate_sensitivity(sensitivity)
    kernel_type = _validate_choice("kernel_type", kernel_type, _VALID_KERNELS)
    threshold_mode = _validate_choice("threshold_mode", threshold_mode, _VALID_THRESHOLDS)
    image = _prepare_image(image_bchw)
    mask = _prepare_mask(mask_bchw, image)

    batch, _, height, width = image.shape
    if sensitivity == 0.0:
        output = torch.zeros((batch, height, width), dtype=torch.float32, device=image.device)
        return output if mask is None else output * mask.squeeze(1)

    if sensitivity == 1.0:
        output = torch.ones((batch, height, width), dtype=torch.float32, device=image.device)
        return output if mask is None else output * mask.squeeze(1)

    luma = image.mean(dim=1, keepdim=True)
    hp_abs = _high_pass_luma(luma, kernel_type).abs()

    if threshold_mode == "relative_percentile":
        flat = hp_abs.flatten(1)
        kth_index = max(1, min(flat.shape[1], math.ceil((1.0 - sensitivity) * flat.shape[1])))
        thresholds = flat.kthvalue(kth_index, dim=1).values.view(batch, 1, 1, 1)
        output = (hp_abs >= thresholds).to(dtype=torch.float32)
    else:
        max_vals = hp_abs.flatten(1).amax(dim=1).clamp_min(1e-8).view(batch, 1, 1, 1)
        hp_norm = hp_abs / max_vals
        output = (hp_norm >= (1.0 - sensitivity)).to(dtype=torch.float32)

    if mask is not None:
        output = output * mask

    return output.squeeze(1)
