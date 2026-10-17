"""Sub-pixel mask refinement core math for BHW and BC1HW mask tensors."""

from __future__ import annotations

import logging
import math
from numbers import Real

import torch
import torch.nn.functional as functional
from kornia.filters import gaussian_blur2d

logger = logging.getLogger(__name__)


def _validate_int_param(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int in [0, 64].")
    if not 0 <= value <= 64:
        raise ValueError(f"{name} must be in [0, 64].")


def _validate_real_param(name: str, value: float, *, lower: float, upper: float | None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        if upper is None:
            raise ValueError(f"{name} must be > {lower}.")
        raise ValueError(f"{name} must be in [{lower}, {upper}].")

    value_float = float(value)
    if upper is None:
        if value_float <= lower:
            raise ValueError(f"{name} must be > {lower} (use 0.1 for minimal feather).")
    elif not lower <= value_float <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_float


def _prepare_mask(mask_bhw: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if not isinstance(mask_bhw, torch.Tensor):
        raise TypeError("mask_bhw must be a torch.Tensor.")

    preserve_channel_dim = False
    if mask_bhw.ndim == 3:
        mask_bchw = mask_bhw.unsqueeze(1)
    elif mask_bhw.ndim == 4:
        if mask_bhw.shape[1] != 1:
            raise ValueError(f"Expected BHW or BC1HW mask, got C={mask_bhw.shape[1]}.")
        mask_bchw = mask_bhw
        preserve_channel_dim = True
    else:
        raise ValueError(f"Expected BHW or BC1HW mask, got shape {tuple(mask_bhw.shape)}.")

    if mask_bchw.dtype == torch.bool:
        logger.warning("bool mask input cast to float32.")
        mask_bchw = mask_bchw.to(dtype=torch.float32)
    elif not torch.is_floating_point(mask_bchw):
        raise ValueError(
            f"Expected float tensor in [0,1], got {mask_bchw.dtype}. Normalize upstream first."
        )
    elif mask_bchw.dtype != torch.float32:
        logger.warning("mask input cast to float32.")
        mask_bchw = mask_bchw.to(dtype=torch.float32)

    value_min, value_max = torch.aminmax(mask_bchw.detach())
    if value_min.item() < 0.0 or value_max.item() > 1.0:
        logger.warning("mask input values outside [0,1]; clamped to [0,1].")
        mask_bchw = mask_bchw.clamp(0.0, 1.0)

    midtone_ratio = ((mask_bchw >= 0.1) & (mask_bchw <= 0.9)).float().mean().item()
    if midtone_ratio > 0.05:
        logger.warning("input not really binary; more than 5%% of pixels lie in [0.1, 0.9].")

    return mask_bchw, preserve_channel_dim


def _dilate_square(mask_bchw: torch.Tensor, radius: int) -> torch.Tensor:
    # Square all-ones dilation == sliding max, which separates into a row max then a
    # column max. O(H*W) memory, unlike kornia.morphology's unfold engine
    # (O(H*W*k^2), multi-GB at large radii); -inf padding matches its geodesic border.
    kernel_size = 2 * radius + 1
    rows = functional.max_pool2d(
        mask_bchw, kernel_size=(1, kernel_size), stride=1, padding=(0, radius)
    )
    return functional.max_pool2d(rows, kernel_size=(kernel_size, 1), stride=1, padding=(radius, 0))


def _erode_square(mask_bchw: torch.Tensor, radius: int) -> torch.Tensor:
    return -_dilate_square(-mask_bchw, radius)


def _feather_kernel_size(feather_sigma: float, radius: int, height: int, width: int) -> int:
    # Cover +-3 sigma so the Gaussian is not truncated into a box when sigma is large
    # relative to the radii; never smaller than the band, capped at twice the image extent.
    kernel_size = max(2 * max(radius, 1) + 1, 2 * math.ceil(3.0 * feather_sigma) + 1)
    return min(kernel_size, 2 * max(height, width) + 1)


def subpixel_mask_refine(
    mask_bhw: torch.Tensor,
    erosion_radius: int = 2,
    dilation_radius: int = 4,
    feather_sigma: float = 2.0,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Refine a BHW/BC1HW mask into a sub-pixel alpha mask.

    Args:
        mask_bhw: Input mask tensor in `BHW` layout with values in `[0, 1]`. `BC1HW`
            is also accepted and preserved on output for callers that already keep the
            channel dimension.
        erosion_radius: Radius in pixels for the "definitely inside" erosion core.
        dilation_radius: Radius in pixels for the "definitely outside" dilation core.
        feather_sigma: Gaussian sigma used to feather the uncertain band.
        threshold: Strict binarization threshold. Pixels use `mask > threshold`.

    Returns:
        A float32 refined mask on the same device, preserving the input rank:
        `BHW -> BHW`, `BC1HW -> BC1HW`.

    Raises:
        TypeError: If `mask_bhw` is not a tensor.
        ValueError: If the mask shape, dtype, or parameters are invalid.
    """

    _validate_int_param("erosion_radius", erosion_radius)
    _validate_int_param("dilation_radius", dilation_radius)
    feather_sigma = _validate_real_param("feather_sigma", feather_sigma, lower=0.0, upper=None)
    threshold = _validate_real_param("threshold", threshold, lower=0.0, upper=1.0)

    mask_bchw, preserve_channel_dim = _prepare_mask(mask_bhw)
    mask_binary = (mask_bchw > threshold).to(dtype=torch.float32)

    kernel_size = _feather_kernel_size(
        feather_sigma,
        max(erosion_radius, dilation_radius),
        mask_binary.shape[-2],
        mask_binary.shape[-1],
    )
    feather = gaussian_blur2d(
        mask_binary,
        kernel_size=(kernel_size, kernel_size),
        sigma=(feather_sigma, feather_sigma),
        border_type="replicate",
    )

    if erosion_radius == 0 and dilation_radius == 0:
        logger.warning("no protected zone, output = clamped gaussian blur.")
        output = feather.clamp(0.0, 1.0)
        return output if preserve_channel_dim else output.squeeze(1)

    if erosion_radius > dilation_radius:
        logger.warning("unusual config: erosion_radius > dilation_radius.")

    inside_core = mask_binary if erosion_radius == 0 else _erode_square(mask_binary, erosion_radius)

    if dilation_radius == 0:
        outside_core = 1.0 - mask_binary
    else:
        outside_core = 1.0 - _dilate_square(mask_binary, dilation_radius)

    band_weight = 1.0 - inside_core - outside_core
    output = inside_core + band_weight * feather
    output = output.clamp(0.0, 1.0)
    return output if preserve_channel_dim else output.squeeze(1)
