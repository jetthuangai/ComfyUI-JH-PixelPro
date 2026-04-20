"""Selective color helpers operating on BHWC float32 RGB tensors."""

from __future__ import annotations

import math
from numbers import Real

import torch
from kornia.color import hsv_to_rgb, rgb_to_hls, rgb_to_hsv


def _prepare_image(image: torch.Tensor, *, device: str | torch.device) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"image must have shape (B,H,W,3), got {tuple(image.shape)}.")
    if not torch.is_floating_point(image):
        raise ValueError(f"image must be floating point, got {image.dtype}.")

    compute_dtype = image.dtype if image.dtype in {torch.float32, torch.float64} else torch.float32
    return image.to(device=device, dtype=compute_dtype).clamp(0.0, 1.0)


def _prepare_mask(
    mask: torch.Tensor,
    *,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor.")
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (B,H,W), got {tuple(mask.shape)}.")
    if mask.shape[-2:] != (height, width):
        raise ValueError(
            f"mask spatial shape must be {(height, width)}, got {tuple(mask.shape[-2:])}."
        )
    if mask.shape[0] not in (1, batch):
        raise ValueError(f"mask batch must be 1 or {batch}, got {mask.shape[0]}.")
    if not torch.is_floating_point(mask) and mask.dtype is not torch.bool:
        raise ValueError(f"mask must be float or bool, got {mask.dtype}.")

    prepared = mask.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    if prepared.shape[0] == 1 and batch > 1:
        prepared = prepared.expand(batch, -1, -1)
    return prepared


def _validate_real(name: str, value: float, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    value_float = float(value)
    if not lower <= value_float <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_float


def _bhwc_to_bchw(image: torch.Tensor) -> torch.Tensor:
    return image.permute(0, 3, 1, 2)


def _bchw_to_bhwc(image: torch.Tensor) -> torch.Tensor:
    return image.permute(0, 2, 3, 1)


def _smoothstep(edge0: torch.Tensor, edge1: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    denom = (edge1 - edge0).clamp_min(1e-6)
    t = ((values - edge0) / denom).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def hue_range_mask(
    image: torch.Tensor,
    hue_center: float,
    band_width: float,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build a soft band mask on the HSV hue wheel."""

    image_prepared = _prepare_image(image, device=device)
    hue_center = _validate_real("hue_center", hue_center, lower=0.0, upper=360.0)
    band_width = _validate_real("band_width", band_width, lower=0.0, upper=180.0)

    hsv = rgb_to_hsv(_bhwc_to_bchw(image_prepared))
    hue = hsv[:, 0]
    saturation = hsv[:, 1]
    hue_center_rad = math.radians(hue_center % 360.0)
    band_width_rad = math.radians(band_width)

    angular_delta = torch.remainder(hue - hue_center_rad + math.pi, 2.0 * math.pi) - math.pi
    distance = angular_delta.abs()

    if band_width_rad <= 0.0:
        mask = (distance <= 1e-6).to(dtype=image_prepared.dtype)
    else:
        mask = (1.0 - distance / band_width_rad).clamp(0.0, 1.0)
    mask = torch.where(saturation > 1e-6, mask, torch.zeros_like(mask))
    return mask.to(dtype=image.dtype)


def apply_hue_sat_shift(
    image: torch.Tensor,
    mask: torch.Tensor,
    hue_shift: float,
    sat_mult: float,
    sat_add: float = 0.0,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Apply hue / saturation adjustments weighted by a BHW mask."""

    image_prepared = _prepare_image(image, device=device)
    batch, height, width, _ = image_prepared.shape
    mask_prepared = _prepare_mask(
        mask,
        batch=batch,
        height=height,
        width=width,
        device=image_prepared.device,
        dtype=image_prepared.dtype,
    )
    hue_shift = _validate_real("hue_shift", hue_shift, lower=-180.0, upper=180.0)
    sat_mult = _validate_real("sat_mult", sat_mult, lower=0.0, upper=2.0)
    sat_add = _validate_real("sat_add", sat_add, lower=-1.0, upper=1.0)

    hsv = rgb_to_hsv(_bhwc_to_bchw(image_prepared))
    hue = hsv[:, 0]
    saturation = hsv[:, 1]
    value = hsv[:, 2]

    mask_bchw = mask_prepared.unsqueeze(1)
    hue_shift_rad = math.radians(hue_shift)
    hue_shifted = torch.remainder(hue + hue_shift_rad * mask_prepared, 2.0 * math.pi)
    saturation_shifted = torch.clamp(
        saturation * (1.0 + (sat_mult - 1.0) * mask_prepared) + sat_add * mask_prepared,
        0.0,
        1.0,
    )

    shifted_hsv = torch.cat(
        [hue_shifted.unsqueeze(1), saturation_shifted.unsqueeze(1), value.unsqueeze(1)], dim=1
    )
    output = hsv_to_rgb(shifted_hsv)
    blended = _bchw_to_bhwc(output) * mask_bchw.permute(0, 2, 3, 1) + image_prepared * (
        1.0 - mask_bchw.permute(0, 2, 3, 1)
    )
    return blended.to(dtype=image.dtype)


def saturation_range_mask(
    image: torch.Tensor,
    sat_min: float,
    sat_max: float,
    feather: float = 0.0,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Mask pixels whose HLS saturation lies inside a range."""

    image_prepared = _prepare_image(image, device=device)
    sat_min = _validate_real("sat_min", sat_min, lower=0.0, upper=1.0)
    sat_max = _validate_real("sat_max", sat_max, lower=0.0, upper=1.0)
    feather = _validate_real("feather", feather, lower=0.0, upper=1.0)
    if sat_max <= sat_min:
        raise ValueError("sat_max must be strictly greater than sat_min.")

    hls = rgb_to_hls(_bhwc_to_bchw(image_prepared))
    saturation = hls[:, 2]

    if feather == 0.0:
        mask = ((saturation >= sat_min) & (saturation <= sat_max)).to(dtype=image_prepared.dtype)
    else:
        lower = _smoothstep(
            torch.full_like(saturation, sat_min - feather),
            torch.full_like(saturation, sat_min),
            saturation,
        )
        upper = 1.0 - _smoothstep(
            torch.full_like(saturation, sat_max),
            torch.full_like(saturation, sat_max + feather),
            saturation,
        )
        mask = (lower * upper).clamp(0.0, 1.0)
    return mask.to(dtype=image.dtype)
