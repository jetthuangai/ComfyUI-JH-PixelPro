"""Luminosity mask generation for BCHW RGB tensors."""

from __future__ import annotations

import logging
from numbers import Real

import torch
from kornia.color import rgb_to_hsv, rgb_to_ycbcr

logger = logging.getLogger(__name__)

_VALID_SOURCES = {"lab_l", "ycbcr_y", "hsv_v"}


def _validate_choice(name: str, value: str, valid: set[str]) -> str:
    if not isinstance(value, str) or value not in valid:
        raise ValueError(f"{name} must be one of {tuple(sorted(valid))}.")
    return value


def _validate_real_range(name: str, value: float, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be in [{lower}, {upper}].")

    value_float = float(value)
    if not lower <= value_float <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
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


def _luminance(image_bchw: torch.Tensor, source: str) -> torch.Tensor:
    if source == "lab_l":
        threshold = 0.04045
        linear = torch.where(
            image_bchw > threshold,
            ((image_bchw + 0.055) / 1.055) ** 2.4,
            image_bchw / 12.92,
        )
        y_channel = (
            (0.2126 * linear[:, 0:1])
            + (0.7152 * linear[:, 1:2])
            + (0.0722 * linear[:, 2:3])
        )
        delta = 6.0 / 29.0
        f_y = torch.where(
            y_channel > delta**3,
            y_channel.clamp_min(1e-8).pow(1.0 / 3.0),
            y_channel / (3.0 * delta * delta) + (4.0 / 29.0),
        )
        return ((116.0 * f_y) - 16.0).clamp(0.0, 100.0) / 100.0
    if source == "ycbcr_y":
        return rgb_to_ycbcr(image_bchw)[:, :1].clamp(0.0, 1.0)
    return rgb_to_hsv(image_bchw)[:, 2:3].clamp(0.0, 1.0)


def _smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    width = max(edge1 - edge0, 1e-6)
    t = ((x - edge0) / width).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def luminosity_masks(
    image_bchw: torch.Tensor,
    *,
    luminance_source: str = "lab_l",
    shadow_end: float = 0.33,
    highlight_start: float = 0.67,
    soft_edge: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate three normalized luminosity masks."""

    luminance_source = _validate_choice("luminance_source", luminance_source, _VALID_SOURCES)
    shadow_end = _validate_real_range("shadow_end", shadow_end, lower=0.0, upper=0.5)
    highlight_start = _validate_real_range(
        "highlight_start", highlight_start, lower=0.5, upper=1.0
    )
    soft_edge = _validate_real_range("soft_edge", soft_edge, lower=0.01, upper=0.3)
    if shadow_end >= highlight_start:
        raise ValueError("shadow_end must be < highlight_start.")

    image = _prepare_image(image_bchw)
    luma = _luminance(image, luminance_source)
    half_edge = soft_edge / 2.0

    shadows = 1.0 - _smoothstep(shadow_end - half_edge, shadow_end + half_edge, luma)
    highlights = _smoothstep(
        highlight_start - half_edge,
        highlight_start + half_edge,
        luma,
    )
    midtones = (1.0 - shadows - highlights).clamp_min(0.0)
    total = (shadows + midtones + highlights).clamp_min(1e-8)

    shadows = (shadows / total).squeeze(1)
    midtones = (midtones / total).squeeze(1)
    highlights = (highlights / total).squeeze(1)
    return shadows, midtones, highlights
