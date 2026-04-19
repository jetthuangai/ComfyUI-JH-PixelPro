"""Tone curve application via a Catmull-Rom LUT for BCHW RGB tensors."""

from __future__ import annotations

import logging
from numbers import Real

import torch

logger = logging.getLogger(__name__)

_VALID_CHANNELS = {"rgb_master", "r", "g", "b"}
_CHANNEL_TO_INDEX = {"r": 0, "g": 1, "b": 2}
_LUT_SIZE = 1024


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
        logger.warning("image values outside [0,1]; clamped to [0,1].")
        return image_bchw.clamp(0.0, 1.0)

    return image_bchw


def _validate_channel(channel: str) -> str:
    if not isinstance(channel, str) or channel not in _VALID_CHANNELS:
        raise ValueError(f"channel must be one of {tuple(sorted(_VALID_CHANNELS))}.")
    return channel


def _validate_strength(strength: float) -> float:
    if isinstance(strength, bool) or not isinstance(strength, Real):
        raise ValueError("strength must be in [0.0, 1.0].")

    strength_float = float(strength)
    if not 0.0 <= strength_float <= 1.0:
        raise ValueError("strength must be in [0.0, 1.0].")
    return strength_float


def _prepare_control_points(control_points: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if not isinstance(control_points, torch.Tensor):
        raise TypeError("control_points must be a torch.Tensor.")

    if control_points.shape != (8, 2):
        raise ValueError(
            f"control_points must have shape (8, 2), got {tuple(control_points.shape)}."
        )

    if not torch.is_floating_point(control_points):
        raise ValueError(f"control_points must be float32 in [0,1], got {control_points.dtype}.")

    points = control_points.to(device=device, dtype=torch.float32)
    value_min, value_max = torch.aminmax(points.detach())
    if value_min.item() < 0.0 or value_max.item() > 1.0:
        raise ValueError("control_points values must be in [0,1].")

    origin = torch.tensor([0.0, 0.0], device=device, dtype=torch.float32)
    endpoint = torch.tensor([1.0, 1.0], device=device, dtype=torch.float32)
    if not torch.allclose(points[0], origin, atol=1e-6):
        raise ValueError("control_points endpoints must start at (0,0).")
    if not torch.allclose(points[-1], endpoint, atol=1e-6):
        raise ValueError("control_points endpoints must end at (1,1).")

    if torch.any(torch.diff(points[:, 0]) <= 0.0):
        raise ValueError("control_points x values must be strictly increasing.")

    return points


def _build_lut(control_points: torch.Tensor) -> torch.Tensor:
    x_points = control_points[:, 0]
    y_points = control_points[:, 1]
    sample_x = torch.linspace(
        0.0,
        1.0,
        _LUT_SIZE,
        dtype=control_points.dtype,
        device=control_points.device,
    )

    segment_index = torch.searchsorted(x_points[1:].contiguous(), sample_x, right=False)
    segment_index = segment_index.clamp(max=control_points.shape[0] - 2)

    idx_prev = (segment_index - 1).clamp(min=0)
    idx_start = segment_index
    idx_end = segment_index + 1
    idx_next = (segment_index + 2).clamp(max=control_points.shape[0] - 1)

    x0 = x_points[idx_prev]
    x1 = x_points[idx_start]
    x2 = x_points[idx_end]
    x3 = x_points[idx_next]
    y0 = y_points[idx_prev]
    y1 = y_points[idx_start]
    y2 = y_points[idx_end]
    y3 = y_points[idx_next]

    delta = (x2 - x1).clamp_min(1e-6)
    t = ((sample_x - x1) / delta).clamp(0.0, 1.0)
    t2 = t * t
    t3 = t2 * t

    m1 = (y2 - y0) / (x2 - x0).clamp_min(1e-6)
    m2 = (y3 - y1) / (x3 - x1).clamp_min(1e-6)

    h00 = (2.0 * t3) - (3.0 * t2) + 1.0
    h10 = t3 - (2.0 * t2) + t
    h01 = (-2.0 * t3) + (3.0 * t2)
    h11 = t3 - t2

    lut = h00 * y1 + h10 * delta * m1 + h01 * y2 + h11 * delta * m2
    lut = lut.clamp(0.0, 1.0)
    lut[0] = 0.0
    lut[-1] = 1.0
    return lut


def _sample_lut(values: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    positions = values.clamp(0.0, 1.0) * float(_LUT_SIZE - 1)
    lower = torch.floor(positions).to(dtype=torch.long)
    upper = (lower + 1).clamp(max=_LUT_SIZE - 1)
    weight = positions - lower.to(dtype=values.dtype)
    lower_values = lut[lower]
    upper_values = lut[upper]
    return torch.lerp(lower_values, upper_values, weight)


def tone_curve(
    image_bchw: torch.Tensor,
    *,
    control_points: torch.Tensor,
    channel: str = "rgb_master",
    strength: float = 1.0,
) -> torch.Tensor:
    """Apply a tone-curve LUT to an RGB image tensor."""

    image = _prepare_image(image_bchw)
    channel = _validate_channel(channel)
    strength = _validate_strength(strength)
    points = _prepare_control_points(control_points, device=image.device)

    if strength == 0.0:
        return image

    lut = _build_lut(points)
    curved = image.clone()
    if channel == "rgb_master":
        curved = _sample_lut(curved, lut)
    else:
        channel_index = _CHANNEL_TO_INDEX[channel]
        curved[:, channel_index : channel_index + 1] = _sample_lut(
            image[:, channel_index : channel_index + 1],
            lut,
        )

    output = ((1.0 - strength) * image) + (strength * curved)
    return torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
