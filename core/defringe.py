"""Defringe: remove purple/green colour fringes and lateral chromatic aberration.

1. Lateral CA (optional, automatic): lenses magnify red and blue slightly differently
   from green, so high-contrast edges get red/cyan or blue/yellow outlines that grow
   toward the corners. The red and blue planes are scaled about the image centre by the
   factor that best lines up their edges with green (correlation of high-pass planes on a
   1024 px proxy, searched over +-0.4 %), then resampled at full size.
2. Fringe removal: within a few pixels of strong luminance edges, pixels whose hue falls
   in the purple (longitudinal CA, blooming) or green band and that are clearly coloured
   get their colour (Lab a*, b*) moved toward the colour of the nearby non-fringe pixels.
   Lightness is kept, so the edge stays sharp.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import kornia
import torch
import torch.nn.functional as functional

from .color_lab import rgb_to_hsv
from .guided import box_filter, gaussian_blur_box

PURPLE = (270.0, 335.0)  # hue range (degrees) of purple / magenta fringes
GREEN = (75.0, 165.0)
_CA_RANGE = 0.004  # +- relative scale searched for red and blue
_CA_STEPS = 33
_CA_PROXY = 1024
_CA_MIN_GAIN = 0.01  # relative edge-correlation gain needed to apply a CA correction
_EDGE_THRESHOLD = 0.08  # luma gradient (0..1 per px, Sobel) that counts as a strong edge
_MIN_CHROMA = 6.0  # Lab chroma below which a pixel is not a fringe
_HUE_SOFT = 12.0  # degrees of soft fall-off at the ends of a hue band
_STRIP_ROWS = 512  # rows per fringe-removal strip


def _check(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"image must have shape (B,H,W,3), got {tuple(image.shape)}.")
    if not torch.is_floating_point(image):
        raise ValueError(f"image must be floating point, got {image.dtype}.")
    return torch.nan_to_num(image.float(), nan=0.0).clamp(0.0, 1.0)


def _radial_grid(height: int, width: int, scale: float, device) -> torch.Tensor:
    """Sampling grid that magnifies by ``scale`` about the centre (align_corners=True)."""
    ys = torch.linspace(-1.0, 1.0, height, device=device)
    xs = torch.linspace(-1.0, 1.0, width, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return (torch.stack([gx, gy], dim=-1) / scale).unsqueeze(0)


def _highpass(x: torch.Tensor) -> torch.Tensor:
    return x - gaussian_blur_box(x, 2.0)


def estimate_lateral_ca(image: torch.Tensor) -> tuple[float, float]:
    """Relative scales (red, blue) that align those planes with green, for one (1,3,H,W)."""
    height, width = image.shape[-2:]
    factor = min(1.0, _CA_PROXY / max(height, width))
    proxy = image
    if factor < 1.0:
        size = (max(8, round(height * factor)), max(8, round(width * factor)))
        proxy = functional.interpolate(image, size=size, mode="area")
    green = _highpass(proxy[:, 1:2])
    h, w = proxy.shape[-2:]
    # Corners matter most (CA grows with radius); ignore the centre third.
    ys = torch.linspace(-1, 1, h, device=image.device)[:, None]
    xs = torch.linspace(-1, 1, w, device=image.device)[None, :]
    weight = ((ys**2 + xs**2).sqrt() > 0.35).float()
    scales = []
    for channel in (0, 2):
        plane = proxy[:, channel : channel + 1]
        best, best_score, neutral = 1.0, -math.inf, None
        for step in range(_CA_STEPS):
            s = 1.0 - _CA_RANGE + 2 * _CA_RANGE * step / (_CA_STEPS - 1)
            warped = functional.grid_sample(
                plane,
                _radial_grid(h, w, s, image.device),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            hp = _highpass(warped)
            score = float((hp * green * weight).sum() / ((hp * hp * weight).sum().sqrt() + 1e-9))
            if step == _CA_STEPS // 2:
                neutral = score  # s == 1
            if score > best_score:
                best, best_score = s, score
        # Resampling softens slightly: only correct when it clearly lines the edges up.
        if neutral is None or best_score <= max(neutral, 0.0) * (1.0 + _CA_MIN_GAIN) + 1e-12:
            best = 1.0
        scales.append(best)
    return scales[0], scales[1]


def correct_lateral_ca(image: torch.Tensor, red: float, blue: float) -> torch.Tensor:
    """Rescale the red and blue planes of (B,3,H,W) about the centre."""
    height, width = image.shape[-2:]
    out = image.clone()
    for channel, scale in ((0, red), (2, blue)):
        if abs(scale - 1.0) < 1e-5:
            continue
        grid = _radial_grid(height, width, scale, image.device).expand(image.shape[0], -1, -1, -1)
        out[:, channel : channel + 1] = functional.grid_sample(
            image[:, channel : channel + 1],
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
    return out


def _hue_band(hue: torch.Tensor, band: tuple[float, float]) -> torch.Tensor:
    lo, hi = band
    rise = ((hue - lo + _HUE_SOFT) / _HUE_SOFT).clamp(0.0, 1.0)
    fall = ((hi + _HUE_SOFT - hue) / _HUE_SOFT).clamp(0.0, 1.0)
    return rise * fall


def remove_fringes(image: torch.Tensor, purple: float, green: float, radius: int) -> torch.Tensor:
    """Neutralise purple/green fringes near strong edges in (B,3,H,W), in row strips.

    Every step looks at most 3 * radius + 2 rows away, so strips with that much
    overlap give exactly the whole-image result while capping memory (~2 GB at 4K
    otherwise).
    """
    height = image.shape[-2]
    halo = 3 * max(radius, 1) + 4
    if height <= _STRIP_ROWS + 2 * halo:
        return _remove_fringes(image, purple, green, radius)
    out = torch.empty_like(image)
    for top in range(0, height, _STRIP_ROWS):
        bottom = min(height, top + _STRIP_ROWS)
        lo, hi = max(0, top - halo), min(height, bottom + halo)
        part = _remove_fringes(image[..., lo:hi, :], purple, green, radius)
        out[..., top:bottom, :] = part[..., top - lo : bottom - lo, :]
    return out


def _remove_fringes(image: torch.Tensor, purple: float, green: float, radius: int) -> torch.Tensor:
    lab = kornia.color.rgb_to_lab(image)
    luma = lab[:, :1] / 100.0
    grad = kornia.filters.spatial_gradient(luma, mode="sobel").pow(2).sum(2).sqrt()
    edges = (grad > _EDGE_THRESHOLD).float()
    near = functional.max_pool2d(edges, (1, 2 * radius + 1), stride=1, padding=(0, radius))
    near = functional.max_pool2d(near, (2 * radius + 1, 1), stride=1, padding=(radius, 0))

    hue, _, _ = rgb_to_hsv(image.permute(0, 2, 3, 1))
    hue = hue.squeeze(-1).unsqueeze(1)
    chroma = lab[:, 1:].pow(2).sum(1, keepdim=True).sqrt()
    coloured = ((chroma - _MIN_CHROMA) / _MIN_CHROMA).clamp(0.0, 1.0)
    fringe = (purple * _hue_band(hue, PURPLE) + green * _hue_band(hue, GREEN)).clamp(0.0, 1.0)
    weight = fringe * coloured * near
    if not bool((weight > 0).any()):
        return image

    # Colour the fringe pixels should have: the mean a*b* of nearby non-fringe pixels.
    keep = 1.0 - (weight > 0.05).float()
    window = max(2, 2 * radius)
    num = box_filter(lab[:, 1:] * keep, window)
    den = box_filter(keep, window)
    target = torch.where(den > 1e-3, num / den.clamp_min(1e-3), torch.zeros_like(num))
    ab = lab[:, 1:] + (target - lab[:, 1:]) * weight
    fixed = kornia.color.lab_to_rgb(torch.cat([lab[:, :1], ab], 1)).clamp(0.0, 1.0)
    # Only fringe pixels change (no Lab round-trip drift elsewhere).
    return torch.where(weight > 0, fixed, image)


def defringe(
    image: torch.Tensor,
    *,
    purple: float = 1.0,
    green: float = 0.0,
    radius: int = 0,
    lateral_ca: bool = True,
) -> tuple[torch.Tensor, list[tuple[float, float]]]:
    """Remove colour fringes and (optionally) lateral chromatic aberration.

    Args:
        image: IMAGE ``(B,H,W,3)``.
        purple, green: 0..1 strength for purple/magenta and green fringes. Green is off by
            default: it would also tint foliage seen against a bright sky.
        radius: Fringe width in pixels around edges, 0 = automatic (0.15 % of the long
            edge, at least 2).
        lateral_ca: Also align the red and blue planes with green.

    Returns:
        ``(image, ca_scales)``: corrected image and the (red, blue) scale used per image.
    """
    image = _check(image)
    for name, value in (("purple", purple), ("green", green)):
        if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1].")
    if isinstance(radius, bool) or not isinstance(radius, Integral) or not 0 <= radius <= 32:
        raise ValueError("radius must be an integer in [0, 32].")
    height, width = image.shape[1:3]
    if radius == 0:
        radius = max(2, round(0.0015 * max(height, width)))

    frames, scales = [], []
    for frame in image:
        x = frame.permute(2, 0, 1).unsqueeze(0)
        scale = (1.0, 1.0)
        if lateral_ca:
            scale = estimate_lateral_ca(x)
            x = correct_lateral_ca(x, *scale)
        if purple > 0 or green > 0:
            x = remove_fringes(x, float(purple), float(green), int(radius))
        frames.append(x[0].permute(1, 2, 0))
        scales.append((round(scale[0], 5), round(scale[1], 5)))
    return torch.stack(frames), scales


def run(image: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, list[tuple[float, float]]]:
    """``defringe`` on the GPU when there is one, one frame at a time."""
    if image.device.type == "cpu" and torch.cuda.is_available():
        try:
            outs, scales = [], []
            for index in range(image.shape[0]):
                out, scale = defringe(image[index : index + 1].cuda(), **kwargs)
                outs.append(out.cpu())
                scales.extend(scale)
            return torch.cat(outs), scales
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    return defringe(image, **kwargs)
