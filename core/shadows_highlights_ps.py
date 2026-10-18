# Portions translated from PhotoCraft `crates/algo/src/tone/shadows_highlights.rs`
# (github.com/storytold/photocraft @ ead6d2197f5d0b82d70f3fddb70e472fc27ce577):
# the Beta mask table, strength/curve, Color and Midtone tables are theirs.
# Copyright (c) 2026 ArtCraft Team and the PhotoCraft contributors. MIT License,
# see THIRD_PARTY_NOTICES.md.
"""Photoshop-style Image > Adjustments > Shadows/Highlights.

PhotoCraft measured Photoshop 25.4 as a black box; the model:

- **Neighbourhood brightness**: per pixel ``(max(R,G,B) + Rec.601 luma) / 2``,
  averaged edge-aware (Gaussian sigma = 1.14 * Radius, weights falling linearly with
  the brightness difference; Durand-Dorsey piecewise-linear levels), quantised to
  8 bits.
- **Tonal mask**: complement of a Beta CDF of that brightness; Tone moves it from
  Beta(1.95, 23) at 0 to ``1 - L`` at 95+. Highlights use the mirrored mask.
- **Curve**: ``1 - (1 - v)^q`` blended between q = 1, 2, 4, 8 by Amount.
- **Colour**: only the new luma is kept; the pixel's colour differences are added
  back scaled by the Color slider and shortened to stay in gamut.
- **Midtone**: global S-curve ``k (Y - 1/2) (4Y(1 - Y))^p``; Black/White Clip
  stretch the new luma.

Works on the sRGB-encoded values, as Photoshop does in an sRGB document.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as functional

from .guided import gaussian_blur_box

_REC601 = (0.299, 0.587, 0.114)
_SPATIAL = 1.1424  # neighbourhood sigma per pixel of Radius
_GUIDE = 1.3559  # sigma of the blur the edge weights compare with
_RANGE = 1.0152  # brightness difference where a neighbour's weight reaches zero
_LEVELS = 17

_MASK_BETA = (
    (1.9468, 22.9366),
    (1.7770, 18.3471),
    (1.6403, 14.7559),
    (1.5270, 11.9346),
    (1.4269, 9.6467),
    (1.3470, 7.8430),
    (1.2807, 6.4038),
    (1.2240, 5.2384),
    (1.1772, 4.2967),
    (1.1373, 3.5350),
    (1.1021, 2.9109),
    (1.0755, 2.4186),
    (1.0508, 2.0189),
    (1.0307, 1.7034),
    (1.0140, 1.4583),
    (1.0015, 1.2758),
    (1.0, 1.1403),
    (1.0, 1.0510),
    (1.0, 1.0002),
    (1.0, 1.0),
)
_COLOR_UP = (
    (0.0, 0.0),
    (1.0, 0.1566),
    (2.0, 0.2271),
    (5.0, 0.3776),
    (10.0, 0.5653),
    (20.0, 0.8669),
    (30.0, 1.1299),
    (40.0, 1.3748),
    (50.0, 1.6093),
    (60.0, 1.8375),
    (75.0, 2.1721),
    (100.0, 2.7193),
)
_COLOR_DOWN = (
    (0.0, 0.0),
    (1.0, -0.0145),
    (5.0, -0.0709),
    (10.0, -0.1373),
    (20.0, -0.2570),
    (30.0, -0.3612),
    (50.0, -0.5312),
    (75.0, -0.6866),
    (100.0, -0.7968),
)
_MIDTONE = (
    (-100.0, -0.6955, 2.267),
    (-75.0, -0.5873, 2.362),
    (-50.0, -0.4429, 2.492),
    (-25.0, -0.2514, 2.667),
    (-10.0, -0.1088, 2.803),
    (0.0, 0.0, 2.9),
    (10.0, 0.1206, 3.016),
    (25.0, 0.3262, 3.216),
    (50.0, 0.7423, 3.625),
    (75.0, 1.2635, 4.160),
    (100.0, 1.9032, 4.850),
)


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb * rgb.new_tensor(_REC601).view(1, 3, 1, 1)).sum(1, keepdim=True)


def _bilinear_axis(n_out: int, n_in: int, d: int, device: torch.device):
    f = ((torch.arange(n_out, device=device, dtype=torch.float32) + 0.5) / d - 0.5).clamp(
        0, n_in - 1
    )
    i0 = f.floor().long()
    return i0, (i0 + 1).clamp(max=n_in - 1), f - i0


def _neighbourhood(lum: torch.Tensor, radius: float) -> torch.Tensor:
    """Edge-aware neighbourhood brightness (B,1,H,W) for Photoshop's Radius."""

    guide = gaussian_blur_box(lum, _GUIDE)
    sigma = _SPATIAL * radius
    if sigma < 0.3:
        return guide
    height, width = lum.shape[-2:]
    d = max(1, math.floor(sigma / 4.0))
    pad = (0, (-width) % d, 0, (-height) % d)
    step = 1.0 / (_LEVELS - 1)
    t = (guide.clamp(0.0, 1.0) / step).clamp(max=_LEVELS - 1 - 1e-4)
    y0, y1, ty = _bilinear_axis(height, math.ceil(height / d), d, lum.device)
    x0, x1, tx = _bilinear_axis(width, math.ceil(width / d), d, lum.device)
    ty, tx = ty.view(-1, 1), tx.view(1, -1)

    out = torch.zeros_like(lum)
    for k in range(_LEVELS):
        g = k * step
        weight = (1.0 - (lum - g).abs() / _RANGE).clamp_min(0.0)
        cells = (
            functional.avg_pool2d(functional.pad(torch.cat((weight * lum, weight), 1), pad), d)
            * d
            * d
        )
        cells = gaussian_blur_box(cells, sigma / d)
        avg = torch.where(cells[:, 1:] > 1e-6, cells[:, :1] / cells[:, 1:].clamp_min(1e-12), g)
        top = avg[..., y0, :]
        bot = avg[..., y1, :]
        rows = top + (bot - top) * ty
        level = rows[..., x0] + (rows[..., x1] - rows[..., x0]) * tx
        out += level * (1.0 - (t - k).abs()).clamp_min(0.0)
    return out


@lru_cache(maxsize=64)
def _tonal_mask_table(tone: float) -> tuple[float, ...]:
    """1 - Beta CDF at the 256 levels for Tone 0..100 (midpoint sums, 16 per level)."""

    t = min(max(tone, 0.0), 95.0) / 5.0
    i = min(math.floor(t), len(_MASK_BETA) - 2)
    f = t - i
    (a0, b0), (a1, b1) = _MASK_BETA[i], _MASK_BETA[i + 1]
    a, b = a0 + (a1 - a0) * f, b0 + (b1 - b0) * f
    sub = 16
    xs = (np.arange(255 * sub, dtype=np.float64) + 0.5) / (255 * sub)
    dens = np.exp((a - 1.0) * np.log(xs) + (b - 1.0) * np.log1p(-xs))
    cum = np.concatenate(([0.0], np.cumsum(dens)[sub - 1 :: sub]))
    total = max(cum[-1], np.finfo(np.float64).tiny)
    return tuple((1.0 - cum / total).tolist())


def _tonal_mask(tone: float, like: torch.Tensor) -> torch.Tensor:
    return like.new_tensor(_tonal_mask_table(float(tone)))


def _strength(amount: float) -> float:
    """Curve strength: 0 at Amount 0, ~1 at 15, 2 at 50, 3 at 100."""

    u = (min(max(amount, 0.0), 100.0) / 100.0) ** 0.6216
    return u / 0.3 if u <= 0.3 else (1.0 + 20.0 * u) / 7.0


def _lift(s: float, v: torch.Tensor) -> torch.Tensor:
    n = min(max(math.floor(s), 0), 2)
    t = min(max(s - n, 0.0), 1.0)
    lo = (1.0 - v).clamp(0.0, 1.0) ** (1 << n)
    return 1.0 - ((1.0 - t) * lo + t * lo * lo)


def _interp(table: tuple[tuple[float, float], ...], x: float) -> float:
    xs, ys = zip(*table, strict=True)
    return float(np.interp(x, xs, ys))


def _midtone(m: float, y: torch.Tensor) -> torch.Tensor:
    if m == 0.0:
        return y
    k = _interp(tuple((r[0], r[1]) for r in _MIDTONE), m)
    p = _interp(tuple((r[0], r[2]) for r in _MIDTONE), m)
    return (y + k * (y - 0.5) * (4.0 * y * (1.0 - y)).clamp_min(0.0) ** p).clamp(0.0, 1.0)


def _clip_levels(y2: torch.Tensor, black: float, white: float) -> torch.Tensor:
    """Stretch each image's new luma so ``black``% reach 0 and ``white``% reach 1."""

    if black <= 0.0 and white <= 0.0:
        return y2
    out = y2.clone()
    for i in range(y2.shape[0]):
        bins = (y2[i].clamp(0.0, 1.0) * 4095.0).round().long().flatten()
        cum = torch.bincount(bins, minlength=4096).cumsum(0)
        n = int(cum[-1])

        def find(pct: float, cumulative: torch.Tensor, n: int = n) -> int:
            target = int(min(max(pct, 0.0), 50.0) / 100.0 * n)
            return int(torch.searchsorted(cumulative, target, right=True))

        lo = find(black, cum) / 4095.0 if black > 0.0 else 0.0
        rev = torch.bincount(4095 - bins, minlength=4096).cumsum(0)
        hi = (4095 - find(white, rev)) / 4095.0 if white > 0.0 else 1.0
        if hi - lo >= 1e-3:
            out[i] = ((y2[i] - lo) / (hi - lo)).clamp(0.0, 1.0)
    return out


def shadows_highlights_photoshop(
    image: torch.Tensor,
    *,
    shadow_amount: float = 35.0,
    shadow_tone: float = 50.0,
    shadow_radius: float = 30.0,
    highlight_amount: float = 0.0,
    highlight_tone: float = 50.0,
    highlight_radius: float = 30.0,
    color: float = 20.0,
    midtone: float = 0.0,
    black_clip: float = 0.01,
    white_clip: float = 0.01,
) -> torch.Tensor:
    """Photoshop-style Shadows/Highlights on a ComfyUI IMAGE ``(B,H,W,3)`` in [0,1].

    Defaults are Photoshop's (Shadows 35 %, Tone 50 %, Radius 30 px, Color +20,
    Black/White Clip 0.01 %). Amounts are percentages 0..100; Color and Midtone
    -100..100; Radius in pixels.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"image must have shape (B,H,W,3), got {tuple(image.shape)}.")
    checks = (
        ("shadow_amount", shadow_amount, 0.0, 100.0),
        ("highlight_amount", highlight_amount, 0.0, 100.0),
        ("shadow_tone", shadow_tone, 0.0, 100.0),
        ("highlight_tone", highlight_tone, 0.0, 100.0),
        ("shadow_radius", shadow_radius, 0.0, 2500.0),
        ("highlight_radius", highlight_radius, 0.0, 2500.0),
        ("color", color, -100.0, 100.0),
        ("midtone", midtone, -100.0, 100.0),
        ("black_clip", black_clip, 0.0, 50.0),
        ("white_clip", white_clip, 0.0, 50.0),
    )
    for name, value, lower, upper in checks:
        if not lower <= float(value) <= upper:
            raise ValueError(f"{name} must be in [{lower}, {upper}], got {value}.")

    q = image.to(dtype=torch.float32).clamp(0.0, 1.0).permute(0, 3, 1, 2)
    lum = (q.amax(1, keepdim=True) + _luma(q)) / 2.0
    adj = q
    if shadow_amount > 0.0:
        neighbourhood = _neighbourhood(lum, shadow_radius)
        mask = _tonal_mask(shadow_tone, q)[_level(neighbourhood)]
        adj = adj + mask * (_lift(_strength(shadow_amount), adj) - adj)
    if highlight_amount > 0.0:
        if shadow_amount > 0.0 and abs(highlight_radius - shadow_radius) < 1e-3:
            bright = neighbourhood
        else:
            bright = _neighbourhood(lum, highlight_radius)
        mask = _tonal_mask(highlight_tone, q)[255 - _level(bright)]
        adj = adj - mask * (_lift(_strength(highlight_amount), 1.0 - q) - (1.0 - q))
    y2 = _clip_levels(_midtone(float(midtone), _luma(adj).clamp(0.0, 1.0)), black_clip, white_clip)

    y1 = _luma(q)
    f = (y2 - y1).abs() / torch.maximum(y2, y1).clamp_min(1e-6)
    if color >= 0.0:
        gain = torch.exp(_interp(_COLOR_UP, color) * f)
    else:
        gain = 1.0 + _interp(_COLOR_DOWN, -color) * f
    diff = (q - y1) * gain
    # Shorten the colour difference where it would leave 0..1.
    limit_up = torch.where(diff > 1e-9, (1.0 - y2) / diff.clamp_min(1e-9), torch.inf)
    limit_down = torch.where(diff < -1e-9, y2 / (-diff).clamp_min(1e-9), torch.inf)
    k = torch.minimum(limit_up, limit_down).amin(1, keepdim=True).clamp(0.0, 1.0)
    out = (y2 + k * diff).clamp(0.0, 1.0)
    return out.permute(0, 2, 3, 1)


def _level(v: torch.Tensor) -> torch.Tensor:
    return (v.clamp(0.0, 1.0) * 255.0).round().long()
