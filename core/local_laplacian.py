# Portions translated from PhotoCraft `crates/algo/src/camera_raw/local_tone.rs`
# (github.com/storytold/photocraft @ ead6d2197f5d0b82d70f3fddb70e472fc27ce577):
# the strength knots, thresholds and percentile rules below are theirs.
# Copyright (c) 2026 ArtCraft Team and the PhotoCraft contributors. MIT License,
# see THIRD_PARTY_NOTICES.md.
"""Camera Raw-style Shadows/Highlights as a fast local Laplacian filter.

Follows Paris, Hasinoff & Kautz 2011 "Local Laplacian Filters" in the fast form of
Aubry et al. 2014 (remapping sampled at reference values 1/8 stop apart, every
pyramid coefficient interpolated between them), run on log2 luminance:

- **Shadows** raises pixels darker than their surroundings by an amount set by
  their own and their surroundings' place in the image range (1st-99th
  percentile); the image's 99th percentile is then put back.
- **Highlights** pulls pixels brighter than their surroundings toward them past a
  small detail threshold; the 1st percentile is then put back.
- A flat image is unchanged, hard edges keep no halo, and the result does not
  depend on the image size: the filter runs on a proxy (long edge <= 1600 px)
  and its change is carried to full size by a fast guided upsampling.

The strengths were fitted by PhotoCraft to Photoshop 25.4's Camera Raw Filter on
ten photos (black-box measurement); treat the match as approximate.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as functional
from kornia.color import linear_rgb_to_rgb, rgb_to_linear_rgb

from .guided import guided_upsample, make_proxy

REF_STEP = 0.125  # stops between reference values; wider gaps draw contours on ramps
WORK_SIDE = 1600  # long edge of the proxy the filter runs on
MAX_REFS = 96
REF_CHUNK = 16  # references filtered together (memory ~ chunk x proxy pyramid)
LOG_FLOOR = 1.0 / 65536.0
_REC709 = (0.2126, 0.7152, 0.0722)
_TAPS = (1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0)

# Shadows: lift in stops per unit slider = own(pixel place) * around(surroundings
# place) * smoothstep(how far below the surroundings, full at `sigma` stops).
_SHADOW_OWN = (-0.017, 0.466, 0.886, 1.221, -0.081)
_SHADOW_AROUND = (4.566, 4.067, 1.088, 0.149, 0.878)
_SHADOW_SIGMA = 0.571
_SHADOW_DOWN = 0.808  # negative slider is this much weaker
# Highlights: strength over the surroundings' place, for lowering / raising.
_HIGHLIGHT_PULL = (0.179, 0.141, 0.196, 0.629, 2.07)
_HIGHLIGHT_PUSH = (0.156, 0.08, 0.175, 0.049, 1.261)
_HIGHLIGHT_SIGMA = 0.05

Remap = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _mirror_index(n: int, device: torch.device) -> torch.Tensor:
    """Indices ``-2 .. n+1`` mirrored into ``0..n-1`` without repeating the edge sample."""

    i = torch.arange(-2, n + 2, device=device)
    if n == 1:
        return torch.zeros_like(i)
    period = 2 * (n - 1)
    m = torch.remainder(i, period)
    return torch.where(m >= n, period - m, m)


def _blur(x: torch.Tensor, step: int) -> torch.Tensor:
    """Binomial 5-tap blur with mirrored borders, sampled every ``step`` pixels from 0."""

    taps = x.new_tensor(_TAPS)
    height, width = x.shape[-2:]
    rows = x.index_select(-2, _mirror_index(height, x.device))
    rows = functional.conv2d(rows, taps.view(1, 1, 5, 1))[..., ::step, :]
    cols = rows.index_select(-1, _mirror_index(width, x.device))
    return functional.conv2d(cols, taps.view(1, 1, 1, 5))[..., ::step]


def _down(x: torch.Tensor) -> torch.Tensor:
    return _blur(x, 2)


def _up(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Zero-stuff ``x`` to ``height x width`` and blur times four."""

    lh, lw = min(x.shape[-2], math.ceil(height / 2)), min(x.shape[-1], math.ceil(width / 2))
    stuffed = x.new_zeros((*x.shape[:-2], height, width))
    stuffed[..., 0 : 2 * lh : 2, 0 : 2 * lw : 2] = x[..., :lh, :lw] * 4.0
    return _blur(stuffed, 1)


def _depth(height: int, width: int) -> int:
    side, levels = min(height, width), 1
    while side > 2:
        side = math.ceil(side / 2)
        levels += 1
    return levels


def _gaussian(x: torch.Tensor, levels: int) -> list[torch.Tensor]:
    pyramid = [x]
    for _ in range(1, levels):
        pyramid.append(_down(pyramid[-1]))
    return pyramid


def _quantile(x: torch.Tensor, q: float) -> torch.Tensor:
    """Nearest-rank quantile (index ``round((n-1)q)`` of the sorted values)."""

    flat = x.flatten()
    k = int(round((flat.numel() - 1) * q))
    return torch.kthvalue(flat, k + 1).values


def _llf(u: torch.Tensor, remap: Remap) -> torch.Tensor:
    """Fast local Laplacian filter of ``u`` (1,1,h,w) with ``remap(value, reference)``."""

    height, width = u.shape[-2:]
    levels = _depth(height, width)
    g = _gaussian(u, levels)
    hi = u.max()
    lo = torch.minimum(_quantile(u, 0.001), hi)
    span = float(hi - lo)
    if not math.isfinite(span) or span <= 1e-6:
        return u.clone()
    refs = min(max(math.ceil(span / REF_STEP) + 1, 2), MAX_REFS)
    step = span / (refs - 1)
    laplacian = [torch.zeros_like(level) for level in g[:-1]]
    clamped = [level.clamp(float(lo), float(hi)) for level in g[:-1]]

    for start in range(0, refs, REF_CHUNK):
        ks = torch.arange(start, min(refs, start + REF_CHUNK), device=u.device, dtype=u.dtype)
        r = (lo + step * ks).view(-1, 1, 1, 1)
        remapped = _gaussian(remap(u.expand(len(ks), -1, -1, -1), r), levels)
        for level in range(levels - 1):
            fine = remapped[level]
            coarse = _up(remapped[level + 1], fine.shape[-2], fine.shape[-1])
            weight = (1.0 - (clamped[level] - r).abs() / step).clamp_min(0.0)
            laplacian[level] += (weight * (fine - coarse)).sum(0, keepdim=True)

    out = g[-1]
    for level in range(levels - 2, -1, -1):
        out = _up(out, g[level].shape[-2], g[level].shape[-1]) + laplacian[level]
    return out


def _range(u: torch.Tensor) -> tuple[float, float]:
    """1st and 99th percentiles of ``u`` block-averaged to at most 256 px across."""

    height, width = u.shape[-2:]
    f = max(1, math.ceil(max(height, width) / 256))
    pad = (0, (-width) % f, 0, (-height) % f)
    sums = functional.avg_pool2d(functional.pad(u, pad), f, f)
    counts = functional.avg_pool2d(functional.pad(torch.ones_like(u), pad), f, f)
    cells = (sums / counts.clamp_min(1e-12))[counts > 0]
    return float(_quantile(cells, 0.01)), float(_quantile(cells, 0.99))


def _knot(t: torch.Tensor, knots: tuple[float, ...]) -> torch.Tensor:
    """Piecewise-linear curve through ``knots`` evenly spaced over ``t`` in [0,1]."""

    table = t.new_tensor(knots)
    x = t.clamp(0.0, 1.0) * (len(knots) - 1)
    i = x.floor().clamp(max=len(knots) - 2).long()
    return table[i] + (table[i + 1] - table[i]) * (x - i)


def _excess(d: torch.Tensor, sigma: float) -> torch.Tensor:
    """``d`` past the detail threshold: ~0 below ``sigma``, ``d - sigma`` well above."""

    return d - sigma * torch.tanh(d / sigma)


def _highlights_pass(u: torch.Tensor, amount: float) -> torch.Tensor:
    lo, hi = _range(u)
    span = max(hi - lo, 1.0)
    knots = _HIGHLIGHT_PULL if amount < 0.0 else _HIGHLIGHT_PUSH

    def remap(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        strength = _knot((g - lo) / span, knots)
        return x + strength * amount * _excess((x - g).clamp_min(0.0), _HIGHLIGHT_SIGMA)

    out = _llf(u, remap)
    return out + (_quantile(u, 0.01) - _quantile(out, 0.01))


def _shadows_pass(u: torch.Tensor, amount: float) -> torch.Tensor:
    lo, hi = _range(u)
    span = max(hi - lo, 1.0)
    k = amount if amount > 0.0 else amount * _SHADOW_DOWN

    def remap(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        below = ((g - x) / _SHADOW_SIGMA).clamp(0.0, 1.0)
        weight = below * below * (3.0 - 2.0 * below)
        lift = _knot((x - lo) / span, _SHADOW_OWN) * _knot((g - lo) / span, _SHADOW_AROUND)
        return x + k * lift * weight

    out = _llf(u, remap)
    return out + (_quantile(u, 0.99) - _quantile(out, 0.99))


def log_luminance(linear_bchw: torch.Tensor) -> torch.Tensor:
    """log2 of Rec. 709 luminance of linear RGB (B,3,H,W) -> (B,1,H,W)."""

    weights = linear_bchw.new_tensor(_REC709).view(1, 3, 1, 1)
    luminance = (linear_bchw * weights).sum(1, keepdim=True)
    return torch.log2(luminance.clamp_min(0.0) + LOG_FLOOR)


def local_tone_change(u: torch.Tensor, shadows: float, highlights: float) -> torch.Tensor:
    """Change in log2 luminance (B,1,H,W) for Shadows/Highlights in -100..100."""

    height, width = u.shape[-2:]
    out = torch.zeros_like(u)
    if (shadows == 0.0 and highlights == 0.0) or height < 2 or width < 2:
        return out
    for i in range(u.shape[0]):
        full = u[i : i + 1]
        small = make_proxy(full, WORK_SIDE)
        o = small
        if highlights:
            o = _highlights_pass(o, max(-1.0, min(1.0, highlights / 100.0)))
        if shadows:
            o = _shadows_pass(o, max(-1.0, min(1.0, shadows / 100.0)))
        change = o - small
        if small.shape[-2:] != full.shape[-2:]:
            change = guided_upsample(full, small, change, radius=2, eps=0.01)
        out[i : i + 1] = change
    return out


def shadows_highlights_camera_raw(
    image: torch.Tensor, shadows: float = 0.0, highlights: float = 0.0
) -> torch.Tensor:
    """Camera Raw-style Shadows/Highlights on a ComfyUI IMAGE.

    Args:
        image: ``(B,H,W,3)`` sRGB float tensor in ``[0,1]``.
        shadows: -100..100; positive lifts shadows.
        highlights: -100..100; negative recovers highlights.

    Returns:
        ``(B,H,W,3)`` float32 sRGB tensor in ``[0,1]``.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"image must have shape (B,H,W,3), got {tuple(image.shape)}.")
    for name, value in (("shadows", shadows), ("highlights", highlights)):
        if not -100.0 <= float(value) <= 100.0:
            raise ValueError(f"{name} must be in [-100, 100], got {value}.")
    image = image.to(dtype=torch.float32).clamp(0.0, 1.0)
    if shadows == 0.0 and highlights == 0.0:
        return image

    linear = rgb_to_linear_rgb(image.permute(0, 3, 1, 2))
    change = local_tone_change(log_luminance(linear), float(shadows), float(highlights))
    out = linear_rgb_to_rgb((linear * torch.exp2(change)).clamp(0.0, 1.0))
    return out.permute(0, 2, 3, 1).clamp(0.0, 1.0)
