"""Decontaminate colours: remove the background colour that bleeds into a cut-out's edges.

Edge pixels of a matte (``0 < alpha < 1``) mix foreground and background colours
(``I = alpha * F + (1 - alpha) * B``), so green-screen spill or a bright sky shows as a
coloured fringe once the subject is placed on a new background. Both colours are estimated
by normalised convolution from nearby solid pixels: ``F_blur`` from ``alpha`` close to 1,
``B`` from ``alpha`` close to 0 (``blur(w * RGB) / blur(w)``). Where ``alpha`` is large the
compositing equation is solved for ``F = (I - (1 - alpha) * B) / alpha``, which keeps fine
detail such as hair strands; where ``alpha`` is small (unstable solve) ``F_blur`` is used.
Solid foreground (``alpha = 1``) and background (``alpha = 0``) are left untouched. Where
no solid pixels lie within the radius, wider blurs fill in.

Memory is O(H*W): the blurs are three box passes (``core.guided.gaussian_blur_box``) on a
reduced copy, upsampled bilinearly.
"""

from __future__ import annotations

from numbers import Integral, Real

import torch
import torch.nn.functional as functional

from .guided import gaussian_blur_box

_SOLID = 0.95  # alpha above which a pixel counts as solid foreground
_CLEAR = 0.05  # alpha below which a pixel counts as pure background
_SOLVE_FROM, _SOLVE_TO = 0.2, 0.6  # alpha range blending F_blur into the exact solve
_MAX_PASSES = 3  # radius, 4x radius, 16x radius


def _validate(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(image, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("image and mask must be torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"image must have shape (B,H,W,3), got {tuple(image.shape)}.")
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (B,H,W), got {tuple(mask.shape)}.")
    if mask.shape[1:] != image.shape[1:3]:
        raise ValueError(
            f"mask spatial shape {tuple(mask.shape[1:])} must match image "
            f"{tuple(image.shape[1:3])}."
        )
    if mask.shape[0] not in (1, image.shape[0]):
        raise ValueError(f"mask batch must be 1 or {image.shape[0]}, got {mask.shape[0]}.")
    if not torch.is_floating_point(image):
        raise ValueError(f"image must be floating point, got {image.dtype}.")
    image = image.to(torch.float32).clamp(0.0, 1.0)
    mask = mask.to(device=image.device, dtype=torch.float32).clamp(0.0, 1.0)
    if mask.shape[0] != image.shape[0]:
        mask = mask.expand(image.shape[0], -1, -1)
    return image, mask


def _check_range(name: str, value: float, lower: float, upper: float, integer: bool) -> None:
    kind = Integral if integer else Real
    if isinstance(value, bool) or not isinstance(value, kind) or not lower <= value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")


def _blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur of ``sigma`` full-res pixels, run on a copy reduced to ~sigma/4."""
    height, width = x.shape[-2:]
    factor = max(1, min(int(sigma // 4), height // 4 or 1, width // 4 or 1))
    small = x if factor == 1 else functional.avg_pool2d(x, factor, ceil_mode=True)
    small = gaussian_blur_box(small, sigma / factor)
    if small.shape[-2:] != (height, width):
        small = functional.interpolate(
            small, size=(height, width), mode="bilinear", align_corners=False
        )
    return small


def _spread(
    rgb: torch.Tensor, weight: torch.Tensor, edge: torch.Tensor, sigma: float
) -> torch.Tensor | None:
    """Mean colour of the ``weight`` pixels around each pixel (normalised convolution).

    Widens the blur up to ``_MAX_PASSES`` times until every edge pixel has a value;
    pixels still out of reach keep their own colour. ``None`` if ``weight`` is empty.
    """
    if not bool(weight.any()):
        return None
    estimate = rgb.clone()
    missing = torch.ones_like(weight, dtype=torch.bool)
    for _ in range(_MAX_PASSES):
        den = _blur(weight, sigma)
        num = _blur(rgb * weight, sigma)
        found = missing & (den > 1e-3)
        estimate = torch.where(found, num / den.clamp_min(1e-3), estimate)
        missing = missing & ~found
        if not bool((missing & edge).any()):
            break
        sigma *= 4.0
    return estimate


def decontaminate_colors(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    radius: int = 16,
    strength: float = 1.0,
) -> torch.Tensor:
    """Replace edge colours of a cut-out with the nearby foreground colour.

    Args:
        image: IMAGE ``(B,H,W,3)`` in ``[0,1]``.
        mask: MASK ``(B,H,W)`` (or batch 1) in ``[0,1]``; 1 = subject.
        radius: How far (pixels) to look for solid foreground colour, 1..256.
        strength: 0..1; 0 returns the image unchanged.

    Returns:
        ``(B,H,W,3)`` float32 image. Pixels with ``alpha`` 0 or 1 are unchanged.
    """
    image, mask = _validate(image, mask)
    _check_range("radius", radius, 1, 256, integer=True)
    _check_range("strength", strength, 0.0, 1.0, integer=False)
    if strength == 0.0:
        return image

    alpha = mask.unsqueeze(1)
    edge = (alpha > 0.0) & (alpha < 1.0)
    if not bool(edge.any()):
        return image

    rgb = image.permute(0, 3, 1, 2)
    fg = _spread(rgb, (alpha >= _SOLID).to(rgb.dtype), edge, float(radius) / 2.0)
    bg = _spread(rgb, (alpha <= _CLEAR).to(rgb.dtype), edge, float(radius) / 2.0)

    if fg is None and bg is None:
        return image
    # Without solid foreground nearby the pixel's own colour is the best blur estimate.
    estimate = rgb if fg is None else fg
    trust = ((alpha - _SOLVE_FROM) / (_SOLVE_TO - _SOLVE_FROM)).clamp(0.0, 1.0)
    # Mostly-subject pixels: the exact solve keeps detail; without a background estimate
    # the pixel's own colour is safer than a blurred average.
    solved = rgb if bg is None else (rgb - (1.0 - alpha) * bg) / alpha.clamp_min(1e-3)
    estimate = estimate + (solved - estimate) * trust

    # Fade in over the first few percent of alpha: a faint mask tail (left by blurring or
    # resizing a mask) must not repaint the background, which stays visible in the IMAGE.
    amount = strength * edge.to(rgb.dtype) * (alpha / _CLEAR).clamp(max=1.0)
    out = rgb + (estimate.clamp(0.0, 1.0) - rgb) * amount
    return out.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)
