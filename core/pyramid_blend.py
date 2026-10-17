"""Multi-band Laplacian pyramid blending (Burt-Adelson) for ComfyUI IMAGE / MASK tensors."""

from __future__ import annotations

from numbers import Integral

import torch
import torch.nn.functional as nnf
from kornia.geometry.transform import pyrdown, pyrup

MIN_COARSEST_SIZE = 8


def _validate_image(name: str, image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (B,H,W,3), got {tuple(image.shape)}.")
    if not torch.is_floating_point(image):
        raise ValueError(f"{name} must be a floating point tensor, got {image.dtype}.")
    if image.shape[1] < 1 or image.shape[2] < 1:
        raise ValueError(f"{name} must have non-empty spatial dimensions.")
    return image.to(dtype=torch.float32).clamp(0.0, 1.0)


def _validate_mask(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor.")
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (B,H,W), got {tuple(mask.shape)}.")
    if not torch.is_floating_point(mask) and mask.dtype is not torch.bool:
        raise ValueError(f"mask must be float or bool, got {mask.dtype}.")
    return mask.to(dtype=torch.float32).clamp(0.0, 1.0)


def _validate_levels(levels: int) -> int:
    if isinstance(levels, bool) or not isinstance(levels, Integral):
        raise ValueError("levels must be an integer in [1, 10].")
    value = int(levels)
    if not 1 <= value <= 10:
        raise ValueError("levels must be in [1, 10].")
    return value


def effective_levels(height: int, width: int, levels: int) -> int:
    """Clamp ``levels`` so the coarsest pyramid level is at least 8 px on its short side."""

    short = min(height, width)
    effective = 1
    while effective < levels and short // (2**effective) >= MIN_COARSEST_SIZE:
        effective += 1
    return effective


def _batch_size(*batches: int) -> int:
    target = max(batches)
    if any(b not in (1, target) for b in batches):
        raise ValueError("image_a, image_b and mask batches must match or be 1.")
    return target


def _gaussian_pyramid(x: torch.Tensor, levels: int) -> list[torch.Tensor]:
    pyramid = [x]
    for _ in range(levels - 1):
        pyramid.append(pyrdown(pyramid[-1]))
    return pyramid


def _laplacian_pyramid(x: torch.Tensor, levels: int) -> list[torch.Tensor]:
    """Band-pass pyramid; last entry is the low-pass residual (coarsest Gaussian level)."""

    gaussian = _gaussian_pyramid(x, levels)
    bands = [fine - pyrup(coarse) for fine, coarse in zip(gaussian[:-1], gaussian[1:], strict=True)]
    bands.append(gaussian[-1])
    return bands


def pyramid_blend(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    mask: torch.Tensor,
    *,
    levels: int = 5,
) -> torch.Tensor:
    """Blend two images seamlessly with a Laplacian pyramid (Burt-Adelson).

    ``mask`` = 1 takes ``image_b``, ``mask`` = 0 keeps ``image_a``. Each frequency band is
    blended with a progressively blurrier mask, so fine detail switches sharply while low
    frequencies (color, exposure) cross-fade over a wide region.

    Any ``H, W`` works: the inputs are replicate-padded up to a multiple of
    ``2 ** (levels - 1)`` and cropped back. ``levels`` is clamped so the coarsest pyramid
    level is at least 8 px on its short side (tiny images therefore fall back to fewer
    levels, down to a plain alpha composite for ``levels=1``).

    Args:
        image_a: IMAGE tensor ``(B,H,W,3)`` float in ``[0,1]``.
        image_b: IMAGE tensor ``(B,H,W,3)``; same ``H, W`` as ``image_a``.
        mask: MASK tensor ``(B,H,W)`` float/bool in ``[0,1]``; same ``H, W``.
        levels: Requested pyramid depth, 1..10 (1 = plain alpha composite).

    Returns:
        Blended IMAGE ``(B,H,W,3)`` float32 in ``[0,1]`` on the input device. Batch is
        the largest input batch; batch-1 inputs broadcast.

    Raises:
        TypeError: If inputs are not tensors.
        ValueError: If shapes, dtypes, batches or ``levels`` are invalid.
    """

    a = _validate_image("image_a", image_a)
    b = _validate_image("image_b", image_b)
    m = _validate_mask(mask)
    levels = _validate_levels(levels)

    if not (a.shape[1:3] == b.shape[1:3] == m.shape[1:3]):
        raise ValueError(
            "image_a, image_b and mask spatial shapes must match, got "
            f"{tuple(a.shape[1:3])}, {tuple(b.shape[1:3])}, {tuple(m.shape[1:3])}."
        )
    batch = _batch_size(a.shape[0], b.shape[0], m.shape[0])
    height, width = a.shape[1], a.shape[2]
    device = a.device
    b = b.to(device)
    m = m.to(device)

    a = a.permute(0, 3, 1, 2).expand(batch, -1, -1, -1)
    b = b.permute(0, 3, 1, 2).expand(batch, -1, -1, -1)
    m = m.unsqueeze(1).expand(batch, -1, -1, -1)

    n = effective_levels(height, width, levels)
    if n == 1:
        out = a * (1.0 - m) + b * m
    else:
        multiple = 2 ** (n - 1)
        pad_h = (-height) % multiple
        pad_w = (-width) % multiple
        if pad_h or pad_w:
            a, b, m = (nnf.pad(t, (0, pad_w, 0, pad_h), mode="replicate") for t in (a, b, m))

        bands_a = _laplacian_pyramid(a.contiguous(), n)
        bands_b = _laplacian_pyramid(b.contiguous(), n)
        mask_pyr = _gaussian_pyramid(m.contiguous(), n)
        blended = [
            la * (1.0 - mk) + lb * mk for la, lb, mk in zip(bands_a, bands_b, mask_pyr, strict=True)
        ]

        out = blended[-1]
        for band in reversed(blended[:-1]):
            out = pyrup(out) + band
        out = out[:, :, :height, :width]

    return out.permute(0, 2, 3, 1).clamp(0.0, 1.0).to(dtype=torch.float32).contiguous()
