"""Luminance-based skin-tone region masks."""

from __future__ import annotations

import math
from numbers import Real

import torch
import torch.nn.functional as functional


def _validate_cutoffs(shadow_cutoff: float, highlight_cutoff: float) -> tuple[float, float]:
    if not isinstance(shadow_cutoff, Real) or not isinstance(highlight_cutoff, Real):
        raise ValueError("shadow_cutoff and highlight_cutoff must be numeric.")
    shadow = float(shadow_cutoff)
    highlight = float(highlight_cutoff)
    if not 0.0 <= shadow < highlight <= 1.0:
        raise ValueError("cutoffs must satisfy 0.0 <= shadow_cutoff < highlight_cutoff <= 1.0.")
    if shadow > 0.5:
        raise ValueError("shadow_cutoff must be <= 0.5.")
    if highlight < 0.5:
        raise ValueError("highlight_cutoff must be >= 0.5.")
    return shadow, highlight


def _prepare_skin_mask(
    skin_mask: torch.Tensor | None,
    *,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if skin_mask is None:
        return torch.ones((batch, height, width), device=device, dtype=dtype)

    if not isinstance(skin_mask, torch.Tensor):
        raise TypeError("skin_mask must be a torch.Tensor or None.")
    if skin_mask.ndim == 4 and skin_mask.shape[1] == 1:
        prepared = skin_mask[:, 0]
    elif skin_mask.ndim == 3:
        prepared = skin_mask
    else:
        raise ValueError(
            f"skin_mask must have shape (B,H,W) or (B,1,H,W), got {tuple(skin_mask.shape)}."
        )
    if prepared.shape[-2:] != (height, width):
        raise ValueError(
            f"skin_mask HxW {tuple(prepared.shape[-2:])} must match image HxW {(height, width)}."
        )
    if prepared.shape[0] not in (1, batch):
        raise ValueError(f"skin_mask batch must be 1 or image batch {batch}.")
    if not torch.is_floating_point(prepared) and prepared.dtype is not torch.bool:
        raise ValueError(f"skin_mask must be float or bool, got {prepared.dtype}.")

    prepared = prepared.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    if prepared.shape[0] == 1 and batch > 1:
        prepared = prepared.expand(batch, -1, -1)
    return prepared


def _gaussian_kernel_1d(sigma: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _blur_masks(masks: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return masks
    batch_channels = masks.shape[0] * masks.shape[1]
    flat = masks.reshape(batch_channels, 1, masks.shape[-2], masks.shape[-1])
    kernel_1d = _gaussian_kernel_1d(sigma, device=masks.device, dtype=masks.dtype)
    pad = kernel_1d.numel() // 2
    horizontal = kernel_1d.view(1, 1, 1, -1)
    vertical = kernel_1d.view(1, 1, -1, 1)
    flat = functional.pad(flat, (pad, pad, 0, 0), mode="replicate")
    flat = functional.conv2d(flat, horizontal)
    flat = functional.pad(flat, (0, 0, pad, pad), mode="replicate")
    flat = functional.conv2d(flat, vertical)
    return flat.reshape_as(masks)


def skin_tone_tri_region(
    image_bchw: torch.Tensor,
    *,
    skin_mask: torch.Tensor | None = None,
    shadow_cutoff: float = 0.33,
    highlight_cutoff: float = 0.66,
    soft_sigma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a skin area into shadow, midtone, and highlight masks.

    The three returned masks always sum back to the supplied ``skin_mask`` (or
    ones when omitted) after internal Gaussian smoothing and normalization.
    """
    if not isinstance(image_bchw, torch.Tensor):
        raise TypeError("image_bchw must be a torch.Tensor.")
    if image_bchw.ndim != 4 or image_bchw.shape[1] != 3:
        raise ValueError(f"image_bchw must have shape (B,3,H,W), got {tuple(image_bchw.shape)}.")
    if not torch.is_floating_point(image_bchw):
        raise ValueError(f"image_bchw must be floating point, got {image_bchw.dtype}.")
    if not isinstance(soft_sigma, Real) or float(soft_sigma) < 0.0:
        raise ValueError("soft_sigma must be a non-negative number.")

    shadow_cutoff, highlight_cutoff = _validate_cutoffs(shadow_cutoff, highlight_cutoff)
    image = image_bchw.clamp(0.0, 1.0)
    batch, _, height, width = image.shape
    skin = _prepare_skin_mask(
        skin_mask,
        batch=batch,
        height=height,
        width=width,
        device=image.device,
        dtype=image.dtype,
    )

    luminance = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
    shadow = (luminance < shadow_cutoff).to(image.dtype) * skin
    highlight = (luminance > highlight_cutoff).to(image.dtype) * skin
    midtone = ((luminance >= shadow_cutoff) & (luminance <= highlight_cutoff)).to(
        image.dtype
    ) * skin

    regions = torch.stack([shadow, midtone, highlight], dim=1)
    regions = _blur_masks(regions, float(soft_sigma)).clamp_min(0.0)
    total = regions.sum(dim=1, keepdim=True).clamp_min(torch.finfo(image.dtype).eps)
    regions = regions / total * skin.unsqueeze(1)
    return regions[:, 0], regions[:, 1], regions[:, 2]
