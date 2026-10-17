"""Dominant-colour palette extraction (k-means in CIE LAB) for ComfyUI IMAGE tensors."""

from __future__ import annotations

import logging
from numbers import Integral

import kornia
import torch

_LOG = logging.getLogger(__name__)

STRIP_WIDTH = 1024
STRIP_HEIGHT = 192
_MAX_ITERATIONS = 25
_SHIFT_TOLERANCE = 0.05  # LAB units; stop when no centre moves more than this
_ASSIGN_CHUNK = 1 << 20


def _validate_image(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"image must have shape (B,H,W,3), got {tuple(image.shape)}.")
    if not torch.is_floating_point(image):
        raise ValueError(f"image must be a floating point tensor, got {image.dtype}.")
    if image.shape[0] < 1 or image.shape[1] < 1 or image.shape[2] < 1:
        raise ValueError("image must not be empty.")
    return image.detach().to(dtype=torch.float32).clamp(0.0, 1.0)


def _validate_mask(mask: torch.Tensor | None, image: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor.")
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (B,H,W), got {tuple(mask.shape)}.")
    if not torch.is_floating_point(mask) and mask.dtype is not torch.bool:
        raise ValueError(f"mask must be float or bool, got {mask.dtype}.")
    if mask.shape[1:] != image.shape[1:3]:
        raise ValueError("mask and image spatial shapes must match.")
    if mask.shape[0] not in (1, image.shape[0]):
        raise ValueError("mask batch must match image batch or be 1.")
    return mask.detach().to(device=image.device, dtype=torch.float32)


def _validate_int(name: str, value: int, *, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}].")
    value_int = int(value)
    if not lower <= value_int <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_int


def _kmeans(x: torch.Tensor, k: int, generator: torch.Generator) -> torch.Tensor:
    """Seeded k-means++ followed by Lloyd iterations on ``(n,3)`` points; returns ``(k,3)``."""
    n = x.shape[0]
    first = int(torch.randint(0, n, (1,), generator=generator))
    centers = [x[first]]
    min_d2 = ((x - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        weights = min_d2.cpu()
        if float(weights.sum()) <= 1e-12:  # fewer distinct colours than clusters
            pick = int(torch.randint(0, n, (1,), generator=generator))
        else:
            pick = int(torch.multinomial(weights, 1, generator=generator))
        centers.append(x[pick])
        min_d2 = torch.minimum(min_d2, ((x - x[pick]) ** 2).sum(1))
    center = torch.stack(centers)
    for _ in range(_MAX_ITERATIONS):
        labels = torch.cdist(x, center).argmin(1)
        sums = torch.zeros_like(center).index_add_(0, labels, x)
        counts = torch.bincount(labels, minlength=k).to(x.dtype).unsqueeze(1)
        updated = torch.where(counts > 0, sums / counts.clamp_min(1.0), center)
        shift = (updated - center).abs().max()
        center = updated
        if float(shift) < _SHIFT_TOLERANCE:
            break
    return center


def _assign(lab: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    out = torch.empty(lab.shape[0], dtype=torch.long, device=lab.device)
    for start in range(0, lab.shape[0], _ASSIGN_CHUNK):
        chunk = lab[start : start + _ASSIGN_CHUNK]
        out[start : start + _ASSIGN_CHUNK] = torch.cdist(chunk, centers).argmin(1)
    return out


def _extract_one(
    image: torch.Tensor,
    mask: torch.Tensor | None,
    colors: int,
    max_samples: int,
    seed: int,
    allow_empty: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (palette_rgb_uint ``(k,3)`` sorted by share, shares ``(k,)``, labels ``(H*W,)``)."""
    lab = kornia.color.rgb_to_lab(image.permute(2, 0, 1).unsqueeze(0))[0].reshape(3, -1).T
    lab = lab.contiguous()
    if mask is None:
        selected = None
        n_selected = lab.shape[0]
    else:
        selected = (mask.reshape(-1) > 0.5).nonzero().squeeze(1)
        n_selected = int(selected.numel())
        if n_selected == 0:
            if not allow_empty:
                raise ValueError("mask selects no pixels.")
            _LOG.warning("palette_extract: empty mask for a batch item; using the full image.")
            selected = None
            mask = None
            n_selected = lab.shape[0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if n_selected > max_samples:
        pick = torch.randperm(n_selected, generator=generator)[:max_samples].to(lab.device)
        sample_idx = pick if selected is None else selected[pick]
        sample = lab[sample_idx]
    else:
        sample = lab if selected is None else lab[selected]
    centers = _kmeans(sample, colors, generator)

    labels = _assign(lab, centers)
    counted = labels if selected is None else labels[selected]
    counts = torch.bincount(counted, minlength=colors).to(torch.float32)
    order = torch.sort(counts, descending=True, stable=True).indices
    rgb = kornia.color.lab_to_rgb(centers[order].T.reshape(1, 3, colors, 1))[0, :, :, 0].T
    palette = (rgb.clamp(0.0, 1.0) * 255.0).round()
    rank = torch.empty_like(order)
    rank[order] = torch.arange(colors, device=order.device)
    shares = counts[order] / counts.sum()
    return palette, shares, rank[labels]


def palette_extract(
    image: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    colors: int = 6,
    max_samples: int = 200_000,
    seed: int = 0,
) -> tuple[torch.Tensor, list[list[dict]], torch.Tensor]:
    """Find the dominant colours of an image with k-means in CIE LAB.

    Clustering runs on a seeded random subsample (at most ``max_samples`` pixels) so the
    result is deterministic; shares and the quantised image use every pixel.

    Args:
        image: IMAGE tensor ``(B,H,W,3)`` float in ``[0,1]``.
        mask: Optional MASK ``(B,H,W)`` (batch 1 broadcasts). Only pixels with value > 0.5
            are used to find the colours and to count shares; the quantised image still
            covers the whole frame.
        colors: Number of palette colours, 2..16.
        max_samples: Subsample size for clustering (>= 1000).
        seed: Seed for subsampling and k-means++ initialisation.

    Returns:
        ``(strip, palettes, quantized)``. ``strip`` is a ``(B,192,1024,3)`` swatch image
        with ``colors`` equal-width bands, largest share first. ``palettes`` is one list per
        batch item of ``{"hex", "rgb", "share"}`` dicts sorted by share (largest first).
        ``quantized`` is the image with every pixel replaced by its nearest palette colour.

    Raises:
        TypeError: If inputs are not tensors.
        ValueError: If shapes, dtypes or parameters are invalid, or the mask selects no pixel
            for any batch item. If only some items have an empty mask, those items use the
            full image and a warning is logged.
    """
    prepared = _validate_image(image)
    prepared_mask = _validate_mask(mask, prepared)
    colors = _validate_int("colors", colors, lower=2, upper=16)
    max_samples = _validate_int("max_samples", max_samples, lower=1000, upper=10_000_000)
    seed = _validate_int("seed", seed, lower=0, upper=2**31 - 1)

    batch, height, width, _ = prepared.shape
    column_band = (torch.arange(STRIP_WIDTH, device=prepared.device) * colors) // STRIP_WIDTH
    strips, quantized, palettes = [], [], []
    # An empty mask on one item falls back to the full image; only an all-empty mask is an error.
    allow_empty = False
    if prepared_mask is not None and batch > 1:
        used = prepared_mask if prepared_mask.shape[0] > 1 else prepared_mask.expand(batch, -1, -1)
        allow_empty = bool(((used > 0.5).flatten(1).any(1)).any())
    for index in range(batch):
        item_mask = None
        if prepared_mask is not None:
            item_mask = prepared_mask[index if prepared_mask.shape[0] > 1 else 0]
        palette, shares, labels = _extract_one(
            prepared[index], item_mask, colors, max_samples, seed, allow_empty
        )
        unit = palette / 255.0
        strips.append(unit[column_band].unsqueeze(0).expand(STRIP_HEIGHT, -1, -1))
        quantized.append(unit[labels].reshape(height, width, 3))
        rgb_list = palette.to(torch.int64).cpu().tolist()
        share_list = shares.cpu().tolist()
        palettes.append(
            [
                {
                    "hex": "#{:02x}{:02x}{:02x}".format(*rgb),
                    "rgb": rgb,
                    "share": round(share, 4),
                }
                for rgb, share in zip(rgb_list, share_list, strict=True)
            ]
        )
    return torch.stack(strips), palettes, torch.stack(quantized)
