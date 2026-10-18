"""Dust and spot removal for ComfyUI IMAGE tensors.

Detection: luminance outliers against a median background computed on a reduced-resolution
copy, thresholded and filtered by connected-component extent (only blobs that fit inside
``spot_size`` pixels survive). Fill: normalised-convolution inpainting from the ring around
every spot, computed at the same reduced resolution; or (``fill_mode="texture"``) the
best-matching nearby patch, chosen by SSD on the ring and healed to the ring's brightness
and colour. Memory is O(H*W): the only window-based op (the median) runs on the reduced
copy in row chunks.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import cv2
import kornia
import numpy as np
import torch
import torch.nn.functional as nnf

_MODES = ("dark", "bright", "both")
_MEDIAN_WINDOW = 9
_MEDIAN_CHUNK_ELEMENTS = 16_000_000
_LUMA = (0.2126, 0.7152, 0.0722)
_MIN_AREA = 4
_MAX_ASPECT = 2.0
_MIN_FILL = 0.5
_MIN_THICKNESS = 1.3
_RING_RATIO = 0.25
_RING_FLOOR = 0.004
FILL_MODES = ("blur", "texture")
_SOURCE_DISTANCES = (1.5, 2.5, 4.0)  # candidate source offsets, in spot extents
_SOURCE_DIRECTIONS = 16
_SOURCE_OUTLIER = 0.03  # max deviation (plus 3 ring std) of a source pixel from its patch


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


def _validate_mask(
    name: str, mask: torch.Tensor | None, image: torch.Tensor
) -> torch.Tensor | None:
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if mask.ndim != 3:
        raise ValueError(f"{name} must have shape (B,H,W), got {tuple(mask.shape)}.")
    if not torch.is_floating_point(mask) and mask.dtype is not torch.bool:
        raise ValueError(f"{name} must be float or bool, got {mask.dtype}.")
    if mask.shape[1:] != image.shape[1:3]:
        raise ValueError(f"{name} and image spatial shapes must match.")
    if mask.shape[0] not in (1, image.shape[0]):
        raise ValueError(f"{name} batch must match image batch or be 1.")
    return mask.detach().to(device=image.device, dtype=torch.float32).clamp(0.0, 1.0)


def _validate_float(name: str, value: float, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    value_float = float(value)
    if not lower <= value_float <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_float


def _validate_int(name: str, value: int, *, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}].")
    value_int = int(value)
    if not lower <= value_int <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_int


def _luminance(image: torch.Tensor) -> torch.Tensor:
    """``(B,H,W,3)`` -> ``(B,1,H,W)``."""
    weights = torch.tensor(_LUMA, device=image.device, dtype=image.dtype)
    return (image * weights).sum(-1).unsqueeze(1)


def _reduce(x: torch.Tensor, factor: int) -> torch.Tensor:
    if factor == 1:
        return x
    height, width = x.shape[-2:]
    size = (max(1, math.ceil(height / factor)), max(1, math.ceil(width / factor)))
    return nnf.adaptive_avg_pool2d(x, size)


def _median_filter(x: torch.Tensor, window: int) -> torch.Tensor:
    """Median over a ``window x window`` neighbourhood of ``(B,1,h,w)``, in row chunks."""
    pad = window // 2
    padded = nnf.pad(x, (pad, pad, pad, pad), mode="replicate")
    height, width = x.shape[-2:]
    rows = max(1, _MEDIAN_CHUNK_ELEMENTS // (width * window * window))
    out = torch.empty_like(x)
    for start in range(0, height, rows):
        stop = min(height, start + rows)
        block = padded[:, :, start : stop + 2 * pad, :]
        patches = block.unfold(2, window, 1).unfold(3, window, 1)
        out[:, :, start:stop, :] = patches.reshape(*patches.shape[:4], -1).median(-1).values
    return out


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return nnf.max_pool2d(mask, 2 * radius + 1, stride=1, padding=radius)


def _component_mean(labels: np.ndarray, count: int, values: np.ndarray) -> np.ndarray:
    sums = np.bincount(labels.ravel(), weights=values.ravel(), minlength=count)
    sizes = np.bincount(labels.ravel(), minlength=count)
    return sums / np.maximum(sizes, 1)


def _shape_filter(frame: np.ndarray, max_extent: int) -> np.ndarray:
    """Keep compact, blob-like 8-connected components; drop thin, elongated or large ones."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(frame, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    if count <= 1:
        return keep[labels]
    widths = stats[:, cv2.CC_STAT_WIDTH]
    heights = stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]
    longest = np.maximum(widths, heights)
    shortest = np.maximum(np.minimum(widths, heights), 1)
    fill = area / np.maximum(widths * heights, 1)
    # Thickness: peak of the distance transform (a strand 1-2 px wide never reaches 1.3).
    dist = cv2.distanceTransform(frame, cv2.DIST_L2, 3)
    thickness = np.zeros(count, dtype=np.float32)
    np.maximum.at(thickness, labels.ravel(), dist.ravel())
    keep = (
        (longest <= max_extent)
        & (area >= _MIN_AREA)
        & (longest / shortest <= _MAX_ASPECT)
        & (fill >= _MIN_FILL)
        & (thickness >= _MIN_THICKNESS)
    )
    keep[0] = False
    return keep[labels]


def _calm_surroundings(
    luma: np.ndarray, diff: np.ndarray, candidates: np.ndarray, radius: int, ratio: float
) -> np.ndarray:
    """Drop candidates whose neighbourhood is busier than the spot contrast.

    The neighbourhood std is measured over a ``(2*radius+1)`` window with the candidates
    themselves excluded, so detail around the blob (eyelashes, hair, edges, texture) raises it
    while the blob's own contrast does not.
    """
    count, labels = cv2.connectedComponents(candidates, connectivity=8)
    if count <= 1:
        return candidates
    ksize = (2 * radius + 1, 2 * radius + 1)
    valid = (1 - candidates).astype(np.float32)
    kwargs = {"ksize": ksize, "normalize": False, "borderType": cv2.BORDER_REFLECT}
    weight = cv2.boxFilter(valid, -1, **kwargs)
    first = cv2.boxFilter(luma * valid, -1, **kwargs)
    second = cv2.boxFilter(luma * luma * valid, -1, **kwargs)
    mean = first / np.maximum(weight, 1.0)
    var = np.maximum(second / np.maximum(weight, 1.0) - mean * mean, 0.0)
    ring_std = np.sqrt(var)
    # Windows swallowed by candidates carry no information: treat them as busy.
    ring_std[weight < 0.25 * ksize[0] * ksize[1]] = 1.0
    ring = _component_mean(labels, count, ring_std)
    contrast = _component_mean(labels, count, diff)
    ok = ring <= ratio * contrast + _RING_FLOOR
    ok[0] = False
    return ok[labels].astype(np.uint8)


def _detect(
    luma: torch.Tensor, spot_size: int, threshold: float, mode: str, factor: int
) -> torch.Tensor:
    """Binary ``(B,1,H,W)`` map of compact luminance outliers inside calm surroundings."""
    height, width = luma.shape[-2:]
    background = _median_filter(_reduce(luma, factor), _MEDIAN_WINDOW)
    if factor > 1 or background.shape[-2:] != (height, width):
        background = nnf.interpolate(
            background, size=(height, width), mode="bilinear", align_corners=False
        )
    diff = background - luma
    if mode == "bright":
        diff = -diff
    elif mode == "both":
        diff = diff.abs()
    binary = (diff > threshold).squeeze(1).to(torch.uint8).cpu().numpy()
    diffs = diff.squeeze(1).cpu().numpy()
    lumas = luma.squeeze(1).cpu().numpy()
    radius = spot_size // 2 + 4
    out = np.zeros_like(binary)
    for index, frame in enumerate(binary):
        if not frame.any():
            continue
        shaped = _shape_filter(frame, spot_size).astype(np.uint8)
        if shaped.any():
            out[index] = _calm_surroundings(
                lumas[index].astype(np.float32),
                diffs[index].astype(np.float32),
                shaped,
                radius,
                _RING_RATIO,
            )
    return torch.from_numpy(out).to(device=luma.device, dtype=torch.float32).unsqueeze(1)


def _fill(image: torch.Tensor, spots: torch.Tensor, spot_size: int, factor: int) -> torch.Tensor:
    """Normalised-convolution inpaint of ``spots`` (B,1,H,W) in ``image`` (B,3,H,W)."""
    height, width = image.shape[-2:]
    weight = 1.0 - spots
    numerator = _reduce(image * weight, factor)
    denominator = _reduce(weight, factor)
    sigma = max(1.5, 0.75 * spot_size / factor)
    kernel = 2 * math.ceil(3.0 * sigma) + 1
    numerator = kornia.filters.gaussian_blur2d(
        numerator, (kernel, kernel), (sigma, sigma), border_type="replicate"
    )
    denominator = kornia.filters.gaussian_blur2d(
        denominator, (kernel, kernel), (sigma, sigma), border_type="replicate"
    )
    fill = numerator / denominator.clamp_min(1e-4)
    # Where the ring is missing entirely (spots cover the whole neighbourhood) keep the pixel.
    valid = (denominator > 1e-3).to(fill.dtype)
    fill = fill * valid + _reduce(image, factor) * (1.0 - valid)
    if fill.shape[-2:] != (height, width):
        fill = nnf.interpolate(fill, size=(height, width), mode="bilinear", align_corners=False)
    return fill


def _heal_component(
    image: np.ndarray,
    labels: np.ndarray,
    label: int,
    blocked: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray] | None:
    """Texture fill of one spot: copy the best nearby patch, matched on the ring around it.

    ``image`` (H,W,3) float32, ``labels`` (H,W) component labels (this spot is ``label``),
    ``blocked`` (H,W) bool for every spot pixel. Returns ``(rows_cols, values)`` for the spot
    pixels, or ``None`` when no clean source patch fits (the blur fill is kept).
    """
    height, width = blocked.shape
    x, y, w, h = box
    extent = max(w, h)
    ring_width = max(2, extent // 2)
    x0, y0 = max(0, x - ring_width), max(0, y - ring_width)
    x1, y1 = min(width, x + w + ring_width), min(height, y + h + ring_width)
    target_spot = labels[y0:y1, x0:x1] == label
    kernel = np.ones((2 * ring_width + 1, 2 * ring_width + 1), np.uint8)
    ring = (cv2.dilate(target_spot.astype(np.uint8), kernel) > 0) & ~blocked[y0:y1, x0:x1]
    if ring.sum() < 4:
        return None
    target = image[y0:y1, x0:x1]
    # A source patch must not carry its own (undetected) speck into the spot.
    outlier = 3.0 * float(target[ring].std(0).max()) + _SOURCE_OUTLIER
    best, best_cost = None, math.inf
    for distance in _SOURCE_DISTANCES:
        step = distance * (extent + ring_width)
        for k in range(_SOURCE_DIRECTIONS):
            angle = 2.0 * math.pi * k / _SOURCE_DIRECTIONS
            dx, dy = round(step * math.cos(angle)), round(step * math.sin(angle))
            sx0, sy0, sx1, sy1 = x0 + dx, y0 + dy, x1 + dx, y1 + dy
            if sx0 < 0 or sy0 < 0 or sx1 > width or sy1 > height:
                continue
            if blocked[sy0:sy1, sx0:sx1][target_spot | ring].any():
                continue
            source = image[sy0:sy1, sx0:sx1]
            patch = source[target_spot]
            if float(np.abs(patch - patch.mean(0)).max()) > outlier:
                continue
            cost = float(((source[ring] - target[ring]) ** 2).mean())
            if cost < best_cost:
                best, best_cost = source, cost
    if best is None:
        return None
    # Heal: the patch keeps its texture, its brightness/colour follows the ring
    # (normalised-convolution spread of target - source from the ring into the spot).
    sigma = max(1.0, 0.5 * extent)
    weight = ring.astype(np.float32)
    diff = (target - best) * weight[..., None]
    spread = cv2.GaussianBlur(diff, (0, 0), sigma)
    norm = cv2.GaussianBlur(weight, (0, 0), sigma)[..., None]
    healed = best + spread / np.maximum(norm, 1e-4)
    rows, cols = np.nonzero(target_spot)
    return (rows + y0, cols + x0), healed[rows, cols]


def _texture_fill(image: torch.Tensor, spots: torch.Tensor, fill: torch.Tensor) -> torch.Tensor:
    """Replace the blur ``fill`` inside each spot with a healed nearby patch (B,3,H,W)."""
    pixels = image.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    out = fill.permute(0, 2, 3, 1).cpu().numpy().copy()
    masks = spots.squeeze(1).cpu().numpy() > 0.5
    for index in range(pixels.shape[0]):
        blocked = masks[index]
        if not blocked.any():
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            blocked.astype(np.uint8), connectivity=8
        )
        for label in range(1, count):
            box = tuple(int(v) for v in stats[label, :4])
            healed = _heal_component(pixels[index], labels, label, blocked, box)
            if healed is not None:
                (rows, cols), values = healed
                out[index, rows, cols] = values
    return torch.from_numpy(out).to(device=fill.device).permute(0, 3, 1, 2)


def dust_spot_removal(
    image: torch.Tensor,
    mask: torch.Tensor | None = None,
    protect_mask: torch.Tensor | None = None,
    *,
    spot_size: int = 12,
    sensitivity: float = 0.5,
    mode: str = "dark",
    fill_mode: str = "blur",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove small dust specks and blemishes by filling them from their surroundings.

    A spot is a small blob of pixels whose luminance differs from the local median
    background by more than a threshold derived from ``sensitivity`` and whose bounding box
    fits inside ``spot_size`` pixels. Larger or longer structures (hair, eyes, edges) are left
    alone. Detected spots are dilated slightly and replaced by a smooth estimate computed from
    the ring of pixels around them.

    Args:
        image: IMAGE tensor ``(B,H,W,3)`` float in ``[0,1]``.
        mask: Optional MASK ``(B,H,W)``; spots are only removed where mask > 0.5.
        protect_mask: Optional MASK ``(B,H,W)``; spots are never removed where it is > 0.5.
        spot_size: Maximum spot diameter in pixels, 2..64.
        sensitivity: 0..1; higher catches fainter spots (luma threshold
            ``0.01 + 0.2 * (1 - sensitivity) ** 2``).
        mode: ``"dark"``, ``"bright"`` or ``"both"``.
        fill_mode: ``"blur"`` (smooth estimate from the ring, the original behaviour) or
            ``"texture"`` (copy the best-matching nearby patch, matched on the ring, and
            heal its brightness/colour to the surroundings, so skin pores, film grain or
            fabric continue through the spot). Falls back to blur where no clean patch fits.

    Returns:
        ``(image, spot_mask)``: cleaned image ``(B,H,W,3)`` and the binary (0/1) mask
        ``(B,H,W)`` of replaced pixels. Pixels where ``spot_mask`` is 0 are unchanged.

    Raises:
        TypeError: If inputs are not tensors.
        ValueError: If shapes, dtypes or parameters are invalid.
    """
    prepared = _validate_image(image)
    allow = _validate_mask("mask", mask, prepared)
    protect = _validate_mask("protect_mask", protect_mask, prepared)
    spot_size = _validate_int("spot_size", spot_size, lower=2, upper=64)
    sensitivity = _validate_float("sensitivity", sensitivity, lower=0.0, upper=1.0)
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}.")
    if fill_mode not in FILL_MODES:
        raise ValueError(f"fill_mode must be one of {FILL_MODES}, got {fill_mode!r}.")

    threshold = 0.01 + 0.2 * (1.0 - sensitivity) ** 2
    factor = max(2, math.ceil(spot_size / 4))
    channels = prepared.permute(0, 3, 1, 2).contiguous()
    spots = _detect(_luminance(prepared), spot_size, threshold, mode, factor)
    if allow is not None:
        spots = spots * (allow.unsqueeze(1) > 0.5).to(spots.dtype)
    if protect is not None:
        spots = spots * (protect.unsqueeze(1) <= 0.5).to(spots.dtype)
    spots = _dilate(spots, max(1, spot_size // 6))
    if allow is not None:
        spots = spots * (allow.unsqueeze(1) > 0.5).to(spots.dtype)
    if protect is not None:
        spots = spots * (protect.unsqueeze(1) <= 0.5).to(spots.dtype)

    if not bool(spots.any()):
        return prepared, spots.squeeze(1)

    fill = _fill(channels, spots, spot_size, factor)
    if fill_mode == "texture":
        fill = _texture_fill(channels, spots, fill)
    kernel = (5, 5)
    soft = kornia.filters.gaussian_blur2d(spots, kernel, (1.0, 1.0), border_type="replicate")
    soft = soft * spots
    cleaned = channels * (1.0 - soft) + fill * soft
    return cleaned.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0), spots.squeeze(1)
