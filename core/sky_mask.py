"""Sky Mask: find the sky without a neural network.

Measured on a proxy (long edge <= 768 px):

1. Colour: blue sky (hue 170-260, some saturation), overcast or white cloud (bright, nearly
   grey; in a blue-sky photo only if not warmer than neutral, which keeps sunlit white
   walls out) and, at half weight, sunset glow (bright, warm, upper half) score as sky.
2. Smoothness: sky has little fine texture; the local standard deviation of luma must be
   low (clouds and gradients pass, leaves, buildings and grass do not).
3. Position: the sky is open to the top of the frame. Candidate regions are kept only if
   they touch the top rows, sit in the upper half and are large, or are large and clearly
   blue; a white wall low in the frame is dropped. Calm water joined to the sky at the
   horizon can be taken in (combine with a mask to exclude it).
4. Edges: the proxy mask is upsampled and refined with a colour guided filter on the full
   image, so it follows branches, hair-like twigs and roof lines with a soft edge.
"""

from __future__ import annotations

from numbers import Real

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

from .color_lab import rgb_to_hsv
from .guided import box_filter, guided_filter, make_proxy

PROXY_SIDE = 768
_TOP_BAND = 0.1  # rows that decide whether the photo has a blue sky
_WARM_WHITE = 0.05  # in a blue-sky photo, whites with red above blue by more are walls
_TOP_ROWS = 0.03  # a region touching this top fraction of rows is open to the sky
_UPPER_REGION = 0.5  # or: centroid above this height and area above _BIG_REGION
_BIG_REGION = 0.03
_BIG_BLUE = 0.08  # a clearly blue region this large is sky wherever it is
_TEXTURE_SCALE = 0.035  # local luma std at which a pixel stops looking smooth
_GUIDE_EPS = 1e-3


def _check(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"image must have shape (B,H,W,3), got {tuple(image.shape)}.")
    if not torch.is_floating_point(image):
        raise ValueError(f"image must be floating point, got {image.dtype}.")
    return torch.nan_to_num(image.float(), nan=0.0).clamp(0.0, 1.0)


def _band(x: torch.Tensor, lo: float, hi: float, soft: float) -> torch.Tensor:
    return ((x - lo) / soft + 0.5).clamp(0, 1) * ((hi - x) / soft + 0.5).clamp(0, 1)


def sky_likelihood(proxy: torch.Tensor, sensitivity: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pixel sky score and blue-sky score (h,w) in 0..1 for one (1,3,h,w) proxy."""
    rgb = proxy[0].permute(1, 2, 0)[None]
    hue, sat, val = (c[0, ..., 0] for c in rgb_to_hsv(rgb))
    luma = (proxy[0] * proxy.new_tensor([0.2126, 0.7152, 0.0722]).view(3, 1, 1)).sum(0)
    h = luma.shape[0]
    rows = torch.linspace(0, 1, h, device=proxy.device)[:, None].expand_as(luma)

    blue = _band(hue, 170.0, 260.0, 20.0) * (sat / 0.12).clamp(0, 1) * (val / 0.35).clamp(0, 1)
    white = (1 - sat / 0.25).clamp(0, 1) * ((luma - 0.45) / 0.2).clamp(0, 1)
    top = max(1, round(_TOP_BAND * h))
    if float(blue[:top].mean()) > 0.5:
        # Blue-sky photo: white near the sun is cool, white walls and stucco are warm.
        r, b = proxy[0, 0], proxy[0, 2]
        # (Sunlit cloud tops may be a little warm: only clearly warm whites are dropped.)
        white = white * ((b - r + _WARM_WHITE) / 0.03 + 0.5).clamp(0, 1)
    warm = (_band(hue, -30.0, 60.0, 20.0) + _band(hue, 300.0, 390.0, 20.0)).clamp(0, 1)
    # Sunset glow counts half: warm, bright and smooth also describes skin and walls, so it
    # only passes on its own at high sensitivity.
    sunset = 0.5 * warm * ((luma - 0.6) / 0.15).clamp(0, 1) * (1 - rows / 0.6).clamp(0, 1)
    colour = torch.maximum(torch.maximum(blue, white), sunset)
    return colour * _smoothness(luma, sensitivity), blue


def _smoothness(luma: torch.Tensor, sensitivity: float) -> torch.Tensor:
    x = luma[None, None]
    mean = box_filter(x, 3)
    std = (box_filter(x * x, 3) - mean * mean).clamp_min(0).sqrt()[0, 0]
    scale = _TEXTURE_SCALE * (0.5 + sensitivity)
    return (1 - (std - scale) / scale).clamp(0, 1)


def _connected_sky(candidate: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Keep candidate regions open to the top of the frame (uint8 0/1).

    Also kept: large regions in the upper half and, when little sky reaches the top
    (looking up through branches), large clearly blue regions anywhere.
    """
    h, w = candidate.shape
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    top = max(1, int(round(_TOP_ROWS * h)))
    touching = set(np.unique(labels[:top])) - {0}
    open_top = sum(stats[label, cv2.CC_STAT_AREA] for label in touching) / (h * w)
    blue_mean = np.bincount(labels.ravel(), weights=blue.ravel(), minlength=count) / np.maximum(
        stats[:, cv2.CC_STAT_AREA], 1
    )
    keep = np.zeros(count, dtype=bool)
    for label in range(1, count):
        big_upper = (
            centroids[label][1] < _UPPER_REGION * h
            and stats[label, cv2.CC_STAT_AREA] >= _BIG_REGION * h * w
        )
        big_blue = (
            open_top < _BIG_BLUE  # little sky reaches the top (looking up through branches)
            and stats[label, cv2.CC_STAT_AREA] >= _BIG_BLUE * h * w
            and blue_mean[label] > 0.5
        )
        keep[label] = label in touching or big_upper or big_blue
    return keep[labels].astype(np.uint8)


def sky_mask(
    image: torch.Tensor, *, sensitivity: float = 0.5, refine_edges: bool = True
) -> torch.Tensor:
    """Soft sky mask (B,H,W) in 0..1 for each image of a batch.

    Args:
        image: IMAGE ``(B,H,W,3)``.
        sensitivity: 0..1; higher accepts slightly textured or less typical sky (haze,
            thin clouds) at the risk of taking in smooth walls.
        refine_edges: Follow fine edges (branches, roof lines) at full resolution.
    """
    image = _check(image)
    if isinstance(sensitivity, bool) or not isinstance(sensitivity, Real):
        raise ValueError("sensitivity must be in [0, 1].")
    if not 0.0 <= float(sensitivity) <= 1.0:
        raise ValueError("sensitivity must be in [0, 1].")
    sensitivity = float(sensitivity)
    batch, height, width = image.shape[:3]
    out = torch.zeros(batch, height, width, device=image.device)
    for index in range(batch):
        full = image[index].permute(2, 0, 1)[None]
        proxy = make_proxy(full, PROXY_SIDE)
        score, blue = sky_likelihood(proxy, sensitivity)
        threshold = 0.6 - 0.3 * sensitivity
        candidate = (score > threshold).cpu().numpy().astype(np.uint8)
        # Close small gaps (birds, thin wires) before the connectivity test.
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        sky = _connected_sky(candidate, blue.cpu().numpy())
        sky = torch.from_numpy(sky).to(image.device).float()
        if not bool(sky.any()):
            continue
        soft = functional.interpolate(
            sky[None, None], size=(height, width), mode="bilinear", align_corners=False
        )
        if refine_edges:
            radius = max(2, round(0.004 * max(height, width)))
            soft = guided_filter(
                full,
                soft,
                radius=radius,
                eps=_GUIDE_EPS,
                subsample=max(1, radius // 2),
            )
        out[index] = soft[0, 0].clamp(0.0, 1.0)
    return out
