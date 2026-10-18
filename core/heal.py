"""Heal / Content-Aware Fill: fill a masked region from the rest of the photo.

``content_aware``: exemplar-based completion (Wexler, Shechtman & Irani 2007) with a
parallel PatchMatch search (Barnes et al. 2009). The hole is filled coarse to fine over an
image pyramid. At each level every pixel whose 7x7 patch touches the hole looks for the
most similar fully known patch nearby (propagation from neighbours at 8/4/2/1 px jumps,
plus random search at shrinking radii). Each hole pixel then becomes the weighted vote of
the overlapping source patches, so texture and structure continue into the hole. Only
the region around each part of the mask (its bounding box plus a margin) is processed;
parts far apart are healed separately. The feather band is synthesised too and blended
from opaque at the mask edge to the original at the outer edge of the band.

``smooth``: normalised-convolution fill from the surrounding colours (no texture), like
the blur fill of Dust & Spot; fast and safe for flat areas.

Memory is O(region pixels): patch distances are accumulated offset by offset with
gathers, never as an unfolded (49 x C x N) tensor.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

from .decontaminate import _spread

MODES = ("content_aware", "smooth")
_PATCH_RADIUS = 3  # 7x7 patches
_COARSE_EXTENT = 24  # hole size (px) at the coarsest level
_MIN_SIDE = 32  # smallest pyramid level side
_ITERS_COARSE, _ITERS_FINE, _ITERS_FULL = 15, 3, 2
_CONFIDENCE = 1.3  # vote weight falls by this factor per pixel of depth into the hole
_CHUNK = 65536  # targets per gather (49 offsets x 3 channels x 4 B each)
_BRUTE_PAIRS = 4_000_000  # target x source pairs per brute-force block (~16 MB)
_BRUTE_LIMIT = 400_000_000  # skip brute force above this many pairs
_JUMPS = (8, 4, 2, 1)
_JUMPS_FINE = (2, 1)
_RANDOM_TRIES = 4
_RANDOM_TRIES_FINE = 5
_FINE_RADIUS = 16
_MAX_DEPTH = 40.0  # px; deeper hole pixels share the smallest confidence
_CLUSTER_GAP = 24  # mask parts closer than this are healed together
_SEED = 20261018


def _check(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(image, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("image and mask must be torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"image must have shape (B,H,W,3), got {tuple(image.shape)}.")
    if not torch.is_floating_point(image):
        raise ValueError(f"image must be floating point, got {image.dtype}.")
    if mask.ndim != 3 or mask.shape[1:] != image.shape[1:3]:
        raise ValueError("mask must have shape (B,H,W) matching the image.")
    if mask.shape[0] not in (1, image.shape[0]):
        raise ValueError(f"mask batch must be 1 or {image.shape[0]}, got {mask.shape[0]}.")
    image = torch.nan_to_num(image.float(), nan=0.0).clamp(0.0, 1.0)
    mask = mask.to(device=image.device, dtype=torch.float32)
    if mask.shape[0] != image.shape[0]:
        mask = mask.expand(image.shape[0], -1, -1)
    return image, mask


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation of a (h,w) bool mask with a square window."""
    if radius <= 0:
        return mask
    x = mask[None, None].float()
    size = 2 * radius + 1  # separable square max filter
    x = functional.max_pool2d(x, (1, size), stride=1, padding=(0, radius))
    return functional.max_pool2d(x, (size, 1), stride=1, padding=(radius, 0))[0, 0] > 0.5


def _region(hole: torch.Tensor, margin_scale: float) -> tuple[int, int, int, int]:
    """Bounding box of the hole plus a context margin, clipped to the image."""
    ys, xs = torch.nonzero(hole, as_tuple=True)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    extent = max(y1 - y0, x1 - x0)
    margin = max(48, round(margin_scale * extent))
    h, w = hole.shape
    return max(0, y0 - margin), min(h, y1 + margin), max(0, x0 - margin), min(w, x1 + margin)


def _downsample(img: torch.Tensor, hole: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Half size: image by area average of known pixels, hole if any child is hole."""
    known = (~hole).float()[None, None]
    num = functional.avg_pool2d(img[None] * known, 2, ceil_mode=True)
    den = functional.avg_pool2d(known, 2, ceil_mode=True)
    small = (num / den.clamp_min(1e-6))[0]
    small_hole = functional.max_pool2d(hole.float()[None, None], 2, ceil_mode=True)[0, 0] > 0.5
    return small, small_hole


class _Level:
    """One pyramid level: image estimate, hole, valid source centres and targets."""

    def __init__(self, img: torch.Tensor, hole: torch.Tensor, sparse: bool = False) -> None:
        r = _PATCH_RADIUS
        self.img = img.clone()
        self.hole = hole
        _, self.h, self.w = img.shape
        inside = torch.zeros_like(hole)
        inside[r : self.h - r, r : self.w - r] = True
        self.valid = inside & ~_dilate(hole, r)
        self.targets = _dilate(hole, r)
        self.ty, self.tx = torch.nonzero(self.targets, as_tuple=True)
        self.index = torch.full((self.h, self.w), -1, dtype=torch.long, device=img.device)
        self.index[self.ty, self.tx] = torch.arange(self.ty.numel(), device=img.device)
        self.vy, self.vx = torch.nonzero(self.valid, as_tuple=True)
        self.hole_flat = hole.reshape(-1)
        self.hole_weight = 1.0  # weight of not-yet-real (hole) pixels in patch distances
        span = torch.arange(-r, r + 1, device=img.device)
        grid_y, grid_x = torch.meshgrid(span, span, indexing="ij")
        grid_y, grid_x = grid_y.reshape(-1), grid_x.reshape(-1)
        if sparse:
            # Full-resolution level: compare patches on a checkerboard (25 of 49 pixels).
            keep = (grid_y + grid_x) % 2 == 0
            grid_y, grid_x = grid_y[keep], grid_x[keep]
        self.offsets = (grid_y.reshape(-1, 1), grid_x.reshape(-1, 1))
        # Votes from patches centred deep inside the hole see less real image: trust them
        # less (Wexler et al. 2007, confidence gamma^-distance to the known area).
        depth = cv2.distanceTransform(hole.cpu().numpy().astype(np.uint8), cv2.DIST_L2, 3)
        depth = np.minimum(depth, _MAX_DEPTH)  # keep the weight above float32 underflow
        conf = torch.from_numpy(_CONFIDENCE ** (-depth)).to(img.device)
        self.confidence = conf[self.ty, self.tx]

    def random_sources(self, count: int, gen: torch.Generator) -> tuple[torch.Tensor, ...]:
        pick = torch.randint(0, self.vy.numel(), (count,), generator=gen, device="cpu")
        pick = pick.to(self.img.device)
        return self.vy[pick], self.vx[pick]

    def is_valid(self, sy: torch.Tensor, sx: torch.Tensor) -> torch.Tensor:
        ok = (sy >= 0) & (sy < self.h) & (sx >= 0) & (sx < self.w)
        out = torch.zeros_like(ok)
        out[ok] = self.valid[sy[ok], sx[ok]]
        return out

    def distance(self, sy: torch.Tensor, sx: torch.Tensor) -> torch.Tensor:
        """Patch SSD between every target and its source (sources must be valid)."""
        flat = self.img.reshape(self.img.shape[0], -1)
        oy, ox = self.offsets
        out = torch.empty(sy.shape, device=self.img.device)
        for start in range(0, sy.numel(), _CHUNK):
            stop = start + _CHUNK
            ty = (self.ty[start:stop][None] + oy).clamp(0, self.h - 1)
            tx = (self.tx[start:stop][None] + ox).clamp(0, self.w - 1)
            src = (sy[start:stop][None] + oy) * self.w + sx[start:stop][None] + ox
            target = ty * self.w + tx
            diff = flat[:, target] - flat[:, src]
            sq = diff * diff
            if self.hole_weight != 1.0:
                sq = sq * torch.where(self.hole_flat[target], self.hole_weight, 1.0)[None]
            out[start:stop] = sq.sum((0, 1))
        return out

    def exhaustive(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Best source for every target by brute force (coarsest level only)."""
        r = _PATCH_RADIUS
        channels = self.img.shape[0]
        patches = functional.unfold(self.img[None], 2 * r + 1, padding=r)[0]
        weight_map = torch.where(self.hole, self.hole_weight, 1.0)[None, None]
        weights = functional.unfold(weight_map, 2 * r + 1, padding=r)[0].repeat(channels, 1)
        tidx = self.ty * self.w + self.tx
        vidx = self.vy * self.w + self.vx
        src = patches[:, vidx]  # (D, Ns)
        src_sq = src * src
        best = torch.empty(tidx.numel(), dtype=torch.long, device=self.img.device)
        rows = max(1, _BRUTE_PAIRS // max(1, vidx.numel()))
        for start in range(0, tidx.numel(), rows):
            t = patches[:, tidx[start : start + rows]].T  # (n, D)
            w = weights[:, tidx[start : start + rows]].T
            dist = (w * t * t).sum(1, keepdim=True) - 2.0 * (w * t) @ src + w @ src_sq
            best[start : start + rows] = dist.argmin(1)
        return self.vy[best], self.vx[best]

    def vote(self, sy: torch.Tensor, sx: torch.Tensor, dist: torch.Tensor) -> None:
        """Replace hole pixels by the weighted mean of overlapping source patches."""
        r = _PATCH_RADIUS
        channels = self.img.shape[0]
        flat = self.img.reshape(channels, -1)
        sample = dist[:: max(1, dist.numel() >> 20)]  # strided: quantile caps at 16M values
        sigma2 = float(torch.quantile(sample, 0.75)) + 1e-6
        weight = torch.exp((-dist / (2.0 * sigma2)).clamp_min(-60.0)) * self.confidence
        num = torch.zeros(channels, self.h * self.w, device=self.img.device)
        den = torch.zeros(self.h * self.w, device=self.img.device)
        hole_flat = self.hole.reshape(-1)
        for dy in range(-r, r + 1):
            py = self.ty + dy
            for dx in range(-r, r + 1):
                px = self.tx + dx
                inside = (py >= 0) & (py < self.h) & (px >= 0) & (px < self.w)
                dest = py.clamp(0, self.h - 1) * self.w + px.clamp(0, self.w - 1)
                keep = inside & hole_flat[dest]
                dest = dest[keep]
                src = (sy[keep] + dy) * self.w + sx[keep] + dx
                num.index_add_(1, dest, flat[:, src] * weight[keep])
                den.index_add_(0, dest, weight[keep])
        filled = den > 0
        flat[:, filled] = num[:, filled] / den[filled]
        self.img = flat.reshape(self.img.shape)


def _improve(
    level: _Level, sy: torch.Tensor, sx: torch.Tensor, gen: torch.Generator, fine: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One PatchMatch pass: jump-flood propagation, then random search.

    At fine levels the field comes from the coarser level and is nearly right: short jumps
    and a local random search (16 px down to 1 px) refine it at a fraction of the cost.
    """
    dist = level.distance(sy, sx)

    def consider(cy: torch.Tensor, cx: torch.Tensor, usable: torch.Tensor) -> None:
        nonlocal sy, sx, dist
        usable = usable & level.is_valid(cy, cx)
        if not bool(usable.any()):
            return
        cy = torch.where(usable, cy, sy)
        cx = torch.where(usable, cx, sx)
        cand = level.distance(cy, cx)
        better = usable & (cand < dist)
        sy = torch.where(better, cy, sy)
        sx = torch.where(better, cx, sx)
        dist = torch.where(better, cand, dist)

    for jump in _JUMPS_FINE if fine else _JUMPS:
        for ey, ex in ((jump, 0), (-jump, 0), (0, jump), (0, -jump)):
            ny, nx = level.ty + ey, level.tx + ex
            inside = (ny >= 0) & (ny < level.h) & (nx >= 0) & (nx < level.w)
            neighbour = torch.full_like(ny, -1)
            neighbour[inside] = level.index[ny[inside], nx[inside]]
            has = neighbour >= 0
            safe = neighbour.clamp_min(0)
            consider(sy[safe] - ey, sx[safe] - ex, has)
    radius = min(_FINE_RADIUS, max(level.h, level.w)) if fine else max(level.h, level.w)
    for _ in range(_RANDOM_TRIES_FINE if fine else _RANDOM_TRIES):
        if radius < 1:
            break
        off = torch.randint(-radius, radius + 1, (2, sy.numel()), generator=gen, device="cpu")
        off = off.to(sy.device)
        consider(sy + off[0], sx + off[1], torch.ones_like(sy, dtype=torch.bool))
        radius //= 2
    return sy, sx, dist


def _upsample_sources(
    fine: _Level, coarse: _Level, sy: torch.Tensor, sx: torch.Tensor, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Carry the coarse nearest-neighbour field to the fine level (x2, keep the offset)."""
    parent = coarse.index[
        (fine.ty // 2).clamp(max=coarse.h - 1), (fine.tx // 2).clamp(max=coarse.w - 1)
    ]
    ry, rx = fine.random_sources(fine.ty.numel(), gen)
    has = parent >= 0
    safe = parent.clamp_min(0)
    cy = sy[safe] * 2 + (fine.ty % 2)
    cx = sx[safe] * 2 + (fine.tx % 2)
    ok = has & fine.is_valid(cy, cx)
    return torch.where(ok, cy, ry), torch.where(ok, cx, rx)


def _content_aware(img: torch.Tensor, hole: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Fill ``hole`` (h,w bool) of ``img`` (3,h,w) by multi-scale patch voting."""
    # Pyramid depth follows the hole thickness (not its bounding box): a scattered mask
    # must not be shrunk until the whole coarse level is hole.
    depth_map = cv2.distanceTransform(hole.cpu().numpy().astype(np.uint8), cv2.DIST_L2, 3)
    extent = 2.0 * float(depth_map.max()) + 1.0
    levels = [(img, hole)]
    while (
        extent / (2 ** (len(levels) - 1)) > _COARSE_EXTENT
        and min(levels[-1][0].shape[1:]) // 2 >= _MIN_SIDE
    ):
        levels.append(_downsample(*levels[-1]))

    # Coarsest level starts from a smooth fill and random sources.
    coarse_img, coarse_hole = levels[-1]
    smooth = _spread(
        coarse_img[None],
        (~coarse_hole).float()[None, None],
        coarse_hole[None, None],
        max(1.0, extent / 2 ** len(levels)),
    )
    if smooth is None:
        return torch.where(hole[None], _smooth(img, hole), img)
    start = torch.where(coarse_hole[None], smooth[0], coarse_img)
    level = _Level(start, coarse_hole)
    if level.vy.numel() == 0:
        return torch.where(hole[None], _smooth(img, hole), img)
    # Coarsest level: exhaustive search, first matching only the real (known) pixels of each
    # patch, then giving the filled-in pixels more and more say.
    previous = None
    sy = sx = None
    for depth in range(len(levels) - 1, -1, -1):
        if previous is not None:
            img_l, hole_l = levels[depth]
            up = functional.interpolate(
                previous.img[None], size=img_l.shape[1:], mode="bilinear", align_corners=False
            )[0]
            level = _Level(torch.where(hole_l[None], up, img_l), hole_l, sparse=depth == 0)
            if level.vy.numel() == 0:
                return torch.where(hole[None], _smooth(img, hole), img)
            sy, sx = _upsample_sources(level, previous, sy, sx, gen)
            for _ in range(_ITERS_FULL if depth == 0 else _ITERS_FINE):
                sy, sx, dist = _improve(level, sy, sx, gen, fine=True)
                level.vote(sy, sx, dist)
        else:
            for step in range(_ITERS_COARSE):
                level.hole_weight = (step + 1) / _ITERS_COARSE
                if step % 3 == 0 and level.ty.numel() * level.vy.numel() <= _BRUTE_LIMIT:
                    sy, sx = level.exhaustive()
                elif sy is None:
                    sy, sx = level.random_sources(level.ty.numel(), gen)
                sy, sx, dist = _improve(level, sy, sx, gen)
                level.vote(sy, sx, dist)
        previous = level
    return level.img


def _smooth(img: torch.Tensor, hole: torch.Tensor) -> torch.Tensor:
    ys, xs = torch.nonzero(hole, as_tuple=True)
    extent = max(int(ys.max() - ys.min()) + 1, int(xs.max() - xs.min()) + 1)
    weight = (~hole).float()[None, None]
    out = _spread(img[None], weight, hole[None, None], max(1.0, extent / 4.0))
    return out[0] if out is not None else img


def heal(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    mode: str = "content_aware",
    grow: int = 2,
    feather: float = 1.5,
    context: float = 1.0,
) -> torch.Tensor:
    """Fill the masked area of each image from its surroundings.

    Args:
        image: IMAGE ``(B,H,W,3)``.
        mask: MASK ``(B,H,W)`` (or batch 1); pixels > 0.5 are replaced.
        mode: ``content_aware`` (patch-based, keeps texture) or ``smooth``.
        grow: Pixels added around the mask before filling (covers soft halos), 0..64.
        feather: Blend width (px) at the mask edge, 0..32.
        context: Search area around the mask, as a multiple of the mask size, 0.25..8.

    Returns:
        ``(B,H,W,3)`` float32 image; unchanged outside the grown, feathered mask.
    """
    image, mask = _check(image, mask)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}.")
    if isinstance(grow, bool) or not isinstance(grow, Integral) or not 0 <= grow <= 64:
        raise ValueError("grow must be an integer in [0, 64].")
    for name, value, lo, hi in (("feather", feather, 0.0, 32.0), ("context", context, 0.25, 8.0)):
        if isinstance(value, bool) or not isinstance(value, Real) or not lo <= value <= hi:
            raise ValueError(f"{name} must be in [{lo}, {hi}].")

    out = image.clone()
    for index in range(image.shape[0]):
        out[index] = _heal_frame(
            image[index], mask[index] > 0.5, mode, int(grow), float(feather), float(context)
        )
    return out


def _heal_frame(
    frame: torch.Tensor, mask: torch.Tensor, mode: str, grow: int, feather: float, context: float
) -> torch.Tensor:
    """Heal one (H,W,3) frame.

    Mask parts far apart are healed separately, each with its own search region, so a
    scattered mask (many small spots) stays fast and accurate.
    """
    hole_full = _dilate(mask, grow)
    if not bool(hole_full.any()):
        return frame
    synth = _dilate(hole_full, math.ceil(feather))  # synthesised: hole plus the feather band
    gen = torch.Generator().manual_seed(_SEED)  # same result wherever the frame sits in a batch
    near = _dilate(synth, _CLUSTER_GAP // 2).cpu().numpy().astype(np.uint8)
    count, labels = cv2.connectedComponents(near, connectivity=8)
    labels_t = torch.from_numpy(labels).to(frame.device)
    out = frame.clone()
    for label in range(1, count):
        part = synth & (labels_t == label)
        if not bool(part.any()):
            continue
        y0, y1, x0, x1 = _region(part, context)
        crop = frame[y0:y1, x0:x1].permute(2, 0, 1)
        hole = synth[y0:y1, x0:x1]  # every masked pixel in the region counts as unknown
        if bool(hole.all()):
            continue  # nothing to copy from
        if mode == "content_aware":
            filled = _content_aware(crop, hole, gen)
        else:
            filled = torch.where(hole[None], _smooth(crop, hole), crop)
        # Opaque over the (grown) mask, fading to the original across the feather band.
        inner = hole_full[y0:y1, x0:x1].cpu().numpy().astype(np.uint8)
        dist = cv2.distanceTransform(1 - inner, cv2.DIST_L2, 3)
        alpha = torch.from_numpy(np.clip(1.0 - dist / (feather + 1.0), 0.0, 1.0)).to(frame.device)
        alpha = alpha * part[y0:y1, x0:x1].float()
        blended = crop + (filled.clamp(0.0, 1.0) - crop) * alpha[None]
        out[y0:y1, x0:x1] = blended.permute(1, 2, 0)
    return out


def run(image: torch.Tensor, mask: torch.Tensor, **kwargs: object) -> torch.Tensor:
    """``heal`` on the GPU when there is one (ComfyUI images live on the CPU)."""
    if image.device.type == "cpu" and torch.cuda.is_available():
        try:
            return heal(image.cuda(), mask.cuda(), **kwargs).cpu()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    return heal(image, mask, **kwargs)
