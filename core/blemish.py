"""Auto Blemish: find and heal spots, pimples and small marks on facial skin.

1. Region: each detected face box (YuNet, ``core.face_detect_lite``) grown by 25 %,
   processed as its own crop, kept only where the colour is skin-like (YCrCb range,
   softly), minus discs around the eyes, brows and mouth placed from the five landmarks.
   A user mask replaces the face search (any skin: body, back, hands).
2. Detection: the Dust & Spot detector (median background, compact-blob shape filter,
   calm-surroundings test) run on a "blemish signal" = luma darkening plus redness
   (R - (G+B)/2 above the local skin mean), because pimples are redder more than darker.
   The largest spot follows the face size (about 5 % of the face width).
3. Healing: the texture fill of Dust & Spot (best nearby skin patch, matched on the ring
   and brightness-corrected), so pores continue through the healed spot.
"""

from __future__ import annotations

import math
from numbers import Real

import kornia
import torch

from . import dust_spot as _dust
from .face_detect_lite import face_detect_lite
from .guided import box_filter

_FACE_GROW = 0.25
_SPOT_FRACTION = 0.05  # largest blemish, as a fraction of the face width
_SPOT_RANGE = (4, 64)
_REDNESS_WEIGHT = 0.6  # redness excess counted as this much luma darkening
_EYE_RADIUS, _MOUTH_RADIUS = 0.42, 0.55  # x eye distance
_BROW_LIFT = 0.45  # brow disc centre above each eye, x eye distance
_RING_RATIO = 0.6  # skin texture around a pimple may be busier than around sensor dust


def _check(image: torch.Tensor, name: str, mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor) or mask.ndim != 3:
        raise ValueError(f"{name} must have shape (B,H,W).")
    if mask.shape[1:] != image.shape[1:3] or mask.shape[0] not in (1, image.shape[0]):
        raise ValueError(f"{name} must match the image size (batch 1 or the image batch).")
    mask = mask.to(device=image.device, dtype=torch.float32).clamp(0.0, 1.0)
    return mask.expand(image.shape[0], -1, -1) if mask.shape[0] != image.shape[0] else mask


def skin_likelihood(image: torch.Tensor) -> torch.Tensor:
    """Soft skin-colour map (B,H,W) from YCrCb ranges that cover light to dark skin."""
    ycc = kornia.color.rgb_to_ycbcr(image.permute(0, 3, 1, 2))
    cr, cb = ycc[:, 2], ycc[:, 1]

    def band(x: torch.Tensor, lo: float, hi: float, soft: float = 0.03) -> torch.Tensor:
        return torch.sigmoid((x - lo) / soft * 4) * torch.sigmoid((hi - x) / soft * 4)

    return band(cr, 0.53, 0.70) * band(cb, 0.30, 0.53)


def _disc(height: int, width: int, cx: float, cy: float, radius: float, device) -> torch.Tensor:
    yy = torch.arange(height, device=device, dtype=torch.float32)[:, None]
    xx = torch.arange(width, device=device, dtype=torch.float32)[None, :]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius * radius


def face_regions(image: torch.Tensor, max_faces: int = 10) -> list[dict]:
    """Faces found in the batch, each with its crop box, width and protected features.

    Every entry: ``batch`` index, ``box`` (y0, y1, x0, x1) of the face grown by 25 %,
    ``width`` of the face (px) and ``protect`` (h, w) float mask of eyes, brows and mouth
    in crop coordinates. Masks are built only inside the crop, not the whole frame.
    """
    height, width = image.shape[1:3]
    landmarks, boxes, _, _ = face_detect_lite(
        image, mode="multi_top_k", max_faces=max_faces, confidence_threshold=0.6
    )
    faces = []
    for points, box in zip(landmarks, boxes, strict=False):
        x, y, w, h, b = box["x"], box["y"], box["w"], box["h"], int(box["batch_index"])
        grow_w, grow_h = w * _FACE_GROW, h * _FACE_GROW
        x0, y0 = max(0, int(x - grow_w)), max(0, int(y - grow_h))
        x1, y1 = min(width, int(x + w + grow_w)), min(height, int(y + h + grow_h))
        ch, cw = y1 - y0, x1 - x0
        if ch < 8 or cw < 8:
            continue
        local = [(px - x0, py - y0) for px, py in points]
        (lx, ly), (rx, ry), _, (mlx, mly), (mrx, mry) = local
        eye = max(1.0, math.hypot(rx - lx, ry - ly))
        dev = image.device
        protect = torch.zeros(ch, cw, device=dev, dtype=torch.bool)
        for ex, ey in ((lx, ly), (rx, ry)):
            protect |= _disc(ch, cw, ex, ey, _EYE_RADIUS * eye, dev)
            protect |= _disc(ch, cw, ex, ey - _BROW_LIFT * eye, _EYE_RADIUS * eye, dev)
        mx, my = (mlx + mrx) / 2, (mly + mry) / 2
        mouth = max(_MOUTH_RADIUS * eye, 0.6 * math.hypot(mrx - mlx, mry - mly))
        protect |= _disc(ch, cw, mx, my, mouth, dev)
        faces.append(
            {"batch": b, "box": (y0, y1, x0, x1), "width": float(w), "protect": protect.float()}
        )
    return faces


def _blemish_signal(image: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W) 'luma' in which red spots look as dark as they look red."""
    luma = _dust._luminance(image)
    rgb = image.permute(0, 3, 1, 2)
    redness = rgb[:, :1] - 0.5 * (rgb[:, 1:2] + rgb[:, 2:3])
    # Redness above the local skin tone (31 px box mean, two 1-D passes).
    excess = (redness - box_filter(redness, 15)).clamp_min(0.0)
    return (luma - _REDNESS_WEIGHT * excess).clamp(0.0, 1.0)


def remove_blemishes(
    image: torch.Tensor,
    area: torch.Tensor,
    protect: torch.Tensor | None = None,
    *,
    spot_size: int = 12,
    sensitivity: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Heal blemishes inside ``area`` (B,H,W); never touch ``protect``.

    Returns ``(image, spot_mask)``; pixels outside ``spot_mask`` are unchanged.
    """
    image = _dust._validate_image(image)
    area = _check(image, "area", area)
    protect = _check(image, "protect", protect)
    spot_size = _dust._validate_int("spot_size", spot_size, lower=2, upper=64)
    sensitivity = _dust._validate_float("sensitivity", sensitivity, lower=0.0, upper=1.0)
    allow = (area > 0.5) & (skin_likelihood(image) > 0.5)
    if protect is not None:
        allow &= protect <= 0.5
    if not bool(allow.any()):
        return image, torch.zeros_like(area)

    threshold = 0.01 + 0.2 * (1.0 - sensitivity) ** 2
    factor = max(2, math.ceil(spot_size / 4))
    spots = _dust._detect(
        _blemish_signal(image), spot_size, threshold, "dark", factor, ring_ratio=_RING_RATIO
    )
    keep = allow.unsqueeze(1).to(spots.dtype)
    spots = _dust._dilate(spots * keep, max(1, spot_size // 6)) * keep
    if not bool(spots.any()):
        return image, spots.squeeze(1)
    channels = image.permute(0, 3, 1, 2).contiguous()
    fill = _dust._texture_fill(channels, spots, _dust._fill(channels, spots, spot_size, factor))
    soft = kornia.filters.gaussian_blur2d(spots, (5, 5), (1.0, 1.0), border_type="replicate")
    soft = soft * spots
    cleaned = channels * (1.0 - soft) + fill * soft
    return cleaned.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0), spots.squeeze(1)


def _spot_size(base: float, spot_scale: float) -> int:
    lo, hi = _SPOT_RANGE
    return int(min(hi, max(lo, round(base * spot_scale))))


def auto_blemish(
    image: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    protect_mask: torch.Tensor | None = None,
    sensitivity: float = 0.5,
    spot_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Find faces (or use ``mask``) and heal blemishes on their skin.

    Each face (or each frame of a user mask) is processed in its own crop with a spot size
    from its own width, so small faces in a group photo and a large face in the same
    frame are both handled at the right scale.

    Args:
        image: IMAGE ``(B,H,W,3)``.
        mask: Optional skin area (B,H,W); skips face detection when given.
        protect_mask: Optional area never touched (moles you keep, tattoos).
        sensitivity: 0..1; higher catches fainter marks (and eventually freckles).
        spot_scale: 0.25..4; multiplies the largest spot size (auto: 5 % of face width).

    Returns:
        ``(image, spot_mask, face_count)``.
    """
    image = _dust._validate_image(image)
    if isinstance(spot_scale, bool) or not isinstance(spot_scale, Real):
        raise ValueError("spot_scale must be in [0.25, 4].")
    if not 0.25 <= float(spot_scale) <= 4.0:
        raise ValueError("spot_scale must be in [0.25, 4].")
    user = _check(image, "mask", mask)
    keep = _check(image, "protect_mask", protect_mask)
    height, width = image.shape[1:3]

    jobs = []  # (batch, (y0, y1, x0, x1), area crop, protect crop, base size)
    count = 0
    if user is not None:
        for b in range(image.shape[0]):
            rows = torch.nonzero((user[b] > 0.5).any(1), as_tuple=True)[0]
            cols = torch.nonzero((user[b] > 0.5).any(0), as_tuple=True)[0]
            if cols.numel() == 0:
                continue
            # No face to measure: treat the mask width like a face width.
            base = _SPOT_FRACTION * float(cols.max() - cols.min() + 1)
            pad = 2 * _SPOT_RANGE[1]
            y0, y1 = max(0, int(rows.min()) - pad), min(height, int(rows.max()) + 1 + pad)
            x0, x1 = max(0, int(cols.min()) - pad), min(width, int(cols.max()) + 1 + pad)
            area = user[b, y0:y1, x0:x1]
            jobs.append((b, (y0, y1, x0, x1), area, torch.zeros_like(area), base))
    else:
        faces = face_regions(image)
        count = len(faces)
        for face in faces:
            y0, y1, x0, x1 = face["box"]
            area = torch.ones(y1 - y0, x1 - x0, device=image.device)
            jobs.append(
                (face["batch"], face["box"], area, face["protect"], _SPOT_FRACTION * face["width"])
            )

    out = image.clone()
    spots = torch.zeros(image.shape[:3], device=image.device)
    for b, (y0, y1, x0, x1), area, protect, base in jobs:
        if keep is not None:
            protect = torch.maximum(protect, keep[b, y0:y1, x0:x1])
        crop = out[b : b + 1, y0:y1, x0:x1]
        cleaned, found = remove_blemishes(
            crop,
            area[None],
            protect[None],
            spot_size=_spot_size(base, float(spot_scale)),
            sensitivity=sensitivity,
        )
        out[b, y0:y1, x0:x1] = cleaned[0]
        spots[b, y0:y1, x0:x1] = torch.maximum(spots[b, y0:y1, x0:x1], found[0])
    return out, spots, count
