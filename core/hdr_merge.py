"""HDR merge: fuse a bracketed exposure series into one well-exposed image.

1. Order: frames are sorted by brightness; the middle one is the reference.
2. Align (optional): Ward's median-threshold-bitmap alignment (``cv2.AlignMTB``) on a
   ~1024 px proxy removes hand-held shifts between frames (translation only); a shift is
   kept only if it improves the match.
3. Deghost (optional): each frame is scaled to the reference brightness (median ratio over
   pixels well exposed in both); where it still disagrees strongly (a person or leaves that
   moved) the frame takes the reference pixels, brightness-matched, so nothing doubles.
4. Exposure fusion (Mertens, Kautz & Van Reeth 2007, ``cv2.MergeMertens``): every pixel is
   a multi-resolution blend of the frames weighted by contrast, saturation and
   well-exposedness. No camera response or exposure values are needed and the result is
   already display-ready (no tone mapping step).

Runs on the CPU in OpenCV (float32), in place where possible: peak RAM is about 6x the
batch in float32.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch

_GOOD_LOW, _GOOD_HIGH = 0.08, 0.92  # well-exposed range used to compare frames
_GHOST_THRESHOLD = 0.12  # brightness-matched difference that counts as motion
_GHOST_BLUR = 5  # px, smooths the ghost map so replaced areas have soft edges
_ALIGN_SIDE = 1024  # MTB runs on a proxy this large; shifts are scaled back up
_MIN_SIDE = 16
_ALL_WHITE = 0.995  # every frame brighter than this everywhere: return the reference


def _check(images: torch.Tensor) -> None:
    """Validate shape and dtype only; frames are converted one at a time."""
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor.")
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"images must have shape (B,H,W,3), got {tuple(images.shape)}.")
    if images.shape[0] < 2:
        raise ValueError("HDR merge needs a batch of at least 2 exposures.")
    if not torch.is_floating_point(images):
        raise ValueError(f"images must be floating point, got {images.dtype}.")
    if min(images.shape[1:3]) < _MIN_SIDE:
        raise ValueError(f"images must be at least {_MIN_SIDE} x {_MIN_SIDE} pixels.")


def _luma(frame: np.ndarray) -> np.ndarray:
    return frame @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _grey_proxy(frame: np.ndarray, scale: float) -> np.ndarray:
    grey = (np.clip(_luma(frame), 0.0, 1.0) * 255).astype(np.uint8)
    if scale < 1.0:
        size = (max(16, round(grey.shape[1] * scale)), max(16, round(grey.shape[0] * scale)))
        grey = cv2.resize(grey, size, interpolation=cv2.INTER_AREA)
    return grey


def _mismatch(a: np.ndarray, b: np.ndarray) -> float:
    """Median-threshold-bitmap disagreement of two grey images (lower = better aligned)."""
    return float(np.mean((a > np.median(a)) != (b > np.median(b))))


def _align(frames: list[np.ndarray], reference: int) -> None:
    """Shift every frame onto the reference (in place) with MTB on a ~1024 px proxy.

    The proxy keeps the search range proportional to the image (large hand-held shifts of
    a 24 MP frame are found); the shift is scaled back up and kept only if it lowers the
    bitmap mismatch.
    """
    height, width = frames[0].shape[:2]
    scale = min(1.0, _ALIGN_SIDE / max(height, width))
    ref = _grey_proxy(frames[reference], scale)
    bits = int(max(1, min(6, math.floor(math.log2(max(min(ref.shape), 32) / 16)))))
    mtb = cv2.createAlignMTB(max_bits=bits, exclude_range=4, cut=False)
    for index, frame in enumerate(frames):
        if index == reference:
            continue
        grey = _grey_proxy(frame, scale)
        dx, dy = mtb.calculateShift(ref, grey)
        if (dx, dy) == (0, 0) or _mismatch(ref, mtb.shiftMat(grey, (dx, dy))) >= _mismatch(
            ref, grey
        ):
            continue  # no shift, or MTB did not improve the match
        full = (int(round(dx / scale)), int(round(dy / scale)))
        frames[index] = mtb.shiftMat(frame, full)


def _deghost(frames: list[np.ndarray], reference: int) -> None:
    """Replace moved areas of each frame (in place) by the brightness-matched reference."""
    ref = frames[reference]
    ref_luma = _luma(ref)
    for index, frame in enumerate(frames):
        if index == reference:
            continue
        luma = _luma(frame)
        good = (
            (luma > _GOOD_LOW)
            & (luma < _GOOD_HIGH)
            & (ref_luma > _GOOD_LOW)
            & (ref_luma < _GOOD_HIGH)
        )
        if good.sum() < 64:
            continue
        ratio = float(np.median(luma[good] / np.maximum(ref_luma[good], 1e-4)))
        # Compare only where both frames carry information (clipped areas always differ).
        valid = (luma > _GOOD_LOW) & (luma < _GOOD_HIGH) & (ref_luma * ratio < _GOOD_HIGH)
        moved = (np.abs(luma - np.clip(ref_luma * ratio, 0.0, 1.0)) > _GHOST_THRESHOLD) & valid
        if not moved.any():
            continue
        weight = cv2.GaussianBlur(moved.astype(np.float32), (0, 0), _GHOST_BLUR)
        weight = np.clip(weight * 2.0, 0.0, 1.0)[..., None]
        frame *= 1.0 - weight
        frame += np.clip(ref * ratio, 0.0, 1.0) * weight


def hdr_merge(images: torch.Tensor, *, align: bool = True, deghost: bool = True) -> torch.Tensor:
    """Fuse a batch of exposures (B,H,W,3) into one image (1,H,W,3).

    The reference is the middle exposure (for an even count, the brighter of the two in
    the middle). Peak RAM is about 6x the batch in float32.
    """
    _check(images)
    frames = []
    for frame in images:
        array = frame.detach().float().cpu().numpy()
        frames.append(np.clip(np.nan_to_num(array, nan=0.0), 0.0, 1.0).astype(np.float32))
    order = np.argsort([float(_luma(f).mean()) for f in frames])
    frames = [frames[i] for i in order]
    reference = len(frames) // 2
    if min(float(f.min()) for f in frames) > _ALL_WHITE:
        return torch.from_numpy(frames[reference]).to(images.device)[None]  # nothing to fuse
    if align:
        _align(frames, reference)
    if deghost:
        _deghost(frames, reference)
    for frame in frames:
        frame *= 255.0  # MergeMertens divides float input by 255 (0..255 scale)
    fused = cv2.createMergeMertens().process([np.ascontiguousarray(f) for f in frames])
    del frames
    fused = np.clip(np.nan_to_num(fused), 0.0, 1.0)
    return torch.from_numpy(fused).to(images.device)[None]
