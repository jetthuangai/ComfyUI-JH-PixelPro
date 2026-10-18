"""Culling score: rank a batch of photos by sharpness, exposure, closed eyes and duplicates.

Everything is measured on a 1024 px proxy, one photo at a time, so scores do not depend on
image size and memory does not grow with the batch.

- Sharpness: Laplacian energy per tile (8 x 8 grid), 90th percentile, so a sharp subject on
  a blurred background still counts as sharp. With faces (YuNet) the face boxes are
  measured instead. Scored on a log scale (1e-5 .. 1e-2); when the batch holds other shots
  of the same scene, half the score is relative to the sharpest of them (10x blurrier = 0).
- Exposure: penalties for clipped highlights, crushed shadows and a mean far from mid-grey.
- Eyes (optional, MediaPipe face mesh): eye aspect ratio (Soukupova & Cech 2016) below
  0.18 on one of the two largest faces counts as closed (strangers in the background are
  ignored).
- Duplicates: 64-bit difference hash; photos linked by chains of hashes within 10 bits
  form one group and only the best of each group is kept.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
import torch
import torch.nn.functional as functional

from .guided import make_proxy

PROXY_SIDE = 1024
_MIN_SIDE = 8
_LAPLACE = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
_SHARP_LOG = (-5.0, -2.0)  # log10 Laplacian energy mapped to 0..1
_REL_DECADES = 1.0  # this many decades below the sharpest shot of the scene scores 0
_EAR_CLOSED = 0.18
_EYE_FACES = 2  # only the largest faces decide "eyes closed"
_MIN_EYE_FACE = 0.01  # ... and only if they cover at least this share of the frame
_DUP_BITS = 10
_SCENE_BITS = 16  # hash distance within which shots count as the same scene
# MediaPipe face-mesh eye contours: p1 (corner), p2, p3 (upper lid), p4 (corner), p5, p6.
_LEFT_EYE = (33, 160, 158, 133, 153, 144)
_RIGHT_EYE = (362, 385, 387, 263, 373, 380)
_WEIGHTS = {"sharpness": 0.6, "exposure": 0.4}
_EYES_PENALTY = 0.4


def _check(images: torch.Tensor) -> None:
    """Validate shape and dtype only; frames are sanitised one at a time later."""
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor.")
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"images must have shape (B,H,W,3), got {tuple(images.shape)}.")
    if not torch.is_floating_point(images):
        raise ValueError(f"images must be floating point, got {images.dtype}.")
    if min(images.shape[1:3]) < _MIN_SIDE:
        raise ValueError(f"images must be at least {_MIN_SIDE} x {_MIN_SIDE} pixels.")


def _proxy(frame: torch.Tensor) -> torch.Tensor:
    """(H,W,3) frame -> sanitised (1,3,h,w) proxy on the CPU."""
    chw = frame.permute(2, 0, 1)[None].float()
    small = make_proxy(chw, PROXY_SIDE).cpu()
    return torch.nan_to_num(small, nan=0.0).clamp(0.0, 1.0)


def _luma(rgb_bchw: torch.Tensor) -> torch.Tensor:
    weights = rgb_bchw.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    return (rgb_bchw * weights).sum(1, keepdim=True)


def _laplacian_energy(luma: torch.Tensor) -> torch.Tensor:
    kernel = _LAPLACE.to(luma).view(1, 1, 3, 3)
    return functional.conv2d(luma, kernel, padding=1)[..., 1:-1, 1:-1] ** 2


def sharpness_energy(luma: torch.Tensor, boxes: list[tuple[int, int, int, int]]) -> float:
    """Raw sharpness of one proxy luma (1,1,h,w); ``boxes`` (y0,y1,x0,x1) limit it to faces."""
    energy = _laplacian_energy(luma)
    if boxes:
        values = []
        for y0, y1, x0, x1 in boxes:
            part = energy[..., max(0, y0 - 1) : max(1, y1 - 1), max(0, x0 - 1) : max(1, x1 - 1)]
            if part.shape[-1] >= 4 and part.shape[-2] >= 4:
                tiles = functional.adaptive_avg_pool2d(part, (4, 4)).flatten()
                values.append(float(torch.quantile(tiles, 0.75)))
        if values:
            return max(values)
    tiles = functional.adaptive_avg_pool2d(energy, (8, 8)).flatten()
    return float(torch.quantile(tiles, 0.9))


def exposure_score(luma: torch.Tensor) -> float:
    clipped = float((luma > 0.98).float().mean())
    crushed = float((luma < 0.02).float().mean())
    mean = float(luma.mean())
    penalty = 4.0 * clipped + 2.0 * crushed + max(0.0, abs(mean - 0.45) - 0.15) * 2.0
    return max(0.0, 1.0 - penalty)


def dhash(luma: torch.Tensor) -> int:
    small = functional.interpolate(luma, size=(8, 9), mode="area")[0, 0]
    bits = (small[:, 1:] > small[:, :-1]).flatten().tolist()
    return sum(1 << i for i, bit in enumerate(bits) if bit)


def _ear(points: np.ndarray, ids: tuple[int, ...]) -> float:
    p1, p2, p3, p4, p5, p6 = (points[i] for i in ids)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 1.0
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * horizontal))


def _eyes_closed(proxy_bhwc: torch.Tensor) -> bool | None:
    """Closed eyes on one of the largest faces of a (1,h,w,3) proxy; None if unavailable."""
    try:
        from .face_pipeline import extract_landmarks

        landmarks, visible = extract_landmarks(proxy_bhwc, max_num_faces=5)
    except torch.OutOfMemoryError:
        raise
    except (ImportError, RuntimeError, OSError):  # MediaPipe or its model not available
        return None
    height, width = proxy_bhwc.shape[1:3]
    scale = np.array([width, height], dtype=np.float32)
    faces = []
    for f in range(landmarks.shape[1]):
        if float(visible[0, f]) <= 0:
            continue
        points = landmarks[0, f].cpu().numpy() * scale
        span = points.max(0) - points.min(0)
        faces.append((float(span[0] * span[1]) / (height * width), points))
    faces.sort(key=lambda item: -item[0])
    closed = False
    for area, points in faces[:_EYE_FACES]:
        if area < _MIN_EYE_FACE:
            continue
        closed |= min(_ear(points, _LEFT_EYE), _ear(points, _RIGHT_EYE)) < _EAR_CLOSED
    return closed


def _face_boxes(proxy_bhwc: torch.Tensor) -> list[tuple[int, int, int, int]] | None:
    """YuNet face boxes on a (1,h,w,3) proxy; None when the detector is unavailable."""
    try:
        from .face_detect_lite import face_detect_lite

        _, boxes, _, _ = face_detect_lite(proxy_bhwc, mode="multi_top_k", max_faces=5)
    except torch.OutOfMemoryError:
        raise
    except (ImportError, RuntimeError, OSError):  # weights could not be loaded
        return None
    out = []
    for box in boxes:
        y0, x0 = int(box["y"]), int(box["x"])
        y1, x1 = y0 + int(box["h"]), x0 + int(box["w"])
        if y1 - y0 >= 8 and x1 - x0 >= 8:
            out.append((y0, y1, x0, x1))
    return out


def _distance(a: dict, b: dict) -> int:
    return bin(a["hash"] ^ b["hash"]).count("1")


def _groups(rows: list[dict], bits: int) -> list[int]:
    """Union-find over hash distance: the group id of every row (chains link)."""
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if _distance(rows[i], rows[j]) <= bits:
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(rows))]


def cull_scores(
    images: torch.Tensor, *, check_eyes: bool = True, use_faces: bool = True
) -> list[dict]:
    """Score every image of a batch. Returns one dict per image (see module docstring)."""
    _check(images)
    rows = []
    for b in range(images.shape[0]):
        proxy = _proxy(images[b])
        bhwc = proxy.permute(0, 2, 3, 1)
        luma = _luma(proxy)
        boxes = _face_boxes(bhwc) if use_faces else None
        closed = _eyes_closed(bhwc) if check_eyes else None
        rows.append(
            {
                "index": b,
                "energy": sharpness_energy(luma, boxes or []),
                "exposure": round(exposure_score(luma), 3),
                "eyes_closed": closed,
                "eyes_checked": closed is not None,
                "faces": len(boxes or []),
                "faces_checked": boxes is not None,
                "hash": dhash(luma),
            }
        )

    lo, hi = _SHARP_LOG
    scenes = _groups(rows, _SCENE_BITS)
    for r, scene in zip(rows, scenes, strict=True):
        energy = max(r["energy"], 1e-12)
        absolute = min(1.0, max(0.0, (math.log10(energy) - lo) / (hi - lo)))
        same = [o["energy"] for o, s in zip(rows, scenes, strict=True) if s == scene]
        if len(same) > 1:
            # Half relative to the sharpest other shot of the scene: a smooth portrait must
            # not lose to a busy street photo in the same batch.
            sharpest = max(max(same), 1e-12)
            relative = min(1.0, max(0.0, 1.0 + math.log10(energy / sharpest) / _REL_DECADES))
            r["sharpness"] = round(0.5 * absolute + 0.5 * relative, 3)
        else:
            r["sharpness"] = round(absolute, 3)  # alone: nothing to compare against
        score = _WEIGHTS["sharpness"] * r["sharpness"] + _WEIGHTS["exposure"] * r["exposure"]
        if r["eyes_closed"]:
            score -= _EYES_PENALTY
        r["score"] = round(max(0.0, min(1.0, score)), 3)

    # Duplicates: every group keeps its best photo (ties: lowest index).
    duplicates = _groups(rows, _DUP_BITS)
    for group in set(duplicates):
        members = [r for r, g in zip(rows, duplicates, strict=True) if g == group]
        best = max(members, key=lambda r: (r["score"], -r["index"]))
        for r in members:
            r["duplicate_of"] = None if r is best else best["index"]
    for r in rows:
        r.pop("hash")
        r["energy"] = float(f"{r['energy']:.3g}")
    return rows


def select(rows: list[dict], top_k: int, min_score: float) -> list[int]:
    """Indices of the photos to keep, best first."""
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 0:
        raise ValueError("top_k must be an integer >= 0.")
    if isinstance(min_score, bool) or not isinstance(min_score, Real) or not 0 <= min_score <= 1:
        raise ValueError("min_score must be in [0, 1].")
    keep = [r for r in rows if r["duplicate_of"] is None and r["score"] >= min_score]
    keep.sort(key=lambda r: (-r["score"], r["index"]))
    if top_k:
        keep = keep[:top_k]
    return [r["index"] for r in keep]
