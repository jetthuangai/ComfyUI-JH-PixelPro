"""GPU face detection with Kornia YuNet (5-point landmarks) for BHWC RGB tensors."""

from __future__ import annotations

import math
from numbers import Real

import torch
import torch.nn.functional as func

from .face_detect import _prepare_image, _validate_confidence, _validate_int, _validate_mode

_MIN_SIDE = 64  # YuNet's stride-64 head collapses below this (crash under ~56 px).
_MAX_DETECT_SIZE = 1024  # YuNet anchors top out at 256 px: larger inputs break big faces.
_DETECTOR_CACHE: dict[str, torch.nn.Module] = {}


def _validate_nms(nms_threshold: float) -> float:
    if isinstance(nms_threshold, bool) or not isinstance(nms_threshold, Real):
        raise ValueError("nms_threshold must be in [0.1, 0.9].")
    value = float(nms_threshold)
    if not 0.1 <= value <= 0.9:
        raise ValueError("nms_threshold must be in [0.1, 0.9].")
    return value


def _pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_detector(
    device: torch.device, confidence_threshold: float, nms_threshold: float
) -> torch.nn.Module:
    """Return the cached ``kornia.contrib.FaceDetector`` (YuNet) for ``device``.

    One instance per device; thresholds are plain attributes read on every call, so they are
    set here instead of being part of the cache key.
    """
    key = str(device)
    detector = _DETECTOR_CACHE.get(key)
    if detector is None:
        try:
            from kornia.contrib import FaceDetector

            # Weight init must not advance the global torch RNG (reproducible downstream noise).
            with torch.random.fork_rng(devices=[]):
                detector = FaceDetector()
            detector = detector.to(device).eval()
        except Exception as exc:
            raise RuntimeError(
                "Could not load the YuNet face detector weights (yunet_final.pth). Kornia "
                "downloads them once into the torch hub cache (~/.cache/torch/hub/checkpoints); "
                "check network access or copy the file there manually. Alternative: use "
                f"Face Detect (MediaPipe). Original error: {exc}"
            ) from exc
        _DETECTOR_CACHE[key] = detector
    detector.confidence_threshold = confidence_threshold
    detector.nms_threshold = nms_threshold
    return detector


def _clear_detector_cache() -> None:
    _DETECTOR_CACHE.clear()


_get_detector.cache_clear = _clear_detector_cache  # type: ignore[attr-defined]


def _to_candidate(
    row: list[float], *, width: int, height: int, batch_index: int
) -> tuple[list[list[float]], dict[str, float | int]]:
    # Exclusive end bounds clamped to the image size so edge-touching boxes keep the last
    # column/row; start is at most size - 1 so every box is at least 1 px.
    x0 = int(math.floor(min(max(row[0], 0.0), width - 1)))
    y0 = int(math.floor(min(max(row[1], 0.0), height - 1)))
    x1 = int(math.ceil(min(max(row[2], 0.0), width)))
    y1 = int(math.ceil(min(max(row[3], 0.0), height)))
    points = [[row[4 + 2 * i], row[5 + 2 * i]] for i in range(5)]
    # N-10 order is image-left first; swap pairs if the detector reports them mirrored.
    if points[0][0] > points[1][0]:
        points[0], points[1] = points[1], points[0]
    if points[3][0] > points[4][0]:
        points[3], points[4] = points[4], points[3]
    box = {
        "x": x0,
        "y": y0,
        "w": min(max(1, x1 - x0), width - x0),
        "h": min(max(1, y1 - y0), height - y0),
        "conf": round(float(row[14]), 4),
        "batch_index": batch_index,
    }
    return points, box


def face_detect_lite(
    image_bhwc: torch.Tensor,
    *,
    mode: str = "single_largest",
    max_faces: int = 1,
    confidence_threshold: float = 0.6,
    nms_threshold: float = 0.3,
    detect_size: int = 640,
) -> tuple[list[list[list[float]]], list[dict], int, torch.Tensor]:
    """Detect faces with YuNet (Kornia) on GPU when available.

    Args:
        image_bhwc: float32 RGB tensor (B, H, W, 3) in [0, 1].
        mode: ``single_largest`` (1 face per image) or ``multi_top_k`` (up to ``max_faces``).
        max_faces: cap per image for ``multi_top_k``, in [1, 10].
        confidence_threshold: YuNet score gate in [0.1, 0.95].
        nms_threshold: NMS IoU threshold in [0.1, 0.9].
        detect_size: long side of the detection copy in [160, 1024]; never upscales.
            Larger values break big faces (YuNet anchors are size-limited); 640 suits portraits.

    Returns:
        ``(landmarks, boxes, face_count, face_mask)``. ``landmarks`` has one entry per face with
        5 ``[x, y]`` pixel points ordered image-left eye, image-right eye, nose tip, image-left
        mouth corner, image-right mouth corner (same as N-10). ``boxes`` hold ``x, y, w, h,
        conf, batch_index``. ``face_mask`` is float32 (B, H, W), 1 inside selected boxes.

    Raises:
        TypeError / ValueError: invalid inputs.
        RuntimeError: YuNet weights could not be loaded.
    """
    image = _prepare_image(image_bhwc)
    mode = _validate_mode(mode)
    max_faces = _validate_int("max_faces", max_faces, lower=1, upper=10)
    confidence_threshold = _validate_confidence(confidence_threshold)
    nms_threshold = _validate_nms(nms_threshold)
    detect_size = _validate_int("detect_size", detect_size, lower=160, upper=_MAX_DETECT_SIZE)

    batch, height, width, _ = image.shape
    device = _pick_device()
    detector = _get_detector(device, confidence_threshold, nms_threshold)

    scale = min(1.0, detect_size / max(height, width))
    x = image.detach().to(device).permute(0, 3, 1, 2)
    if scale < 1.0:
        new_h = max(1, round(height * scale))
        new_w = max(1, round(width * scale))
        x = func.interpolate(x, size=(new_h, new_w), mode="bilinear", antialias=True)
        sx, sy = width / new_w, height / new_h
    else:
        sx = sy = 1.0
    # Tiny images: pad right/bottom with the image's mean colour so YuNet gets >= 64 px.
    pad_h = max(0, _MIN_SIDE - x.shape[2])
    pad_w = max(0, _MIN_SIDE - x.shape[3])
    if pad_h or pad_w:
        fill = x.mean(dim=(2, 3), keepdim=True)
        canvas = fill.expand(-1, -1, x.shape[2] + pad_h, x.shape[3] + pad_w).clone()
        canvas[:, :, : x.shape[2], : x.shape[3]] = x
        x = canvas
    # YuNet expects RGB in 0..255.
    with torch.no_grad():
        detections = detector(x * 255.0)

    mask = torch.zeros((batch, height, width), dtype=torch.float32)
    all_landmarks: list[list[list[float]]] = []
    all_boxes: list[dict] = []
    factors = torch.tensor([sx, sy] * 7 + [1.0])
    for batch_index, dets in enumerate(detections):
        rows = (dets.detach().float().cpu() * factors).tolist()
        candidates = [
            _to_candidate(row, width=width, height=height, batch_index=batch_index) for row in rows
        ]
        candidates.sort(key=lambda item: int(item[1]["w"]) * int(item[1]["h"]), reverse=True)
        keep = candidates[:1] if mode == "single_largest" else candidates[:max_faces]
        for points, box in keep:
            all_landmarks.append(points)
            all_boxes.append(box)
            mask[batch_index, box["y"] : box["y"] + box["h"], box["x"] : box["x"] + box["w"]] = 1.0

    return all_landmarks, all_boxes, len(all_landmarks), mask.to(image.device)
