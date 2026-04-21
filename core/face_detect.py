"""MediaPipe face landmark detection helpers for BHWC RGB tensors."""

from __future__ import annotations

import atexit
import logging
import sys
import urllib.request
from numbers import Integral, Real
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_VALID_MODES = {"multi_top_k", "single_largest"}
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_FACE_MODEL_NAME = "face_landmarker.task"
_FIVE_POINT_INDICES = (33, 263, 1, 61, 291)
_LANDMARKER_CACHE: dict[tuple[str, float, int], object] = {}


def _validate_mode(mode: str) -> str:
    if not isinstance(mode, str) or mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {tuple(sorted(_VALID_MODES))}.")
    return mode


def _validate_int(name: str, value: int, *, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    value_int = int(value)
    if not lower <= value_int <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_int


def _validate_confidence(confidence_threshold: float) -> float:
    if isinstance(confidence_threshold, bool) or not isinstance(confidence_threshold, Real):
        raise ValueError("confidence_threshold must be in [0.1, 0.95].")
    confidence_float = float(confidence_threshold)
    if not 0.1 <= confidence_float <= 0.95:
        raise ValueError("confidence_threshold must be in [0.1, 0.95].")
    return confidence_float


def _prepare_image(image_bhwc: torch.Tensor) -> torch.Tensor:
    if not isinstance(image_bhwc, torch.Tensor):
        raise TypeError("image_bhwc must be a torch.Tensor.")

    if image_bhwc.ndim != 4:
        raise ValueError(f"Expected BHWC tensor, got shape {tuple(image_bhwc.shape)}.")

    if image_bhwc.shape[-1] != 3:
        raise ValueError(f"Expected 3-channel RGB image, got C={image_bhwc.shape[-1]}.")

    if image_bhwc.dtype != torch.float32:
        raise ValueError(f"Expected float32 image tensor, got {image_bhwc.dtype}.")

    value_min, value_max = torch.aminmax(image_bhwc.detach())
    if value_min.item() < 0.0 or value_max.item() > 1.0:
        logger.warning("image input values outside [0,1]; clamped to [0,1].")
        return image_bhwc.clamp(0.0, 1.0)

    return image_bhwc


def _comfy_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_model_dir() -> Path:
    comfy_root = _comfy_root()
    if str(comfy_root) not in sys.path:
        sys.path.insert(0, str(comfy_root))

    try:
        import folder_paths  # type: ignore

        models_dir = Path(folder_paths.models_dir)
    except Exception:
        models_dir = comfy_root / "models"

    model_dir = models_dir / "mediapipe"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _resolve_model_path() -> Path:
    return _resolve_model_dir() / _FACE_MODEL_NAME


def _import_mediapipe() -> tuple[object, object, object, object, object]:
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install via 'pip install mediapipe' (tasks API >= 0.10.33 required for Python 3.12+). "
            "See https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker. "
            "Alternative: bypass S-10 and paste 5-point JSON manually into S-06."
        ) from exc

    return mp, BaseOptions, FaceLandmarker, FaceLandmarkerOptions, RunningMode


def _ensure_model_file() -> Path:
    model_path = _resolve_model_path()
    if model_path.exists():
        return model_path

    print(f"[N-10] Downloading face_landmarker.task (~5MB) to {model_path.parent}...")
    urllib.request.urlretrieve(_MODEL_URL, model_path)
    return model_path


def _get_landmarker(model_path: str, confidence_threshold: float, num_faces: int) -> object:
    cache_key = (model_path, confidence_threshold, num_faces)
    if cache_key in _LANDMARKER_CACHE:
        return _LANDMARKER_CACHE[cache_key]

    _, base_options_cls, face_landmarker_cls, face_landmarker_options_cls, running_mode_cls = (
        _import_mediapipe()
    )
    options = face_landmarker_options_cls(
        base_options=base_options_cls(model_asset_path=model_path),
        running_mode=running_mode_cls.IMAGE,
        min_face_detection_confidence=confidence_threshold,
        num_faces=num_faces,
    )
    landmarker = face_landmarker_cls.create_from_options(options)
    _LANDMARKER_CACHE[cache_key] = landmarker
    return landmarker


def _clear_landmarker_cache() -> None:
    for landmarker in _LANDMARKER_CACHE.values():
        close = getattr(landmarker, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug(
                    "Ignored error while closing FaceLandmarker cache entry.",
                    exc_info=True,
                )
    _LANDMARKER_CACHE.clear()


_get_landmarker.cache_clear = _clear_landmarker_cache  # type: ignore[attr-defined]
atexit.register(_clear_landmarker_cache)


def _bbox_from_points(
    points_xy: torch.Tensor,
    *,
    width: int,
    height: int,
    confidence_threshold: float,
    batch_index: int,
) -> dict[str, float | int]:
    min_xy = points_xy.amin(dim=0)
    max_xy = points_xy.amax(dim=0)
    size = (max_xy - min_xy).clamp_min(1.0)
    padding = size * 0.1
    x0 = int(torch.floor((min_xy[0] - padding[0]).clamp(0.0, width - 1)).item())
    y0 = int(torch.floor((min_xy[1] - padding[1]).clamp(0.0, height - 1)).item())
    x1 = int(torch.ceil((max_xy[0] + padding[0]).clamp(0.0, width - 1)).item())
    y1 = int(torch.ceil((max_xy[1] + padding[1]).clamp(0.0, height - 1)).item())
    return {
        "x": x0,
        "y": y0,
        "w": max(1, x1 - x0),
        "h": max(1, y1 - y0),
        # MediaPipe FaceLandmarker does not expose a per-face score here; keep the gate used.
        "conf": confidence_threshold,
        "batch_index": batch_index,
    }


def _select_faces(
    faces: list[tuple[list[list[float]], dict[str, float | int]]],
    *,
    mode: str,
    max_faces: int,
) -> list[tuple[list[list[float]], dict[str, float | int]]]:
    ordered = sorted(
        faces,
        key=lambda item: int(item[1]["w"]) * int(item[1]["h"]),
        reverse=True,
    )
    if mode == "single_largest":
        return ordered[:1]
    return ordered[:max_faces]


def face_detect(
    image_bhwc: torch.Tensor,
    *,
    mode: str = "single_largest",
    max_faces: int = 1,
    confidence_threshold: float = 0.5,
) -> tuple[list[list[list[float]]], list[dict], int]:
    """Detect 5-point landmarks and padded face boxes from a BHWC RGB tensor batch."""

    image = _prepare_image(image_bhwc)
    mode = _validate_mode(mode)
    max_faces = _validate_int("max_faces", max_faces, lower=1, upper=10)
    confidence_threshold = _validate_confidence(confidence_threshold)

    mp, _, _, _, _ = _import_mediapipe()
    model_path = _ensure_model_file()
    requested_faces = max_faces if mode == "multi_top_k" else 10
    landmarker = _get_landmarker(str(model_path), confidence_threshold, requested_faces)

    batch, height, width, _ = image.shape
    all_landmarks: list[list[list[float]]] = []
    all_boxes: list[dict] = []

    for batch_index in range(batch):
        image_uint8 = (
            image[batch_index].detach().cpu().mul(255.0).round().clamp(0.0, 255.0).to(torch.uint8)
        )
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_uint8.numpy())
        result = landmarker.detect(mp_image)

        candidates: list[tuple[list[list[float]], dict[str, float | int]]] = []
        for face_landmarks in result.face_landmarks:
            points = torch.tensor(
                [[landmark.x * width, landmark.y * height] for landmark in face_landmarks],
                dtype=torch.float32,
            )
            five_points = points[list(_FIVE_POINT_INDICES)].tolist()
            bbox = _bbox_from_points(
                points,
                width=width,
                height=height,
                confidence_threshold=confidence_threshold,
                batch_index=batch_index,
            )
            candidates.append((five_points, bbox))

        for five_points, bbox in _select_faces(candidates, mode=mode, max_faces=max_faces):
            all_landmarks.append(five_points)
            all_boxes.append(bbox)

    return all_landmarks, all_boxes, len(all_landmarks)
