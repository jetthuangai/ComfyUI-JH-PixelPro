"""Face-pipeline v2 helpers: dense landmarks, Delaunay warp, and beauty blend."""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
import torch
from kornia.filters import gaussian_blur2d

from .face_detect import _ensure_model_file, _get_landmarker, _import_mediapipe

_LANDMARK_COUNT = 468


def _prepare_image(name: str, image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (B,H,W,3), got {tuple(image.shape)}.")
    if not torch.is_floating_point(image):
        raise ValueError(f"{name} must be floating point, got {image.dtype}.")
    return image.to(dtype=torch.float32).clamp(0.0, 1.0)


def _prepare_landmarks(name: str, points: torch.Tensor, *, batch: int) -> torch.Tensor:
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if points.ndim != 3 or points.shape[1:] != (_LANDMARK_COUNT, 2):
        raise ValueError(f"{name} must have shape (B,468,2), got {tuple(points.shape)}.")
    if points.shape[0] not in (1, batch):
        raise ValueError(f"{name} batch must be 1 or {batch}, got {points.shape[0]}.")
    prepared = points.to(dtype=torch.float32)
    if prepared.shape[0] == 1 and batch > 1:
        prepared = prepared.expand(batch, -1, -1)
    return prepared


def _validate_max_faces(max_num_faces: int) -> int:
    if isinstance(max_num_faces, bool) or not isinstance(max_num_faces, Integral):
        raise ValueError("max_num_faces must be an integer in [1, 10].")
    max_faces = int(max_num_faces)
    if not 1 <= max_faces <= 10:
        raise ValueError("max_num_faces must be an integer in [1, 10].")
    return max_faces


def _validate_confidence(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be in [0, 1].")
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return confidence


def _import_face_landmarker():
    try:
        mp, _, _, _, _ = _import_mediapipe()
    except RuntimeError as exc:
        raise ImportError(
            "MediaPipe FaceLandmarker is required for extract_landmarks(); "
            "install `mediapipe>=0.10.0`."
        ) from exc
    return mp


def _import_cv2_scipy():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for face_warp_delaunay(); install `opencv-python-headless`."
        ) from exc
    try:
        from scipy.spatial import Delaunay
    except ImportError as exc:
        raise ImportError(
            "SciPy is required for face_warp_delaunay(); install `scipy>=1.10`."
        ) from exc
    return cv2, Delaunay


def _gaussian_kernel_size(sigma: float) -> int:
    return max(3, 2 * math.ceil(3.0 * sigma) + 1)


def _to_pixel_landmarks(points: torch.Tensor, *, width: int, height: int) -> np.ndarray:
    cpu = points.detach().to("cpu", torch.float32)
    finite = torch.isfinite(cpu).all(dim=-1)
    if not finite.any():
        return np.empty((0, 2), dtype=np.float32)

    valid = cpu[finite]
    if valid.numel() == 0:
        return np.empty((0, 2), dtype=np.float32)

    if valid.max().item() <= 1.5 and valid.min().item() >= -0.5:
        scale = torch.tensor([max(width - 1, 1), max(height - 1, 1)], dtype=torch.float32)
        cpu = cpu * scale
    return cpu.numpy().astype(np.float32, copy=False)


def extract_landmarks(
    image: torch.Tensor,
    *,
    max_num_faces: int = 1,
    min_detection_confidence: float = 0.5,
    refine_landmarks: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract MediaPipe face landmarks as a padded ``(B,max_num_faces,468,2)`` tensor."""

    del refine_landmarks  # Tasks FaceLandmarker always returns dense landmarks; we keep the API.
    image_prepared = _prepare_image("image", image)
    max_num_faces = _validate_max_faces(max_num_faces)
    min_detection_confidence = _validate_confidence(
        "min_detection_confidence",
        min_detection_confidence,
    )

    mp = _import_face_landmarker()
    model_path = _ensure_model_file()
    landmarker = _get_landmarker(str(model_path), min_detection_confidence, max_num_faces)

    batch, height, width, _ = image_prepared.shape
    landmarks = torch.full(
        (batch, max_num_faces, _LANDMARK_COUNT, 2),
        float("nan"),
        dtype=torch.float32,
        device=image.device,
    )
    visibility = torch.zeros((batch, max_num_faces), dtype=torch.float32, device=image.device)

    for batch_index in range(batch):
        image_uint8 = (
            image_prepared[batch_index]
            .detach()
            .to("cpu")
            .mul(255.0)
            .round()
            .clamp(0.0, 255.0)
            .to(torch.uint8)
            .numpy()
        )
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_uint8)
        result = landmarker.detect(mp_image)

        for face_index, face_landmarks in enumerate(result.face_landmarks[:max_num_faces]):
            coords = torch.tensor(
                [[landmark.x, landmark.y] for landmark in face_landmarks[:_LANDMARK_COUNT]],
                dtype=torch.float32,
                device=image.device,
            )
            landmarks[batch_index, face_index] = coords
            visibility[batch_index, face_index] = 1.0

    return landmarks, visibility


def face_warp_delaunay(
    image: torch.Tensor,
    src_landmarks: torch.Tensor,
    dst_landmarks: torch.Tensor,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Warp an image by triangulating source landmarks and mapping them to destination landmarks."""

    cv2, delaunay_cls = _import_cv2_scipy()
    image_prepared = _prepare_image("image", image).to(device=device)
    batch, height, width, _ = image_prepared.shape
    src_prepared = _prepare_landmarks("src_landmarks", src_landmarks, batch=batch)
    dst_prepared = _prepare_landmarks("dst_landmarks", dst_landmarks, batch=batch)

    outputs: list[torch.Tensor] = []
    for batch_index in range(batch):
        image_np = image_prepared[batch_index].detach().to("cpu", torch.float32).numpy()
        src_points = _to_pixel_landmarks(src_prepared[batch_index], width=width, height=height)
        dst_points = _to_pixel_landmarks(dst_prepared[batch_index], width=width, height=height)
        valid_mask = np.isfinite(src_points).all(axis=1) & np.isfinite(dst_points).all(axis=1)
        src_valid = src_points[valid_mask]
        dst_valid = dst_points[valid_mask]

        if src_valid.shape[0] < 3 or dst_valid.shape[0] < 3:
            outputs.append(torch.from_numpy(image_np.copy()))
            continue

        triangulation = delaunay_cls(src_valid)
        composite = np.zeros_like(image_np, dtype=np.float32)
        coverage = np.zeros((height, width), dtype=np.float32)

        for simplex in triangulation.simplices:
            src_tri = src_valid[simplex].astype(np.float32)
            dst_tri = dst_valid[simplex].astype(np.float32)
            if np.linalg.det(np.stack([src_tri[1] - src_tri[0], src_tri[2] - src_tri[0]])) == 0.0:
                continue

            src_rect = cv2.boundingRect(src_tri)
            dst_rect = cv2.boundingRect(dst_tri)
            if src_rect[2] <= 0 or src_rect[3] <= 0 or dst_rect[2] <= 0 or dst_rect[3] <= 0:
                continue

            src_offset = src_tri - np.array([src_rect[0], src_rect[1]], dtype=np.float32)
            dst_offset = dst_tri - np.array([dst_rect[0], dst_rect[1]], dtype=np.float32)

            src_patch = image_np[
                src_rect[1] : src_rect[1] + src_rect[3],
                src_rect[0] : src_rect[0] + src_rect[2],
            ]
            matrix = cv2.getAffineTransform(src_offset, dst_offset)
            warped_patch = cv2.warpAffine(
                src_patch,
                matrix,
                (dst_rect[2], dst_rect[3]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            tri_mask = np.zeros((dst_rect[3], dst_rect[2]), dtype=np.float32)
            cv2.fillConvexPoly(
                tri_mask,
                np.round(dst_offset).astype(np.int32),
                1.0,
                lineType=cv2.LINE_AA,
            )
            mask_3c = tri_mask[..., None]

            y0, x0 = dst_rect[1], dst_rect[0]
            y1, x1 = y0 + dst_rect[3], x0 + dst_rect[2]
            composite[y0:y1, x0:x1] = (
                composite[y0:y1, x0:x1] * (1.0 - mask_3c) + warped_patch * mask_3c
            )
            coverage[y0:y1, x0:x1] = np.maximum(coverage[y0:y1, x0:x1], tri_mask)

        output_np = composite * coverage[..., None] + image_np * (1.0 - coverage[..., None])
        outputs.append(torch.from_numpy(np.clip(output_np, 0.0, 1.0)))

    return torch.stack(outputs, dim=0).to(device=image.device, dtype=image.dtype)


def beauty_blend(
    base: torch.Tensor,
    retouched: torch.Tensor,
    mask: torch.Tensor,
    *,
    strength: float = 1.0,
    feather: int = 0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Blend a retouched image over a base image with optional feathered mask."""

    base_prepared = _prepare_image("base", base).to(device=device)
    retouched_prepared = _prepare_image("retouched", retouched).to(device=base_prepared.device)
    if base_prepared.shape != retouched_prepared.shape:
        raise ValueError("base and retouched must have the same shape.")
    if mask.ndim != 3 or mask.shape != base_prepared.shape[:3]:
        raise ValueError(
            f"mask must have shape {base_prepared.shape[:3]}, got {tuple(mask.shape)}."
        )
    if not torch.is_floating_point(mask) and mask.dtype is not torch.bool:
        raise ValueError(f"mask must be float or bool, got {mask.dtype}.")

    strength = _validate_confidence("strength", strength)
    if isinstance(feather, bool) or not isinstance(feather, Integral) or int(feather) < 0:
        raise ValueError("feather must be an integer >= 0.")
    feather = int(feather)

    mask_prepared = mask.to(device=base_prepared.device, dtype=torch.float32).clamp(0.0, 1.0)
    if feather > 0:
        sigma = max(float(feather) / 3.0, 1e-3)
        kernel_size = _gaussian_kernel_size(sigma)
        blurred = gaussian_blur2d(
            mask_prepared.unsqueeze(1),
            kernel_size=(kernel_size, kernel_size),
            sigma=(sigma, sigma),
            border_type="replicate",
        ).squeeze(1)
    else:
        blurred = mask_prepared

    mask_eff = (blurred * strength).clamp(0.0, 1.0).unsqueeze(-1)
    blended = base_prepared * (1.0 - mask_eff) + retouched_prepared * mask_eff
    return blended.clamp(0.0, 1.0).to(dtype=base.dtype, device=base.device)
