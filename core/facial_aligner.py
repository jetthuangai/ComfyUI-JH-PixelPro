"""Landmark-based facial alignment for BCHW RGB tensors."""

from __future__ import annotations

from numbers import Real

import torch
from kornia.geometry.transform import invert_affine_transform, warp_affine

_VALID_TARGET_SIZES = {512, 768, 1024}


def _prepare_image(image_bchw: torch.Tensor) -> torch.Tensor:
    if not isinstance(image_bchw, torch.Tensor):
        raise TypeError("image_bchw must be a torch.Tensor.")

    if image_bchw.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(image_bchw.shape)}.")

    if image_bchw.shape[1] != 3:
        raise ValueError(f"Expected 3-channel RGB image, got C={image_bchw.shape[1]}.")

    if image_bchw.dtype != torch.float32:
        raise ValueError(f"Expected float32 image tensor, got {image_bchw.dtype}.")

    value_min, value_max = torch.aminmax(image_bchw.detach())
    if value_min.item() < 0.0 or value_max.item() > 1.0:
        raise ValueError("Expected input tensor range [0, 1].")

    return image_bchw


def _validate_target_size(target_size: int) -> int:
    if isinstance(target_size, bool) or not isinstance(target_size, int):
        raise ValueError(f"target_size must be one of {tuple(sorted(_VALID_TARGET_SIZES))}.")
    if target_size not in _VALID_TARGET_SIZES:
        raise ValueError(f"target_size must be one of {tuple(sorted(_VALID_TARGET_SIZES))}.")
    return target_size


def _validate_padding(padding: float) -> float:
    if isinstance(padding, bool) or not isinstance(padding, Real):
        raise ValueError("padding must be in [0.0, 0.5].")
    padding_float = float(padding)
    if not 0.0 <= padding_float <= 0.5:
        raise ValueError("padding must be in [0.0, 0.5].")
    return padding_float


def _as_landmark_tensor(
    landmarks: list[list[float]] | list[list[list[float]]] | torch.Tensor,
    *,
    image_bchw: torch.Tensor,
) -> torch.Tensor:
    if isinstance(landmarks, torch.Tensor):
        points = landmarks.to(dtype=torch.float32, device=image_bchw.device)
    else:
        points = torch.tensor(landmarks, dtype=torch.float32, device=image_bchw.device)

    if points.ndim == 2:
        if points.shape != (5, 2):
            raise ValueError(f"Expected 5x2 landmarks, got shape {tuple(points.shape)}.")
        points = points.unsqueeze(0)
    elif points.ndim == 3:
        if points.shape[-2:] != (5, 2):
            raise ValueError(f"Expected Bx5x2 landmarks, got shape {tuple(points.shape)}.")
    else:
        raise ValueError(f"Expected 5x2 or Bx5x2 landmarks, got shape {tuple(points.shape)}.")

    image_batch = image_bchw.shape[0]
    if points.shape[0] not in (1, image_batch):
        raise ValueError(
            "landmarks batch "
            f"({points.shape[0]}) must be 1 or equal to image batch ({image_batch})."
        )
    if points.shape[0] == 1 and image_batch > 1:
        points = points.expand(image_batch, -1, -1)
    return points


def _normalize_landmarks(landmarks: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    if torch.any(landmarks < 0):
        raise ValueError("Landmark coordinates must be non-negative.")

    scale = torch.tensor(
        [max(width - 1, 1), max(height - 1, 1)],
        dtype=landmarks.dtype,
        device=landmarks.device,
    ).view(1, 1, 2)
    is_normalized = landmarks.amax(dim=(1, 2)) <= 1.5
    normalized = landmarks.clone()
    if torch.any(~is_normalized):
        normalized[~is_normalized] = normalized[~is_normalized] / scale

    if torch.any(normalized > 1.5):
        raise ValueError("Landmarks look out of bounds after normalization.")
    return normalized.clamp(0.0, 1.0)


def canonical_landmarks(
    *,
    target_size: int = 1024,
    padding: float = 0.2,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return canonical 5-point landmarks in normalized coordinates."""

    _validate_target_size(target_size)
    padding = _validate_padding(padding)
    device = device or torch.device("cpu")
    scale = 1.0 - 2.0 * padding
    eye_half = 0.15 * scale
    mouth_half = 0.12 * scale
    points = torch.tensor(
        [
            [0.5 - eye_half, 0.40],
            [0.5 + eye_half, 0.40],
            [0.5, 0.55],
            [0.5 - mouth_half, 0.70],
            [0.5 + mouth_half, 0.70],
        ],
        dtype=dtype,
        device=device,
    )
    return points


def _canonical_landmarks_px(
    *, batch: int, target_size: int, padding: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    normalized = canonical_landmarks(
        target_size=target_size,
        padding=padding,
        device=device,
        dtype=dtype,
    ).unsqueeze(0)
    scale = torch.tensor(
        [max(target_size - 1, 1), max(target_size - 1, 1)],
        dtype=dtype,
        device=device,
    ).view(1, 1, 2)
    return normalized.expand(batch, -1, -1) * scale


def _umeyama_similarity(src_points: torch.Tensor, dst_points: torch.Tensor) -> torch.Tensor:
    """Estimate a similarity transform mapping src_points -> dst_points."""

    num_points = src_points.shape[1]
    src_mean = src_points.mean(dim=1, keepdim=True)
    dst_mean = dst_points.mean(dim=1, keepdim=True)
    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean

    covariance = dst_centered.transpose(1, 2) @ src_centered / float(num_points)
    u_matrix, singular_values, v_matrix_t = torch.linalg.svd(covariance)

    correction = torch.eye(2, dtype=src_points.dtype, device=src_points.device).unsqueeze(0).repeat(
        src_points.shape[0], 1, 1
    )
    determinant = torch.linalg.det(u_matrix @ v_matrix_t)
    correction[:, 1, 1] = torch.where(determinant < 0.0, -1.0, 1.0)

    rotation = u_matrix @ correction @ v_matrix_t
    src_variance = (src_centered**2).sum(dim=(1, 2)) / float(num_points)
    scale = (
        (singular_values * correction.diagonal(dim1=-2, dim2=-1)).sum(dim=1)
        / src_variance.clamp_min(1e-8)
    )
    translation = dst_mean.squeeze(1) - scale.unsqueeze(1) * (
        rotation @ src_mean.squeeze(1).unsqueeze(-1)
    ).squeeze(-1)

    transform = torch.zeros(
        (src_points.shape[0], 2, 3),
        dtype=src_points.dtype,
        device=src_points.device,
    )
    transform[:, :, :2] = scale.view(-1, 1, 1) * rotation
    transform[:, :, 2] = translation
    return transform


def _estimate_similarity_transform(
    landmarks: torch.Tensor,
    *,
    image_size: tuple[int, int],
    target_size: int,
    padding: float,
) -> torch.Tensor:
    height, width = image_size
    scale = torch.tensor(
        [max(width - 1, 1), max(height - 1, 1)],
        dtype=landmarks.dtype,
        device=landmarks.device,
    ).view(1, 1, 2)
    src_points = landmarks * scale
    dst_points = _canonical_landmarks_px(
        batch=landmarks.shape[0],
        target_size=target_size,
        padding=padding,
        device=landmarks.device,
        dtype=landmarks.dtype,
    )
    return _umeyama_similarity(src_points, dst_points)


def _to_homogeneous(affine_matrix: torch.Tensor) -> torch.Tensor:
    batch = affine_matrix.shape[0]
    output = (
        torch.eye(3, dtype=affine_matrix.dtype, device=affine_matrix.device)
        .unsqueeze(0)
        .repeat(batch, 1, 1)
    )
    output[:, :2, :] = affine_matrix
    return output


def facial_align(
    image_bchw: torch.Tensor,
    landmarks: list[list[float]] | list[list[list[float]]] | torch.Tensor,
    *,
    target_size: int = 1024,
    padding: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align faces to a canonical frame and return the inverse transform."""

    image = _prepare_image(image_bchw)
    target_size = _validate_target_size(target_size)
    padding = _validate_padding(padding)

    points = _as_landmark_tensor(landmarks, image_bchw=image)
    points = _normalize_landmarks(points, height=image.shape[-2], width=image.shape[-1])
    affine = _estimate_similarity_transform(
        points,
        image_size=(image.shape[-2], image.shape[-1]),
        target_size=target_size,
        padding=padding,
    )
    aligned = warp_affine(
        image,
        affine,
        dsize=(target_size, target_size),
        align_corners=False,
    ).clamp(0.0, 1.0)
    inverse_affine = invert_affine_transform(affine)
    return aligned, _to_homogeneous(inverse_affine)


def rotation_degrees_from_affine(affine_matrix: torch.Tensor) -> torch.Tensor:
    """Extract rotation degrees from a Bx2x3 affine matrix."""

    return torch.rad2deg(torch.atan2(affine_matrix[:, 1, 0], affine_matrix[:, 0, 0]))
