"""Brown-Conrady lens distortion correction and simulation for BCHW RGB tensors."""

from __future__ import annotations

import logging
from numbers import Real

import cv2
import torch
from kornia.geometry.calibration import undistort_image, undistort_points
from torch.nn.functional import grid_sample

logger = logging.getLogger(__name__)

_VALID_DIRECTIONS = {"forward", "inverse"}


def _validate_direction(direction: str) -> str:
    if not isinstance(direction, str) or direction not in _VALID_DIRECTIONS:
        raise ValueError(f"direction must be one of {tuple(sorted(_VALID_DIRECTIONS))}.")
    return direction


def _validate_range(name: str, value: float, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    value_float = float(value)
    if not lower <= value_float <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_float


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
        logger.warning("image input values outside [0,1]; clamped to [0,1].")
        return image_bchw.clamp(0.0, 1.0)

    return image_bchw


def _camera_matrix(*, batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    focal = float(max(height, width))
    matrix = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch, 1, 1)
    matrix[:, 0, 0] = focal
    matrix[:, 1, 1] = focal
    matrix[:, 0, 2] = width / 2.0
    matrix[:, 1, 2] = height / 2.0
    return matrix


def _dist_coeffs(
    *,
    batch: int,
    device: torch.device,
    k1: float,
    k2: float,
    k3: float,
    p1: float,
    p2: float,
) -> torch.Tensor:
    values = torch.tensor([k1, k2, p1, p2, k3], dtype=torch.float32, device=device)
    return values.unsqueeze(0).repeat(batch, 1)


def _pixel_grid(
    *, height: int, width: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    y_coords = torch.arange(height, dtype=torch.float32, device=device)
    x_coords = torch.arange(width, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    return xx, yy


def _distort_normalized(
    x_norm: torch.Tensor,
    y_norm: torch.Tensor,
    *,
    k1: float,
    k2: float,
    k3: float,
    p1: float,
    p2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r2 = (x_norm * x_norm) + (y_norm * y_norm)
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + (k1 * r2) + (k2 * r4) + (k3 * r6)
    x_tangent = (2.0 * p1 * x_norm * y_norm) + (p2 * (r2 + (2.0 * x_norm * x_norm)))
    y_tangent = (p1 * (r2 + (2.0 * y_norm * y_norm))) + (2.0 * p2 * x_norm * y_norm)
    return (x_norm * radial) + x_tangent, (y_norm * radial) + y_tangent


def _forward_grid(
    *,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    k1: float,
    k2: float,
    k3: float,
    p1: float,
    p2: float,
    iterations: int = 8,
) -> torch.Tensor:
    xx, yy = _pixel_grid(height=height, width=width, device=device)
    points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1).unsqueeze(0).repeat(batch, 1, 1)
    camera = _camera_matrix(batch=batch, height=height, width=width, device=device)
    dist = _dist_coeffs(batch=batch, device=device, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2)
    source_points = undistort_points(points, camera, dist, new_K=camera, num_iters=iterations)
    source_points = source_points.view(batch, height, width, 2)
    grid_x = ((2.0 * (source_points[..., 0] + 0.5)) / width) - 1.0
    grid_y = ((2.0 * (source_points[..., 1] + 0.5)) / height) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid


def _inverse_fallback_cv2(
    image_bchw: torch.Tensor,
    *,
    k1: float,
    k2: float,
    k3: float,
    p1: float,
    p2: float,
) -> torch.Tensor:
    batch, _, height, width = image_bchw.shape
    focal = float(max(height, width))
    camera = torch.tensor(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    ).numpy()
    dist = torch.tensor([k1, k2, p1, p2, k3], dtype=torch.float32).numpy()

    outputs: list[torch.Tensor] = []
    for index in range(batch):
        image_hwc = image_bchw[index].detach().cpu().permute(1, 2, 0).numpy()
        map_x, map_y = cv2.initUndistortRectifyMap(
            camera,
            dist,
            None,
            camera,
            (width, height),
            cv2.CV_32FC1,
        )
        corrected = cv2.remap(
            image_hwc,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        outputs.append(torch.from_numpy(corrected).permute(2, 0, 1))

    return torch.stack(outputs, dim=0).to(device=image_bchw.device, dtype=torch.float32)


def lens_distortion(
    image_bchw: torch.Tensor,
    *,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
    direction: str = "inverse",
) -> torch.Tensor:
    """Apply Brown-Conrady lens distortion correction or simulation."""

    image = _prepare_image(image_bchw)
    k1 = _validate_range("k1", k1, lower=-1.0, upper=1.0)
    k2 = _validate_range("k2", k2, lower=-1.0, upper=1.0)
    k3 = _validate_range("k3", k3, lower=-1.0, upper=1.0)
    p1 = _validate_range("p1", p1, lower=-0.1, upper=0.1)
    p2 = _validate_range("p2", p2, lower=-0.1, upper=0.1)
    direction = _validate_direction(direction)

    batch, _, height, width = image.shape
    if direction == "inverse":
        if image.device.type == "cpu":
            return _inverse_fallback_cv2(image, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2).clamp(0.0, 1.0)

        camera = _camera_matrix(batch=batch, height=height, width=width, device=image.device)
        dist = _dist_coeffs(
            batch=batch,
            device=image.device,
            k1=k1,
            k2=k2,
            k3=k3,
            p1=p1,
            p2=p2,
        )
        try:
            return undistort_image(image, camera, dist).clamp(0.0, 1.0)
        except Exception as exc:
            logger.warning("kornia undistort_image failed; falling back to cv2.remap: %s", exc)
            return _inverse_fallback_cv2(image, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2).clamp(0.0, 1.0)

    grid = _forward_grid(
        batch=batch,
        height=height,
        width=width,
        device=image.device,
        k1=k1,
        k2=k2,
        k3=k3,
        p1=p1,
        p2=p2,
    )
    return grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).clamp(0.0, 1.0)
