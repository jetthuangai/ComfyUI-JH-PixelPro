"""Edge-aware smoothing core math for BCHW RGB tensors."""

from __future__ import annotations

import logging
import math
from numbers import Real

import torch
from kornia.filters import bilateral_blur

logger = logging.getLogger(__name__)

_MEMORY_BUDGET_BYTES = 2_000_000_000
_TILE_SIZE = 512
_VALID_DEVICES = {"auto", "cpu", "cuda"}


def _validate_strength(strength: float) -> float:
    if isinstance(strength, bool) or not isinstance(strength, Real):
        raise ValueError("strength must be in [0.0, 1.0].")

    strength_float = float(strength)
    if not 0.0 <= strength_float <= 1.0:
        raise ValueError("strength must be in [0.0, 1.0].")
    return strength_float


def _validate_sigma_color(sigma_color: float) -> float:
    if isinstance(sigma_color, bool) or not isinstance(sigma_color, Real):
        raise ValueError("sigma_color must be > 0.")

    sigma_color_float = float(sigma_color)
    if sigma_color_float <= 0.0:
        raise ValueError("sigma_color must be > 0.")
    return sigma_color_float


def _validate_sigma_space(sigma_space: float) -> float:
    if isinstance(sigma_space, bool) or not isinstance(sigma_space, Real):
        raise ValueError(
            "sigma_space must be in [1.0, 8.0]. For wider smoothing, downsample image first."
        )

    sigma_space_float = float(sigma_space)
    if not 1.0 <= sigma_space_float <= 8.0:
        raise ValueError(
            "sigma_space must be in [1.0, 8.0]. For wider smoothing, downsample image first."
        )
    return sigma_space_float


def _validate_device(device: str) -> str:
    if not isinstance(device, str) or device not in _VALID_DEVICES:
        raise ValueError(f"device must be one of {tuple(sorted(_VALID_DEVICES))}.")
    return device


def _validate_tile_mode(tile_mode: bool) -> bool:
    if not isinstance(tile_mode, bool):
        raise ValueError("tile_mode must be a bool.")
    return tile_mode


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


def _prepare_mask(mask_bchw: torch.Tensor | None, image_bchw: torch.Tensor) -> torch.Tensor | None:
    if mask_bchw is None:
        return None

    if not isinstance(mask_bchw, torch.Tensor):
        raise TypeError("mask_bchw must be a torch.Tensor or None.")

    if mask_bchw.ndim != 4:
        raise ValueError(f"Expected BC1HW mask tensor, got shape {tuple(mask_bchw.shape)}.")

    if mask_bchw.shape[1] != 1:
        raise ValueError(f"Expected single-channel mask, got C={mask_bchw.shape[1]}.")

    image_batch, _, image_height, image_width = image_bchw.shape
    mask_batch, _, mask_height, mask_width = mask_bchw.shape

    if (mask_height, mask_width) != (image_height, image_width):
        raise ValueError(
            f"mask HxW ({mask_height}, {mask_width}) != image HxW ({image_height}, {image_width})."
        )

    if mask_batch not in (1, image_batch):
        raise ValueError(
            f"mask batch ({mask_batch}) must be 1 or equal to image batch ({image_batch})."
        )

    if mask_bchw.dtype == torch.bool:
        logger.warning("bool mask input cast to float32.")
        mask = mask_bchw.to(dtype=torch.float32)
    elif not torch.is_floating_point(mask_bchw):
        raise ValueError(f"Expected float mask tensor in [0,1], got {mask_bchw.dtype}.")
    elif mask_bchw.dtype != torch.float32:
        logger.warning("mask input cast to float32.")
        mask = mask_bchw.to(dtype=torch.float32)
    else:
        mask = mask_bchw

    value_min, value_max = torch.aminmax(mask.detach())
    if value_min.item() < 0.0 or value_max.item() > 1.0:
        logger.warning("mask input values outside [0,1]; clamped to [0,1].")
        mask = mask.clamp(0.0, 1.0)

    return mask


def _resolve_target_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA unavailable. Use 'auto' or 'cpu'.")

    return torch.device(device)


def _estimate_bilateral_memory(image_bchw: torch.Tensor, kernel_size: int) -> int:
    batch, channels, height, width = image_bchw.shape
    return batch * channels * height * width * kernel_size * kernel_size * 4


def _run_bilateral(
    image_bchw: torch.Tensor,
    kernel_size: int,
    sigma_color: float,
    sigma_space: float,
) -> torch.Tensor:
    return bilateral_blur(
        image_bchw,
        kernel_size=(kernel_size, kernel_size),
        sigma_color=sigma_color,
        sigma_space=(sigma_space, sigma_space),
        border_type="replicate",
        color_distance_type="l1",
    )


def _run_bilateral_tiled(
    image_bchw: torch.Tensor,
    kernel_size: int,
    sigma_color: float,
    sigma_space: float,
) -> torch.Tensor:
    batch, channels, height, width = image_bchw.shape
    pad = kernel_size // 2 + 1
    output = torch.empty(
        (batch, channels, height, width),
        dtype=image_bchw.dtype,
        device=image_bchw.device,
    )

    for tile_y in range(0, height, _TILE_SIZE):
        tile_height = min(_TILE_SIZE, height - tile_y)
        for tile_x in range(0, width, _TILE_SIZE):
            tile_width = min(_TILE_SIZE, width - tile_x)
            y0 = max(0, tile_y - pad)
            y1 = min(height, tile_y + tile_height + pad)
            x0 = max(0, tile_x - pad)
            x1 = min(width, tile_x + tile_width + pad)
            tile = image_bchw[:, :, y0:y1, x0:x1]
            tile_out = _run_bilateral(tile, kernel_size, sigma_color, sigma_space)

            valid_y0 = tile_y - y0
            valid_y1 = valid_y0 + tile_height
            valid_x0 = tile_x - x0
            valid_x1 = valid_x0 + tile_width
            output[:, :, tile_y : tile_y + tile_height, tile_x : tile_x + tile_width] = tile_out[
                :, :, valid_y0:valid_y1, valid_x0:valid_x1
            ]

    return output


def edge_aware_smooth(
    image_bchw: torch.Tensor,
    strength: float = 0.4,
    sigma_color: float = 0.1,
    sigma_space: float = 6.0,
    mask_bchw: torch.Tensor | None = None,
    *,
    device: str = "auto",
    tile_mode: bool = False,
) -> torch.Tensor:
    """Apply bilateral smoothing with optional strength blend and mask gating."""

    strength = _validate_strength(strength)
    sigma_color = _validate_sigma_color(sigma_color)
    sigma_space = _validate_sigma_space(sigma_space)
    device = _validate_device(device)
    tile_mode = _validate_tile_mode(tile_mode)
    image = _prepare_image(image_bchw)
    mask = _prepare_mask(mask_bchw, image)

    if sigma_color > 1.0:
        logger.warning("sigma_color scale is [0,1]; values > 1 will smooth across most edges.")

    kernel_size = 2 * math.ceil(3.0 * sigma_space) + 1
    required_bytes = _estimate_bilateral_memory(image, kernel_size)
    if not tile_mode and required_bytes > _MEMORY_BUDGET_BYTES:
        raise RuntimeError(
            f"ES would allocate ~{required_bytes / 1e9:.1f} GB for image "
            f"{image.shape[0]}x{image.shape[1]}x{image.shape[2]}x{image.shape[3]} "
            f"at sigma_space={sigma_space} (kernel={kernel_size}). "
            "Solutions: (a) enable tile_mode, (b) reduce sigma_space, "
            "(c) downsample image first."
        )

    target_device = _resolve_target_device(device)
    working_image = image.to(device=target_device)
    working_mask = None if mask is None else mask.to(device=target_device)

    if strength == 0.0:
        return working_image.clone().cpu()

    if working_mask is not None and working_mask.sum().item() == 0.0:
        return working_image.clone().cpu()

    if tile_mode:
        smoothed = _run_bilateral_tiled(working_image, kernel_size, sigma_color, sigma_space)
    else:
        smoothed = _run_bilateral(working_image, kernel_size, sigma_color, sigma_space)

    blended = strength * smoothed + (1.0 - strength) * working_image
    output = (
        blended
        if working_mask is None
        else working_mask * blended + (1.0 - working_mask) * working_image
    )
    return output.clamp(0.0, 1.0).cpu()
