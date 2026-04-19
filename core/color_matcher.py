"""Reinhard LAB color transfer for BCHW RGB tensors."""

from __future__ import annotations

import logging
from numbers import Real

import torch
from kornia.color import lab_to_rgb, rgb_to_lab

logger = logging.getLogger(__name__)

_VALID_CHANNELS = {"ab", "lab"}


def _prepare_image(name: str, image_bchw: torch.Tensor) -> torch.Tensor:
    if not isinstance(image_bchw, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")

    if image_bchw.ndim != 4:
        raise ValueError(f"Expected BCHW tensor for {name}, got shape {tuple(image_bchw.shape)}.")

    if image_bchw.shape[1] != 3:
        raise ValueError(f"Expected 3-channel RGB image for {name}, got C={image_bchw.shape[1]}.")

    if image_bchw.dtype != torch.float32:
        raise ValueError(f"Expected float32 image tensor for {name}, got {image_bchw.dtype}.")

    value_min, value_max = torch.aminmax(image_bchw.detach())
    if value_min.item() < 0.0 or value_max.item() > 1.0:
        logger.warning("%s values outside [0,1]; clamped to [0,1].", name)
        return image_bchw.clamp(0.0, 1.0)

    return image_bchw


def _prepare_reference(
    image_reference_bchw: torch.Tensor,
    *,
    target_shape: torch.Size,
    device: torch.device,
) -> torch.Tensor:
    reference = _prepare_image("image_reference_bchw", image_reference_bchw).to(device=device)
    target_batch, _, target_height, target_width = target_shape

    if reference.shape[-2:] != (target_height, target_width):
        raise ValueError(
            "image_target_bchw and image_reference_bchw must have the same HxW, "
            f"got {(target_height, target_width)} and {tuple(reference.shape[-2:])}."
        )

    if reference.shape[0] not in (1, target_batch):
        raise ValueError(
            "image_reference_bchw batch "
            f"({reference.shape[0]}) must be 1 or equal to target batch ({target_batch})."
        )

    if reference.shape[0] == 1 and target_batch > 1:
        reference = reference.expand(target_batch, -1, -1, -1)
    return reference


def _validate_channels(channels: str) -> str:
    if not isinstance(channels, str) or channels not in _VALID_CHANNELS:
        raise ValueError(f"channels must be one of {tuple(sorted(_VALID_CHANNELS))}.")
    return channels


def _validate_strength(strength: float) -> float:
    if isinstance(strength, bool) or not isinstance(strength, Real):
        raise ValueError("strength must be in [0.0, 1.0].")

    strength_float = float(strength)
    if not 0.0 <= strength_float <= 1.0:
        raise ValueError("strength must be in [0.0, 1.0].")
    return strength_float


def _prepare_mask(
    mask: torch.Tensor | None,
    *,
    image_shape: torch.Size,
    device: torch.device,
) -> torch.Tensor | None:
    if mask is None:
        return None

    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor or None.")

    if mask.ndim == 3:
        prepared = mask.unsqueeze(1)
    elif mask.ndim == 4 and mask.shape[1] == 1:
        prepared = mask
    else:
        raise ValueError(f"Expected BHW or BC1HW mask, got shape {tuple(mask.shape)}.")

    batch, _, height, width = image_shape
    if prepared.shape[-2:] != (height, width):
        raise ValueError(
            f"mask HxW {tuple(prepared.shape[-2:])} must match image HxW {(height, width)}."
        )

    if prepared.shape[0] not in (1, batch):
        raise ValueError(
            f"mask batch ({prepared.shape[0]}) must be 1 or equal to image batch ({batch})."
        )

    if prepared.dtype == torch.bool:
        logger.warning("mask values cast from bool to float32.")
        prepared = prepared.to(dtype=torch.float32)
    elif not torch.is_floating_point(prepared):
        raise ValueError(f"Expected float mask tensor in [0,1], got {prepared.dtype}.")
    elif prepared.dtype != torch.float32:
        logger.warning("mask values cast to float32.")
        prepared = prepared.to(dtype=torch.float32)

    prepared = prepared.to(device=device)
    value_min, value_max = torch.aminmax(prepared.detach())
    if value_min.item() < 0.0 or value_max.item() > 1.0:
        logger.warning("mask values outside [0,1]; clamped to [0,1].")
        prepared = prepared.clamp(0.0, 1.0)

    if prepared.shape[0] == 1 and batch > 1:
        prepared = prepared.expand(batch, -1, -1, -1)

    weight_sum = prepared.sum(dim=(-1, -2), keepdim=True)
    if torch.any(weight_sum <= 0.0):
        raise ValueError("mask must contain at least one positive pixel per batch item.")

    return prepared


def _masked_mean_std(
    values_bchw: torch.Tensor,
    mask_b1hw: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask_b1hw is None:
        mean = values_bchw.mean(dim=(-1, -2), keepdim=True)
        variance = ((values_bchw - mean) ** 2).mean(dim=(-1, -2), keepdim=True)
        return mean, torch.sqrt(variance + 1e-6)

    weight_sum = mask_b1hw.sum(dim=(-1, -2), keepdim=True)
    mean = (values_bchw * mask_b1hw).sum(dim=(-1, -2), keepdim=True) / weight_sum
    variance = (
        (((values_bchw - mean) ** 2) * mask_b1hw).sum(dim=(-1, -2), keepdim=True) / weight_sum
    )
    return mean, torch.sqrt(variance + 1e-6)


def color_matcher(
    image_target_bchw: torch.Tensor,
    image_reference_bchw: torch.Tensor,
    *,
    channels: str = "ab",
    strength: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match target RGB colors to a reference image in LAB space."""

    target = _prepare_image("image_target_bchw", image_target_bchw)
    channels = _validate_channels(channels)
    strength = _validate_strength(strength)
    reference = _prepare_reference(
        image_reference_bchw,
        target_shape=target.shape,
        device=target.device,
    )
    mask_b1hw = _prepare_mask(mask, image_shape=target.shape, device=target.device)

    if strength == 0.0:
        return target

    target_lab = rgb_to_lab(target)
    reference_lab = rgb_to_lab(reference)

    channel_slice = slice(1, 3) if channels == "ab" else slice(0, 3)
    target_selected = target_lab[:, channel_slice]
    reference_selected = reference_lab[:, channel_slice]

    target_mean, target_std = _masked_mean_std(target_selected, mask_b1hw)
    reference_mean, reference_std = _masked_mean_std(reference_selected, mask_b1hw)
    matched_selected = (
        (target_selected - target_mean)
        * (reference_std / target_std.clamp_min(1e-6))
        + reference_mean
    )

    matched_lab = target_lab.clone()
    matched_lab[:, channel_slice] = matched_selected

    matched_rgb = lab_to_rgb(matched_lab)
    matched_rgb = torch.nan_to_num(matched_rgb, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    output = ((1.0 - strength) * target) + (strength * matched_rgb)
    return output.clamp(0.0, 1.0)
