"""Reinhard LAB color transfer for BCHW RGB tensors."""

from __future__ import annotations

import logging
from numbers import Real

import torch
from kornia.color import lab_to_rgb, rgb_to_lab

logger = logging.getLogger(__name__)

_VALID_CHANNELS = {"ab", "lab"}
_VALID_METHODS = ("stats", "histogram")
_HIST_BINS = 4096


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


def _prepare_reference_for_stats(
    image_reference_bchw: torch.Tensor,
    *,
    target_batch: int,
    device: torch.device,
) -> torch.Tensor:
    reference = _prepare_image("image_reference_bchw", image_reference_bchw).to(device=device)

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


def _validate_method(method: str) -> str:
    if not isinstance(method, str) or method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}.")
    return method


def _weighted_cdf(
    values_nk: torch.Tensor, weights_nk: torch.Tensor, lo: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Normalised weighted CDF at the _HIST_BINS + 1 bin edges, shape (N, bins + 1)."""

    pos = ((values_nk - lo) * scale).clamp_(0.0, _HIST_BINS - 1e-3)
    hist = torch.zeros(values_nk.shape[0], _HIST_BINS, device=values_nk.device)
    hist.scatter_add_(1, pos.long(), weights_nk)
    cdf = torch.cumsum(hist, dim=1)
    cdf = cdf / cdf[:, -1:].clamp_min(1e-12)
    return torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=1)


def _histogram_match(
    target_bchw: torch.Tensor,
    reference_bchw: torch.Tensor,
    target_mask: torch.Tensor | None,
    reference_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Per-channel quantile mapping of target onto the reference (masked) distribution."""

    batch, channels = target_bchw.shape[:2]
    t = target_bchw.reshape(batch * channels, -1)
    r = reference_bchw.reshape(batch * channels, -1)

    def weights(mask: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return torch.ones_like(like)
        return mask.reshape(batch, 1, -1).expand(batch, channels, -1).reshape(batch * channels, -1)

    wt, wr = weights(target_mask, t), weights(reference_mask, r)
    lo = torch.minimum(t.amin(dim=1, keepdim=True), r.amin(dim=1, keepdim=True))
    hi = torch.maximum(t.amax(dim=1, keepdim=True), r.amax(dim=1, keepdim=True))
    scale = _HIST_BINS / (hi - lo).clamp_min(1e-6)

    cdf_t = _weighted_cdf(t, wt, lo, scale)
    cdf_r = _weighted_cdf(r, wr, lo, scale)

    # Target quantile of every pixel (linear inside a bin).
    pos = ((t - lo) * scale).clamp(0.0, _HIST_BINS - 1e-3)
    idx = pos.long()
    frac = pos - idx
    c0 = cdf_t.gather(1, idx)
    c1 = cdf_t.gather(1, idx + 1)
    p = c0 + frac * (c1 - c0)

    # Inverse reference CDF at that quantile.
    # right=True: p == 0 (target at its global minimum) lands after the leading empty reference
    # bins, i.e. on the reference minimum instead of the shared range start ``lo``.
    j = torch.searchsorted(cdf_r.contiguous(), p.contiguous(), right=True).clamp_(1, _HIST_BINS)
    r0 = cdf_r.gather(1, j - 1)
    r1 = cdf_r.gather(1, j)
    tt = ((p - r0) / (r1 - r0).clamp_min(1e-12)).clamp_(0.0, 1.0)
    out = lo + (j - 1 + tt) / scale
    return out.reshape(target_bchw.shape)


def _transfer(
    target_selected: torch.Tensor,
    reference_selected: torch.Tensor,
    target_mask: torch.Tensor | None,
    reference_mask: torch.Tensor | None,
    method: str,
) -> torch.Tensor:
    if method == "histogram":
        return _histogram_match(target_selected, reference_selected, target_mask, reference_mask)
    target_mean, target_std = _masked_mean_std(target_selected, target_mask)
    reference_mean, reference_std = _masked_mean_std(reference_selected, reference_mask)
    return (target_selected - target_mean) * (
        reference_std / target_std.clamp_min(1e-6)
    ) + reference_mean


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
    variance = (((values_bchw - mean) ** 2) * mask_b1hw).sum(
        dim=(-1, -2), keepdim=True
    ) / weight_sum
    return mean, torch.sqrt(variance + 1e-6)


def color_matcher(
    image_target_bchw: torch.Tensor,
    image_reference_bchw: torch.Tensor,
    *,
    channels: str = "ab",
    strength: float = 1.0,
    mask: torch.Tensor | None = None,
    method: str = "stats",
) -> torch.Tensor:
    """Match target RGB colors to a reference image in LAB space.

    ``method`` is ``"stats"`` (Reinhard mean/std) or ``"histogram"`` (per-channel
    quantile mapping so the target CDF matches the reference CDF). ``mask`` weights
    both target and reference statistics.
    """

    target = _prepare_image("image_target_bchw", image_target_bchw)
    channels = _validate_channels(channels)
    method = _validate_method(method)
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

    matched_selected = _transfer(target_selected, reference_selected, mask_b1hw, mask_b1hw, method)

    matched_lab = target_lab.clone()
    matched_lab[:, channel_slice] = matched_selected

    matched_rgb = lab_to_rgb(matched_lab)
    matched_rgb = torch.nan_to_num(matched_rgb, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    output = ((1.0 - strength) * target) + (strength * matched_rgb)
    return output.clamp(0.0, 1.0)


def color_matcher_region(
    image_target_bchw: torch.Tensor,
    image_reference_bchw: torch.Tensor,
    *,
    channels: str = "ab",
    strength: float = 1.0,
    target_mask: torch.Tensor | None = None,
    reference_mask: torch.Tensor | None = None,
    method: str = "stats",
) -> torch.Tensor:
    """Match target colors to reference stats, allowing different image sizes.

    ``method`` is ``"stats"`` (Reinhard) or ``"histogram"`` (quantile mapping).

    ``target_mask`` controls both target-side statistics and where the result is
    applied. ``reference_mask`` controls reference-side statistics. Missing masks
    mean the full corresponding image is used.
    """

    target = _prepare_image("image_target_bchw", image_target_bchw)
    channels = _validate_channels(channels)
    method = _validate_method(method)
    strength = _validate_strength(strength)
    reference = _prepare_reference_for_stats(
        image_reference_bchw,
        target_batch=target.shape[0],
        device=target.device,
    )
    target_mask_b1hw = _prepare_mask(target_mask, image_shape=target.shape, device=target.device)
    reference_mask_b1hw = _prepare_mask(
        reference_mask,
        image_shape=reference.shape,
        device=target.device,
    )

    if strength == 0.0:
        return target

    target_lab = rgb_to_lab(target)
    reference_lab = rgb_to_lab(reference)

    channel_slice = slice(1, 3) if channels == "ab" else slice(0, 3)
    target_selected = target_lab[:, channel_slice]
    reference_selected = reference_lab[:, channel_slice]

    matched_selected = _transfer(
        target_selected, reference_selected, target_mask_b1hw, reference_mask_b1hw, method
    )

    matched_lab = target_lab.clone()
    matched_lab[:, channel_slice] = matched_selected

    matched_rgb = lab_to_rgb(matched_lab)
    matched_rgb = torch.nan_to_num(matched_rgb, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    matched_output = ((1.0 - strength) * target) + (strength * matched_rgb)
    matched_output = matched_output.clamp(0.0, 1.0)

    if target_mask_b1hw is None:
        return matched_output

    return torch.lerp(target, matched_output, target_mask_b1hw).clamp(0.0, 1.0)
