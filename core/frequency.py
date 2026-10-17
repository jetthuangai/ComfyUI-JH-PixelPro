"""Frequency separation core math for BCHW image tensors."""

from __future__ import annotations

import logging

import torch
from kornia.filters import gaussian_blur2d

logger = logging.getLogger(__name__)

_PRECISION_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
}


def _resolve_precision_dtype(precision: str) -> torch.dtype:
    try:
        return _PRECISION_DTYPES[precision]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported precision {precision!r}. Expected one of {tuple(_PRECISION_DTYPES)}."
        ) from exc


def _validate_img_bchw(img_bchw: torch.Tensor) -> None:
    if not isinstance(img_bchw, torch.Tensor):
        raise TypeError("img_bchw must be a torch.Tensor.")

    if img_bchw.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(img_bchw.shape)}.")

    channels = img_bchw.shape[1]
    if channels != 3:
        suffix = " (use SplitAlpha node first)" if channels == 4 else ""
        raise ValueError(f"Expected 3-channel RGB image, got {channels}{suffix}.")

    value_min, value_max = torch.aminmax(img_bchw.detach())
    if value_min.item() < 0.0 or value_max.item() > 1.0:
        raise ValueError("Expected input tensor range [0, 1].")


def frequency_separation(
    img_bchw: torch.Tensor,
    radius: int,
    sigma: float = 0.0,
    precision: str = "float32",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a BCHW RGB tensor into low/high frequency components.

    Args:
        img_bchw: Input RGB image tensor in BCHW layout with values in `[0, 1]`.
        radius: Gaussian blur radius in pixels. Must be at least `1`.
        sigma: Gaussian sigma override. Use `0.0` to auto-compute `radius / 2.0`.
        precision: Compute dtype, either `"float32"` or `"float16"`.

    Returns:
        A tuple `(low, high)` in BCHW layout, where `low` is the blurred image and
        `high` is the linear residual `img_bchw - low`.

    Raises:
        TypeError: If `img_bchw` is not a tensor.
        ValueError: If the tensor shape, channel count, range, or parameters are invalid.
    """

    _validate_img_bchw(img_bchw)

    if radius < 1:
        raise ValueError("radius must be >= 1.")

    height, width = img_bchw.shape[-2:]
    if radius > min(height, width) / 2:
        raise ValueError(
            f"radius must be <= min(height, width) / 2. Got radius={radius} for {height}x{width}."
        )

    if sigma < 0:
        raise ValueError("sigma must be >= 0 (0 = auto).")

    dtype = _resolve_precision_dtype(precision)

    if dtype == torch.float16 and img_bchw.device.type == "cpu":
        logger.warning("float16 on CPU may be slow and may have precision issues.")

    effective_sigma = sigma if sigma > 0 else radius / 2.0
    if sigma > radius / 2.0:
        logger.warning("sigma > radius/2, kernel will truncate the Gaussian tail.")

    kernel_size = 2 * radius + 1
    working = img_bchw.to(dtype=dtype)
    low = gaussian_blur2d(
        working,
        kernel_size=(kernel_size, kernel_size),
        sigma=(effective_sigma, effective_sigma),
        border_type="replicate",
    )
    high = working - low
    return low, high


def frequency_recombine(
    low_bchw: torch.Tensor,
    high_bchw: torch.Tensor,
    detail: float = 1.0,
) -> torch.Tensor:
    """Rebuild an image from frequency-separation layers: ``low + detail * high``.

    ``high`` is a signed residual (values around 0, often negative), so it is added without
    clamping it first; only the final result is clamped to ``[0, 1]``. With ``detail = 1``
    and untouched layers the original image comes back exactly (up to float rounding);
    ``detail > 1`` boosts texture and ``detail < 1`` softens it.

    Args:
        low_bchw: Low-frequency layer, BCHW RGB.
        high_bchw: High-frequency layer, BCHW RGB, same height and width as ``low_bchw``
            (a batch of 1 broadcasts).
        detail: Multiplier for the high layer in ``[0, 3]``.

    Returns:
        float32 BCHW tensor clamped to ``[0, 1]`` on ``low_bchw``'s device.

    Raises:
        TypeError: If an input is not a tensor.
        ValueError: If shapes, channels, batch sizes or ``detail`` are invalid.
    """

    for name, tensor in (("low_bchw", low_bchw), ("high_bchw", high_bchw)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if tensor.ndim != 4 or tensor.shape[1] != 3:
            raise ValueError(f"{name} must be BCHW RGB, got shape {tuple(tensor.shape)}.")
    if low_bchw.shape[-2:] != high_bchw.shape[-2:]:
        raise ValueError(
            "low and high must have the same height and width, got "
            f"{tuple(low_bchw.shape[-2:])} and {tuple(high_bchw.shape[-2:])}."
        )
    batches = (low_bchw.shape[0], high_bchw.shape[0])
    if batches[0] != batches[1] and 1 not in batches:
        raise ValueError("low and high batch sizes must match, or one of them must be 1.")
    if isinstance(detail, bool) or not isinstance(detail, (int, float)):
        raise ValueError("detail must be a number in [0, 3].")
    if not 0.0 <= float(detail) <= 3.0:
        raise ValueError("detail must be a number in [0, 3].")

    low = low_bchw.to(dtype=torch.float32)
    high = high_bchw.to(device=low.device, dtype=torch.float32)
    return (low + float(detail) * high).clamp(0.0, 1.0)
