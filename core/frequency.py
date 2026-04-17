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
