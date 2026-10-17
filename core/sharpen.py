"""Smart sharpen (unsharp mask with soft threshold, luminance-only mode and mask gating)."""

from __future__ import annotations

import math
from numbers import Real

import torch
from kornia.filters import gaussian_blur2d

_LUMA_WEIGHTS = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)


def _validate_float(name: str, value: float, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    value_float = float(value)
    if not (lower <= value_float <= upper):  # also rejects NaN
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_float


def _validate_image(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"image must have shape (B,H,W,3), got {tuple(image.shape)}.")
    if not torch.is_floating_point(image):
        raise ValueError(f"image must be a floating point tensor, got {image.dtype}.")
    if image.shape[0] < 1 or image.shape[1] < 1 or image.shape[2] < 1:
        raise ValueError("image must not be empty.")
    return image.to(dtype=torch.float32)


def _validate_mask(mask: torch.Tensor | None, image: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor.")
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (B,H,W), got {tuple(mask.shape)}.")
    if not torch.is_floating_point(mask) and mask.dtype is not torch.bool:
        raise ValueError(f"mask must be float or bool, got {mask.dtype}.")
    if mask.shape[1:] != image.shape[1:3]:
        raise ValueError("mask and image spatial shapes must match.")
    if mask.shape[0] not in (1, image.shape[0]):
        raise ValueError("mask batch must match image batch or be 1.")
    return mask.to(device=image.device, dtype=torch.float32).clamp(0.0, 1.0)


def smart_sharpen(
    image: torch.Tensor,
    *,
    amount: float = 1.0,
    radius: float = 1.0,
    threshold: float = 0.0,
    luminance_only: bool = True,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unsharp-mask sharpening with soft threshold, luma-only mode and optional mask.

    ``detail = x - gaussian_blur(x, sigma=radius)`` (reflect borders). The detail is
    soft-thresholded (``sign(d) * max(|d| - threshold, 0)``, a continuous ramp so no
    speckle) and added back: ``out = x + amount * detail * mask``, clamped to [0, 1].
    With ``luminance_only`` only the YCbCr Y channel is sharpened, chroma is untouched.

    Args:
        image: ``(B,H,W,3)`` float RGB in [0,1].
        amount: strength in [0, 5]; 0 returns the (float32) input unchanged.
        radius: Gaussian sigma in pixels, [0.3, 10]. Kernel is ``2*ceil(3*radius)+1``.
        threshold: detail amplitude below which sharpening is suppressed, [0, 0.5].
        luminance_only: sharpen luma only (avoids colour fringes) instead of each RGB channel.
        mask: optional ``(B,H,W)`` (or batch 1) gate in [0,1]; 0 = no sharpening.

    Returns:
        ``(B,H,W,3)`` float32 tensor on the input device.

    Raises:
        TypeError: for non-tensor ``image`` / ``mask``.
        ValueError: for bad shapes, dtypes, out-of-range parameters or images too small for
            the blur kernel.
    """
    image = _validate_image(image)
    amount = _validate_float("amount", amount, lower=0.0, upper=5.0)
    radius = _validate_float("radius", radius, lower=0.3, upper=10.0)
    threshold = _validate_float("threshold", threshold, lower=0.0, upper=0.5)
    if not isinstance(luminance_only, bool):
        raise ValueError("luminance_only must be a bool.")
    gate = _validate_mask(mask, image)

    if amount == 0.0:
        return image.clone()

    half = math.ceil(3.0 * radius)
    if min(image.shape[1:3]) <= half:
        raise ValueError(
            f"image is too small for radius {radius} (needs H and W > {half}), "
            f"got {image.shape[1]}x{image.shape[2]}."
        )
    kernel = 2 * half + 1

    x = image.permute(0, 3, 1, 2).contiguous()
    if luminance_only:
        # BT.601 luma, same Y as YCbCr. Adding a delta to all three RGB channels changes
        # only Y (Cb/Cr depend on B-Y and R-Y), so no lossy colour-space round trip.
        base = (x * _LUMA_WEIGHTS.to(x.device).view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    else:
        base = x

    detail = base - gaussian_blur2d(base, (kernel, kernel), (radius, radius), "reflect")
    if threshold > 0.0:
        detail = detail.sign() * (detail.abs() - threshold).clamp_min(0.0)
    detail = detail * amount
    if gate is not None:
        detail = detail * gate.unsqueeze(1)

    out = x + detail  # (B,1,H,W) detail broadcasts over RGB in luminance mode
    return out.clamp(0.0, 1.0).permute(0, 2, 3, 1).contiguous()
