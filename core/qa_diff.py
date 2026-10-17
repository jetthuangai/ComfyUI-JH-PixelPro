"""Numerical and visual image comparison (SSIM, PSNR, MAE) for retouch QA."""

from __future__ import annotations

import math
from numbers import Integral, Real

import kornia
import torch

DIFF_MODES = ("abs", "heatmap", "ssim_map")
PSNR_CAP_DB = 100.0
CHANGED_THRESHOLD = 1.0 / 255.0
_MSE_FLOOR = 10.0 ** (-PSNR_CAP_DB / 10.0)


def _validate_image(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim != 4 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (B,H,W,3), got {tuple(value.shape)}.")
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must be a floating point tensor, got {value.dtype}.")
    return value.to(dtype=torch.float32).clamp(0.0, 1.0)


def _validate_float(name: str, value: float, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    value_float = float(value)
    if not lower <= value_float <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_float


def _validate_int(name: str, value: int, *, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}].")
    value_int = int(value)
    if not lower <= value_int <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_int


def _validate_mask(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor.")
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (B,H,W), got {tuple(mask.shape)}.")
    if not torch.is_floating_point(mask) and mask.dtype is not torch.bool:
        raise ValueError(f"mask must be float or bool, got {mask.dtype}.")
    return mask.to(dtype=torch.float32).clamp(0.0, 1.0)


def _match_batch(name: str, tensor: torch.Tensor, batch: int) -> torch.Tensor:
    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch, *tensor.shape[1:])
    raise ValueError(f"{name} batch {tensor.shape[0]} does not match {batch} (or 1).")


def _weighted_mean(values: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    """Mean of (H,W) values, optionally weighted by an (H,W) mask (sum > 0)."""
    if weight is None:
        return values.mean()
    return (values * weight).sum() / weight.sum()


def qa_diff(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    *,
    window_size: int = 11,
    amplify: float = 4.0,
    diff_mode: str = "abs",
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[dict]]:
    """Compare two images numerically (SSIM, PSNR, MAE) and visually.

    Args:
        image_a: Reference image, float (B,H,W,3) in [0,1].
        image_b: Image to compare, same spatial size. Batch 1 broadcasts.
        window_size: Odd SSIM Gaussian window size in [3, 31]. Clamped to the largest
            odd number <= min(H, W) for small images.
        amplify: Diff visualisation gain in [1, 100] (ignored by ``ssim_map``).
        diff_mode: ``abs`` (per-channel |a-b|*amplify), ``heatmap`` (turbo colormap
            of max-channel |a-b|*amplify) or ``ssim_map`` (1 - SSIM map, grey RGB).
        mask: Optional (B,H,W) mask in [0,1] (batch 1 broadcasts). All metrics are
            weighted by it; the diff image is not masked. An all-zero mask yields
            neutral metrics (ssim 1.0, psnr 100.0, others 0.0).

    Returns:
        ``(diff, metrics)``: diff float32 (B,H,W,3) in [0,1]; one dict per batch
        item with ``ssim`` (0..1 typical), ``psnr`` (dB, capped at 100.0 for
        identical images), ``mae``, ``max_abs`` (0..1) and ``changed_pct`` (% of
        pixels whose max channel diff exceeds 1/255).

    Raises:
        TypeError: Non-tensor input.
        ValueError: Bad shape/dtype, mismatched spatial size or batch, invalid
            window_size / amplify / diff_mode, or images smaller than 3 px per side.
    """
    a = _validate_image("image_a", image_a)
    b = _validate_image("image_b", image_b)
    window_size = _validate_int("window_size", window_size, lower=3, upper=31)
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")
    amplify = _validate_float("amplify", amplify, lower=1.0, upper=100.0)
    if diff_mode not in DIFF_MODES:
        raise ValueError(f"diff_mode must be one of {DIFF_MODES}, got {diff_mode!r}.")
    if a.shape[1:3] != b.shape[1:3]:
        raise ValueError(
            f"image_a and image_b spatial sizes must match, got "
            f"{tuple(a.shape[1:3])} vs {tuple(b.shape[1:3])} (no automatic resize)."
        )
    if a.shape[0] != b.shape[0] and 1 not in (a.shape[0], b.shape[0]):
        raise ValueError(f"image batch sizes {a.shape[0]} and {b.shape[0]} are incompatible.")
    min_side = min(a.shape[1], a.shape[2])
    if min_side < 3:
        raise ValueError(f"images must be at least 3x3 pixels, got {a.shape[2]}x{a.shape[1]}.")
    window_size = min(window_size, min_side if min_side % 2 else min_side - 1)
    b = b.to(a.device)
    batch = max(a.shape[0], b.shape[0])
    a = _match_batch("image_a", a, batch)
    b = _match_batch("image_b", b, batch)
    weight = None
    if mask is not None:
        weight = _validate_mask(mask).to(a.device)
        if weight.shape[1:] != a.shape[1:3]:
            raise ValueError("mask spatial size must match the images.")
        weight = _match_batch("mask", weight, batch)

    pa = a.permute(0, 3, 1, 2)
    pb = b.permute(0, 3, 1, 2)
    diff = (pa - pb).abs()  # (B,3,H,W)
    max_diff = diff.amax(dim=1)  # (B,H,W)
    ssim_map = kornia.metrics.ssim(pa, pb, window_size, max_val=1.0).mean(dim=1)  # (B,H,W)

    metrics: list[dict] = []
    for i in range(batch):
        w = None if weight is None else weight[i]
        if w is not None and w.sum() <= 0:
            metrics.append(
                {"ssim": 1.0, "psnr": PSNR_CAP_DB, "mae": 0.0, "max_abs": 0.0, "changed_pct": 0.0}
            )
            continue
        mse = _weighted_mean(diff[i].pow(2).mean(dim=0), w).item()
        psnr = PSNR_CAP_DB if mse <= _MSE_FLOOR else -10.0 * math.log10(mse)
        max_abs = max_diff[i] if w is None else max_diff[i] * (w > 0)
        metrics.append(
            {
                "ssim": float(_weighted_mean(ssim_map[i], w).item()),
                "psnr": float(min(psnr, PSNR_CAP_DB)),
                "mae": float(_weighted_mean(diff[i].mean(dim=0), w).item()),
                "max_abs": float(max_abs.max().item()),
                "changed_pct": float(
                    100.0 * _weighted_mean((max_diff[i] > CHANGED_THRESHOLD).float(), w).item()
                ),
            }
        )

    if diff_mode == "abs":
        out = (diff * amplify).clamp(0.0, 1.0)
    elif diff_mode == "heatmap":
        cmap = kornia.color.ColorMap(
            base=kornia.color.ColorMapType.turbo, num_colors=256, device=a.device
        )
        out = kornia.color.apply_colormap((max_diff * amplify).clamp(0.0, 1.0).unsqueeze(1), cmap)
    else:
        out = (1.0 - ssim_map).clamp(0.0, 1.0).unsqueeze(1).expand(-1, 3, -1, -1)
    return out.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0).float(), metrics
