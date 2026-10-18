"""Guided filter, fast guided upsampling and proxy resizing for BCHW tensors.

Shared base for the edge-aware tone tools (Shadows/Highlights Pro, ColorLab
``tone_engine=local``) and the RGB mask refiner.

References: He, Sun & Tang 2010 "Guided Image Filtering" (grey and colour guide,
Eq. 19-21); He & Sun 2015 "Fast Guided Filter" (coefficients solved at low
resolution, upsampled bilinearly and applied to the full-resolution guide).

``kornia.filters.guided_blur(subsample=s)`` crashes when H or W is not a
multiple of ``s`` (Kornia 0.8.1), so the subsampled path is implemented here
with explicit target sizes.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch
import torch.nn.functional as functional

# Peak working memory of the colour-guide solve, measured at 4K on CUDA
# (~350 B/px for RGB guide -> RGB src). Strips keep one call under the budget.
_COLOR_BYTES_PER_PIXEL = 400
_STRIP_BUDGET_BYTES = 768 * 1024**2


def _validate_bchw(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim != 4:
        raise ValueError(f"{name} must have shape (B,C,H,W), got {tuple(value.shape)}.")
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must be floating point, got {value.dtype}.")
    return value.to(dtype=torch.float32)


def _validate_int(name: str, value: int, *, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}].")
    value_int = int(value)
    if not lower <= value_int <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_int


def _validate_eps(eps: float) -> float:
    if isinstance(eps, bool) or not isinstance(eps, Real) or not 0.0 < float(eps) <= 1.0:
        raise ValueError("eps must be in (0, 1].")
    return float(eps)


def box_filter(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Mean over a ``(2r+1)^2`` window with replicate borders, for any image size.

    Two 1-D average pools are exact and cheap. Replicate padding keeps border
    statistics unbiased (zero padding pulls them toward 0; reflect padding fails
    when the radius exceeds the image size).
    """

    if radius <= 0:
        return x
    kernel_size = 2 * radius + 1
    padded = functional.pad(x, (radius, radius, radius, radius), mode="replicate")
    rows = functional.avg_pool2d(padded, kernel_size=(1, kernel_size), stride=1)
    return functional.avg_pool2d(rows, kernel_size=(kernel_size, 1), stride=1)


def _gauss_box_radii(sigma: float) -> tuple[int, int, int]:
    """Three box radii approximating a Gaussian of ``sigma`` (Kovesi 2010)."""

    w_ideal = math.sqrt(12.0 * sigma * sigma / 3.0 + 1.0)
    wl = math.floor(w_ideal)
    if wl % 2 == 0:
        wl -= 1
    wl = max(wl, 1)
    wu = wl + 2
    m = round((12.0 * sigma * sigma - 3 * wl * wl - 12.0 * wl - 9.0) / (-4.0 * wl - 4.0))
    m = min(max(m, 0), 3)
    radii = [(wl - 1) // 2 if i < m else (wu - 1) // 2 for i in range(3)]
    if all(r == 0 for r in radii):
        radii[0] = 1
    return radii[0], radii[1], radii[2]


def gaussian_blur_box(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur of BCHW by three box passes (cost independent of sigma; replicate borders)."""

    if sigma < 0.3:
        return x
    for radius in _gauss_box_radii(sigma):
        x = box_filter(x, radius)
    return x


def resize_to(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize BCHW to ``size``: area average when shrinking, bilinear when enlarging."""

    if tuple(x.shape[-2:]) == tuple(size):
        return x
    if size[0] <= x.shape[-2] and size[1] <= x.shape[-1]:
        return functional.interpolate(x, size=size, mode="area")
    return functional.interpolate(x, size=size, mode="bilinear", align_corners=False)


def proxy_size(height: int, width: int, max_side: int) -> tuple[int, int]:
    """Size whose long edge is at most ``max_side`` with the aspect ratio kept."""

    _validate_int("max_side", max_side, lower=8, upper=65536)
    scale = min(1.0, max_side / max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale))


def make_proxy(x: torch.Tensor, max_side: int) -> torch.Tensor:
    """Area-downscale BCHW so the long edge is at most ``max_side`` (no-op if smaller)."""

    x = _validate_bchw("x", x)
    return resize_to(x, proxy_size(x.shape[-2], x.shape[-1], max_side))


def _coefficients(
    guide: torch.Tensor, src: torch.Tensor, radius: int, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Window-averaged linear coefficients so that ``src ~ a . guide + b``.

    Returns ``mean_a`` shaped (B, Cg*Cs, H, W) (guide-major) and ``mean_b`` (B, Cs, H, W).
    """

    batch, guide_channels, height, width = guide.shape
    src_channels = src.shape[1]
    # A window wider than the image only repeats the replicated border.
    radius = max(1, min(radius, max(height, width)))
    mean_i = box_filter(guide, radius)
    mean_p = box_filter(src, radius)

    if guide_channels == 1:
        var_i = box_filter(guide * guide, radius) - mean_i * mean_i
        cov_ip = box_filter(guide * src, radius) - mean_i * mean_p
        a = cov_ip / (var_i + eps)
        b = mean_p - a * mean_i
        return box_filter(a, radius), box_filter(b, radius)

    # Colour guide: per-pixel 3x3 covariance solve (He 2010, Eq. 19-21).
    products = (guide.unsqueeze(2) * guide.unsqueeze(1)).flatten(1, 2)
    var_i = box_filter(products, radius) - (mean_i.unsqueeze(2) * mean_i.unsqueeze(1)).flatten(1, 2)
    cross = (guide.unsqueeze(2) * src.unsqueeze(1)).flatten(1, 2)
    cov_ip = box_filter(cross, radius) - (mean_i.unsqueeze(2) * mean_p.unsqueeze(1)).flatten(1, 2)

    var_i = var_i.permute(0, 2, 3, 1).reshape(batch, height, width, guide_channels, guide_channels)
    cov_ip = cov_ip.permute(0, 2, 3, 1).reshape(batch, height, width, guide_channels, src_channels)
    identity = torch.eye(guide_channels, device=guide.device, dtype=guide.dtype)
    a = torch.linalg.solve(var_i + eps * identity, cov_ip)  # (B,H,W,Cg,Cs)
    a_bchw = a.reshape(batch, height, width, guide_channels * src_channels).permute(0, 3, 1, 2)
    weighted = (
        mean_i.unsqueeze(2) * a_bchw.view(batch, guide_channels, src_channels, height, width)
    ).sum(1)
    b = mean_p - weighted
    return box_filter(a_bchw, radius), box_filter(b, radius)


def _apply(guide: torch.Tensor, mean_a: torch.Tensor, mean_b: torch.Tensor) -> torch.Tensor:
    batch, guide_channels, height, width = guide.shape
    if guide_channels == 1:
        return mean_a * guide + mean_b
    src_channels = mean_b.shape[1]
    a = mean_a.view(batch, guide_channels, src_channels, height, width)
    return mean_b + (guide.unsqueeze(2) * a).sum(1)


def _check_pair(guide: torch.Tensor, src: torch.Tensor) -> None:
    if guide.shape[1] not in (1, 3):
        raise ValueError(f"guide must have 1 or 3 channels, got {guide.shape[1]}.")
    if guide.shape[0] != src.shape[0] or guide.shape[-2:] != src.shape[-2:]:
        raise ValueError(
            f"guide {tuple(guide.shape)} and src {tuple(src.shape)} "
            "must share batch and spatial size."
        )


def _full_res(guide: torch.Tensor, src: torch.Tensor, radius: int, eps: float) -> torch.Tensor:
    """Full-resolution filter; colour guides run in row strips to bound memory.

    Every output pixel depends only on inputs within ``2*radius`` rows (two box
    passes), so strips with a ``2*radius`` halo give exactly the untiled result.
    """

    batch, guide_channels, height, width = guide.shape
    pixel_bytes = _COLOR_BYTES_PER_PIXEL * max(1, batch) * (src.shape[1] + guide_channels) // 4
    rows_per_strip = max(1, _STRIP_BUDGET_BYTES // max(1, pixel_bytes * width))
    if guide_channels == 1 or rows_per_strip >= height:
        mean_a, mean_b = _coefficients(guide, src, radius, eps)
        return _apply(guide, mean_a, mean_b)

    halo = 2 * min(radius, max(height, width))
    out = torch.empty_like(src)
    for top in range(0, height, rows_per_strip):
        bottom = min(height, top + rows_per_strip)
        lo, hi = max(0, top - halo), min(height, bottom + halo)
        g, p = guide[..., lo:hi, :], src[..., lo:hi, :]
        mean_a, mean_b = _coefficients(g, p, radius, eps)
        out[..., top:bottom, :] = _apply(g, mean_a, mean_b)[..., top - lo : bottom - lo, :]
    return out


def guided_filter(
    guide: torch.Tensor,
    src: torch.Tensor,
    *,
    radius: int,
    eps: float,
    subsample: int = 1,
) -> torch.Tensor:
    """Edge-preserving smoothing of ``src`` steered by ``guide`` (grey or RGB).

    Args:
        guide: ``(B,1,H,W)`` or ``(B,3,H,W)`` float guide, usually in ``[0,1]``.
        src: ``(B,C,H,W)`` float tensor to filter; same batch and spatial size.
        radius: Window radius in full-resolution pixels.
        eps: Regularisation; smaller keeps more edges (variance units of the guide).
        subsample: Fast-guided-filter factor. Coefficients are solved on a
            ``1/subsample`` area-downscaled copy; any image size is accepted.

    Returns:
        Filtered ``(B,C,H,W)`` float32 tensor (not clamped).
    """

    guide = _validate_bchw("guide", guide)
    src = _validate_bchw("src", src).to(device=guide.device)
    _check_pair(guide, src)
    radius = _validate_int("radius", radius, lower=1, upper=4096)
    eps = _validate_eps(eps)
    subsample = _validate_int("subsample", subsample, lower=1, upper=64)

    height, width = guide.shape[-2:]
    if subsample == 1:
        return _full_res(guide, src, radius, eps)

    low_size = (max(1, math.ceil(height / subsample)), max(1, math.ceil(width / subsample)))
    low_radius = max(1, round(radius / subsample))
    mean_a, mean_b = _coefficients(
        resize_to(guide, low_size), resize_to(src, low_size), low_radius, eps
    )
    return _apply(guide, resize_to(mean_a, (height, width)), resize_to(mean_b, (height, width)))


def guided_upsample(
    guide_full: torch.Tensor,
    guide_low: torch.Tensor,
    result_low: torch.Tensor,
    *,
    radius: int,
    eps: float,
) -> torch.Tensor:
    """Carry a low-resolution result to full resolution along the full-res guide's edges.

    Fits ``result_low ~ a . guide_low + b`` locally at low resolution, upsamples
    ``a, b`` bilinearly and applies them to ``guide_full``. Used to run costly
    operators (local Laplacian, dehaze transmission) on a proxy.

    Args:
        guide_full: ``(B,1|3,H,W)`` full-resolution guide.
        guide_low: Same guide downscaled to the proxy size ``(h,w)``.
        result_low: ``(B,C,h,w)`` result computed on the proxy.
        radius: Window radius in proxy pixels.
        eps: Regularisation of the local fit.

    Returns:
        ``(B,C,H,W)`` float32 tensor.
    """

    guide_full = _validate_bchw("guide_full", guide_full)
    guide_low = _validate_bchw("guide_low", guide_low).to(device=guide_full.device)
    result_low = _validate_bchw("result_low", result_low).to(device=guide_full.device)
    _check_pair(guide_low, result_low)
    if guide_full.shape[:2] != guide_low.shape[:2]:
        raise ValueError("guide_full and guide_low must share batch and channel count.")
    radius = _validate_int("radius", radius, lower=1, upper=4096)
    eps = _validate_eps(eps)

    full_size = tuple(guide_full.shape[-2:])
    mean_a, mean_b = _coefficients(guide_low, result_low, radius, eps)
    return _apply(guide_full, resize_to(mean_a, full_size), resize_to(mean_b, full_size))
