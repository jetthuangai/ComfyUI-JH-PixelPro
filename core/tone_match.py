"""Tone-match helpers: LAB covariance transfer + LUT export composition."""

from __future__ import annotations

import os
from numbers import Integral

import torch
from kornia.color import lab_to_rgb, rgb_to_lab

from .lut import export_cube, identity_hald

_LAB_MIN = torch.tensor([0.0, -128.0, -128.0], dtype=torch.float32)
_LAB_MAX = torch.tensor([100.0, 127.0, 127.0], dtype=torch.float32)


def _prepare_image(name: str, image: torch.Tensor, *, device: str | torch.device) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (B,H,W,3), got {tuple(image.shape)}.")
    if not torch.is_floating_point(image):
        raise ValueError(f"{name} must be floating point, got {image.dtype}.")

    compute_dtype = image.dtype if image.dtype in {torch.float32, torch.float64} else torch.float32
    return image.to(device=device, dtype=compute_dtype).clamp(0.0, 1.0)


def _resolve_output_path(filename: str) -> str:
    name = filename.strip()
    if not name.lower().endswith(".cube"):
        name = name + ".cube"
    if os.path.isabs(name):
        return name
    try:
        import folder_paths  # type: ignore

        base = folder_paths.get_output_directory()
    except ImportError:
        base = os.path.abspath("output")
    return os.path.join(base, name)


def _channel_centers(vmin: float, vmax: float, *, n_bins: int) -> torch.Tensor:
    return torch.linspace(vmin, vmax, n_bins, dtype=torch.float32)


def _match_histogram_channel(
    source_channel: torch.Tensor,
    reference_channel: torch.Tensor,
    *,
    n_bins: int,
    vmin: float,
    vmax: float,
) -> torch.Tensor:
    source_cpu = source_channel.detach().to("cpu", torch.float32).clamp(vmin, vmax).reshape(-1)
    reference_cpu = (
        reference_channel.detach().to("cpu", torch.float32).clamp(vmin, vmax).reshape(-1)
    )

    src_hist = torch.histc(source_cpu, bins=n_bins, min=vmin, max=vmax)
    ref_hist = torch.histc(reference_cpu, bins=n_bins, min=vmin, max=vmax)
    src_cdf = torch.cumsum(src_hist, dim=0)
    ref_cdf = torch.cumsum(ref_hist, dim=0)
    src_cdf = src_cdf / src_cdf[-1].clamp_min(1.0)
    ref_cdf = ref_cdf / ref_cdf[-1].clamp_min(1.0)
    centers = _channel_centers(vmin, vmax, n_bins=n_bins)

    bin_scale = (n_bins - 1) / max(vmax - vmin, 1e-6)
    source_flat = source_channel.detach().to("cpu", torch.float32).clamp(vmin, vmax).reshape(-1)
    lower_index = ((source_flat - vmin) * bin_scale).floor().to(torch.long).clamp(0, n_bins - 1)
    upper_index = (lower_index + 1).clamp(0, n_bins - 1)
    lower_center = centers[lower_index]
    upper_center = centers[upper_index]
    interp = torch.where(
        upper_center > lower_center,
        (source_flat - lower_center) / (upper_center - lower_center),
        torch.zeros_like(source_flat),
    )
    source_quantile = src_cdf[lower_index] + interp * (src_cdf[upper_index] - src_cdf[lower_index])
    ref_index = torch.searchsorted(ref_cdf, source_quantile, right=False).clamp(0, n_bins - 1)
    matched = centers[ref_index]
    return matched.view_as(source_channel).to(
        device=source_channel.device,
        dtype=source_channel.dtype,
    )


def _flatten_lab_samples(lab_bchw: torch.Tensor) -> torch.Tensor:
    return lab_bchw.permute(0, 2, 3, 1).reshape(-1, 3)


def _covariance(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = samples.mean(dim=0)
    centered = samples - mean
    denom = max(samples.shape[0] - 1, 1)
    covariance = centered.transpose(0, 1).matmul(centered) / denom
    return mean, covariance


def _mkl_transform_matrix(
    source_samples: torch.Tensor,
    reference_samples: torch.Tensor,
    *,
    eps: float = 1e-5,
    condition_limit: float = 1e6,
) -> torch.Tensor | None:
    """Compute the 3x3 MKL covariance transform matrix, or ``None`` if unstable."""

    _, source_cov = _covariance(source_samples)
    _, reference_cov = _covariance(reference_samples)
    source_cond = torch.linalg.cond(source_cov)
    reference_cond = torch.linalg.cond(reference_cov)
    if (
        not torch.isfinite(source_cond)
        or not torch.isfinite(reference_cond)
        or source_cond > condition_limit
        or reference_cond > condition_limit
    ):
        return None

    eye = torch.eye(3, dtype=source_cov.dtype, device=source_cov.device)
    try:
        source_cholesky = torch.linalg.cholesky(source_cov + eye * eps)
        reference_cholesky = torch.linalg.cholesky(reference_cov + eye * eps)
        matrix = reference_cholesky @ torch.linalg.inv(source_cholesky)
    except RuntimeError:
        return None

    if not torch.isfinite(matrix).all():
        return None
    return matrix


def _apply_mkl_covariance_transfer(
    source_lab: torch.Tensor,
    reference_lab: torch.Tensor,
) -> torch.Tensor:
    reference_samples = _flatten_lab_samples(reference_lab)
    reference_mean, _ = _covariance(reference_samples)
    matched_batches: list[torch.Tensor] = []

    for batch_index in range(source_lab.shape[0]):
        batch_lab = source_lab[batch_index : batch_index + 1]
        source_samples = _flatten_lab_samples(batch_lab)
        source_mean, _ = _covariance(source_samples)
        transform = _mkl_transform_matrix(source_samples, reference_samples)
        centered = source_samples - source_mean
        if transform is None:
            matched_samples = centered + reference_mean
        else:
            matched_samples = centered.matmul(transform.transpose(0, 1)) + reference_mean
        matched_batches.append(
            matched_samples.reshape_as(batch_lab.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        )

    matched_lab = torch.cat(matched_batches, dim=0)
    lab_min = _LAB_MIN.to(device=matched_lab.device, dtype=matched_lab.dtype).view(1, 3, 1, 1)
    lab_max = _LAB_MAX.to(device=matched_lab.device, dtype=matched_lab.dtype).view(1, 3, 1, 1)
    return matched_lab.clamp(min=lab_min, max=lab_max)


def compute_lab_histogram_match(
    reference: torch.Tensor,
    source: torch.Tensor,
    *,
    n_bins: int = 256,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Transfer reference LAB covariance statistics to ``source``.

    The function name is retained for N-17 API compatibility. Batch-6 v0.8.1
    switched away from per-channel histogram matching because identity HALDs are
    uniform in sRGB but not in LAB. Batch-6 v0.8.2 upgrades the transfer to an
    MKL 3x3 covariance mapping so cross-channel color correlations are preserved.
    """

    if isinstance(n_bins, bool) or not isinstance(n_bins, Integral) or int(n_bins) < 8:
        raise ValueError("n_bins must be an integer >= 8.")

    reference_prepared = _prepare_image("reference", reference, device=device)
    source_prepared = _prepare_image("source", source, device=device)

    reference_lab = rgb_to_lab(reference_prepared.permute(0, 3, 1, 2))
    source_lab = rgb_to_lab(source_prepared.permute(0, 3, 1, 2))

    reference_rgb_std = reference_prepared.std(unbiased=False)
    reference_chroma = reference_lab[:, 1:].abs().mean()
    if reference_rgb_std < 1e-4 and reference_chroma < 1.0:
        return source_prepared.to(dtype=source.dtype)

    matched_lab = _apply_mkl_covariance_transfer(source_lab, reference_lab)
    matched_rgb = lab_to_rgb(matched_lab).clamp(0.0, 1.0)
    return matched_rgb.permute(0, 2, 3, 1).to(dtype=source.dtype)


def tone_match_lut(
    reference: torch.Tensor,
    level: int,
    filename: str,
    title: str = "Tone Match LUT",
    *,
    device: str | torch.device = "cpu",
) -> str:
    """Generate an auto tone-match LUT by grading an identity HALD via LAB covariance transfer."""

    if isinstance(level, bool) or not isinstance(level, Integral):
        raise ValueError("level must be an integer in [2, 16].")
    level = int(level)
    reference_prepared = _prepare_image("reference", reference, device=device)
    hald_identity = identity_hald(
        level,
        device=reference_prepared.device,
        dtype=reference_prepared.dtype,
    )
    matched_hald = compute_lab_histogram_match(
        reference_prepared,
        hald_identity,
        device=reference_prepared.device,
    )
    return export_cube(matched_hald, level, _resolve_output_path(filename), title=title)
