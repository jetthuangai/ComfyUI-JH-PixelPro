"""Smooth thin-plate-spline face warp (alternative to the Delaunay warp in ``face_pipeline``)."""

from __future__ import annotations

from numbers import Integral, Real

import torch
import torch.nn.functional as functional

from .face_pipeline import _prepare_image, _prepare_landmarks, _to_pixel_landmarks

_GRID_STEP = 4  # displacement field is evaluated every 4 px, then upsampled bicubically
_EVAL_CHUNK = 8192  # grid points per block: keeps the (chunk, N) kernel matrix near 32 MB
_MOVED_PX = 0.5  # pairs moving more than this are always kept when subsampling
_ANCHORS_PER_EDGE = 5  # corners + 3 interior points on every edge
_RIDGES = (1e-9, 1e-7, 1e-5)  # tiny diagonal regularisation; larger values only as fallback
# smoothness 1.0 -> ridge 1e-3 (normalised [-1, 1] coordinates). Measured on the sample portrait:
# removes the folds caused by ~15 px landmark detection noise between two photos, while a clean
# edit still lands within 0.01 px (median) / 0.9 px (p95) of its target.
_SMOOTHNESS_TO_RIDGE = 1e-3


def _validate_max_points(max_points: int) -> int:
    if isinstance(max_points, bool) or not isinstance(max_points, Integral):
        raise ValueError("max_points must be an integer in [4, 468].")
    value = int(max_points)
    if not 4 <= value <= 468:
        raise ValueError("max_points must be an integer in [4, 468].")
    return value


def _validate_smoothness(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not 0.0 <= float(value) <= 20.0:
        raise ValueError("smoothness must be a number in [0, 20].")
    return float(value)


def _validate_bool(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool.")
    return value


def _border_anchors(width: int, height: int) -> torch.Tensor:
    """Pixel coordinates (x, y) of fixed points along the image border, shape (M, 2)."""
    xs = torch.linspace(0.0, float(max(width - 1, 1)), _ANCHORS_PER_EDGE, dtype=torch.float64)
    ys = torch.linspace(0.0, float(max(height - 1, 1)), _ANCHORS_PER_EDGE, dtype=torch.float64)
    top = torch.stack([xs, torch.zeros_like(xs)], dim=-1)
    bottom = torch.stack([xs, torch.full_like(xs, float(max(height - 1, 1)))], dim=-1)
    left = torch.stack([torch.zeros_like(ys), ys], dim=-1)[1:-1]
    right = torch.stack([torch.full_like(ys, float(max(width - 1, 1))), ys], dim=-1)[1:-1]
    return torch.cat([top, bottom, left, right], dim=0)


def _subsample(src: torch.Tensor, dst: torch.Tensor, max_points: int) -> tuple[torch.Tensor, ...]:
    """Keep every pair that moves more than ``_MOVED_PX``; fill the budget with unmoved ones.

    Half of the remaining budget goes to the unmoved pairs nearest to the edit (otherwise
    unpinned neighbours drift by more than the edit itself), the other half is spaced
    evenly over the rest so distant features stay pinned too.
    """
    count = src.shape[0]
    if count <= max_points:
        return src, dst
    moved = (src - dst).norm(dim=-1) > _MOVED_PX
    moved_index = torch.nonzero(moved).flatten()
    still_index = torch.nonzero(~moved).flatten()
    budget = max(max_points - moved_index.numel(), 0)
    if budget > 0 and still_index.numel() > budget:
        near: torch.Tensor = still_index[:0]
        if moved_index.numel() > 0:
            dist = torch.cdist(src[still_index], src[moved_index]).amin(dim=1)
            near = still_index[dist.argsort()[: budget // 2]]
        rest = still_index[~torch.isin(still_index, near)]
        spread = budget - near.numel()
        pick = torch.linspace(0, rest.numel() - 1, spread).round().long().unique()
        still_index = torch.cat([near, rest[pick]])
    elif budget == 0:
        still_index = still_index[:0]
    index = torch.cat([moved_index, still_index]).sort().values
    return src[index], dst[index]


def _solve_tps(
    pts: torch.Tensor, values: torch.Tensor, ridge: float
) -> tuple[torch.Tensor, torch.Tensor]:
    count = pts.shape[0]
    d2 = torch.cdist(pts, pts).pow(2)
    kernel = 0.5 * d2 * (d2 + 1e-8).log()
    kernel = kernel + ridge * torch.eye(count, dtype=torch.float64, device=pts.device)
    ones = torch.ones(count, 1, dtype=torch.float64, device=pts.device)
    poly = torch.cat([ones, pts], dim=1)
    system = torch.zeros(count + 3, count + 3, dtype=torch.float64, device=pts.device)
    system[:count, :count] = kernel
    system[:count, count:] = poly
    system[count:, :count] = poly.T
    rhs = torch.zeros(count + 3, 2, dtype=torch.float64, device=pts.device)
    rhs[:count] = values
    solution = torch.linalg.solve(system, rhs)
    return solution[:count], solution[count:]


def _fit_tps(
    dst_norm: torch.Tensor, src_norm: torch.Tensor, smoothing: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit f with f(dst_i) = src_i (output space -> input space).

    Returns kernel weights ``(N, 2)`` and affine weights ``(3, 2)`` (float64). The kernel is
    U(r) = r^2 log r. ``smoothing`` is the regularisation ridge (0 = exact interpolation). If the
    solve fails or returns non-finite values, the larger fallback ridges above ``smoothing`` are
    tried in turn (none remain once ``smoothing`` exceeds them).
    """
    pts = dst_norm.to(torch.float64)
    values = src_norm.to(torch.float64)
    last_error: Exception | None = None
    for ridge in sorted({max(smoothing, r) for r in _RIDGES}):
        try:
            kernel_w, affine_w = _solve_tps(pts, values, ridge)
        except torch.linalg.LinAlgError as error:
            last_error = error
            continue
        if torch.isfinite(kernel_w).all() and torch.isfinite(affine_w).all():
            return kernel_w, affine_w
    raise torch.linalg.LinAlgError("TPS system is singular") from last_error


def _eval_kernel_part(
    grid: torch.Tensor, centers: torch.Tensor, kernel_w: torch.Tensor
) -> torch.Tensor:
    """Evaluate the non-affine part sum_i w_i U(|p - c_i|) at ``grid`` (M, 2) in row blocks.

    Blocks keep the (chunk, N) kernel matrix small, so memory stays bounded for any image size.
    """
    out = torch.empty_like(grid)
    for start in range(0, grid.shape[0], _EVAL_CHUNK):
        block = grid[start : start + _EVAL_CHUNK]
        d2 = torch.cdist(block, centers).pow(2)
        out[start : start + _EVAL_CHUNK] = (0.5 * d2 * (d2 + 1e-8).log()) @ kernel_w
    return out


def _eval_field(
    grid: torch.Tensor, centers: torch.Tensor, kernel_w: torch.Tensor, affine_w: torch.Tensor
) -> torch.Tensor:
    """Full spline value (kernel + affine) at ``grid`` (M, 2). Returns (M, 2)."""
    return _eval_kernel_part(grid, centers, kernel_w) + affine_w[0] + grid @ affine_w[1:]


def _warp_single(
    image_chw: torch.Tensor,
    src_px: torch.Tensor,
    dst_px: torch.Tensor,
    *,
    max_points: int,
    anchor_border: bool,
    smoothness: float = 0.0,
) -> torch.Tensor:
    _, height, width = image_chw.shape
    device = image_chw.device
    if src_px.shape[0] == 0 or src_px.shape != dst_px.shape:
        return image_chw
    valid = torch.isfinite(src_px).all(dim=-1) & torch.isfinite(dst_px).all(dim=-1)
    src_valid = src_px[valid].to(torch.float64)
    dst_valid = dst_px[valid].to(torch.float64)
    if src_valid.shape[0] < 3:
        return image_chw
    src_valid, dst_valid = _subsample(src_valid, dst_valid, max_points)

    if anchor_border:
        anchors = _border_anchors(width, height)
        src_valid = torch.cat([src_valid, anchors], dim=0)
        dst_valid = torch.cat([dst_valid, anchors], dim=0)

    scale = torch.tensor([2.0 / max(width - 1, 1), 2.0 / max(height - 1, 1)], dtype=torch.float64)
    src_norm = (src_valid * scale - 1.0).to(device)
    dst_norm = (dst_valid * scale - 1.0).to(device)
    try:
        kernel_w, affine_w = _fit_tps(dst_norm, src_norm, smoothness * _SMOOTHNESS_TO_RIDGE)
    except torch.linalg.LinAlgError:
        return image_chw

    grid_h = (height - 1) // _GRID_STEP + 2
    grid_w = (width - 1) // _GRID_STEP + 2
    gy, gx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, grid_h, dtype=torch.float64, device=device),
        torch.linspace(-1.0, 1.0, grid_w, dtype=torch.float64, device=device),
        indexing="ij",
    )
    coarse = torch.stack([gx, gy], dim=-1).reshape(-1, 2)
    # Only the kernel part is interpolated (bicubic); the affine part is exact at full resolution
    # because the bicubic filter does not reproduce linear ramps.
    kernel_part = _eval_kernel_part(coarse, dst_norm, kernel_w).to(torch.float32)
    kernel_part = kernel_part.reshape(1, grid_h, grid_w, 2).permute(0, 3, 1, 2)
    if (grid_h, grid_w) != (height, width):
        kernel_part = functional.interpolate(
            kernel_part, size=(height, width), mode="bicubic", align_corners=True
        )
    xs = torch.linspace(-1.0, 1.0, width, dtype=torch.float64, device=device)
    ys = torch.linspace(-1.0, 1.0, height, dtype=torch.float64, device=device)
    affine = torch.stack(
        [
            affine_w[0, c] + xs[None, :] * affine_w[1, c] + ys[:, None] * affine_w[2, c]
            for c in range(2)
        ],
        dim=-1,
    )
    mapped = kernel_part.permute(0, 2, 3, 1) + affine.to(torch.float32).unsqueeze(0)
    warped = functional.grid_sample(
        image_chw.unsqueeze(0),
        mapped,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped[0]


def face_warp_tps(
    image: torch.Tensor,
    src_landmarks: torch.Tensor,
    dst_landmarks: torch.Tensor,
    *,
    max_points: int = 468,
    anchor_border: bool = True,
    smoothness: float = 1.0,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Warp ``image`` so features located at ``src_landmarks`` move to ``dst_landmarks``.

    A thin-plate spline (kernel ``r^2 log r``) is fitted on the output->input mapping, so the
    result is one smooth global displacement field with no triangle seams. The field is
    evaluated in memory-bounded blocks on a grid with a 4 px step, upsampled bicubically and
    used to sample the source image.

    Args:
        image: float tensor ``(B, H, W, 3)``; values are clamped to ``[0, 1]``.
        src_landmarks: ``(B, 468, 2)`` (or batch 1, broadcast). Normalised ``[0, 1]`` or pixel
            coordinates, same convention as ``face_warp_delaunay``.
        dst_landmarks: ``(B, 468, 2)`` (or batch 1, broadcast), same convention.
        max_points: landmark budget. Pairs that move more than
            0.5 px are always kept; the rest of the budget is filled with evenly spaced unmoved
            pairs (deterministic). Range ``[4, 468]``.
        anchor_border: add fixed control points along the image border (corners plus three
            points per edge) so the background does not swim.
        smoothness: regularisation in ``[0, 20]``. ``0`` interpolates every landmark exactly;
            the default ``1`` absorbs landmark detection noise (targets taken from another
            photo) that would otherwise fold the image, at < 1 px cost on clean edits.
        device: compute device; defaults to ``image.device``.

    Returns:
        float32 tensor ``(B, H, W, 3)`` in ``[0, 1]`` on the compute device. Items with no
        landmarks (e.g. all NaN) or fewer than 3 finite landmark pairs are returned unchanged.

    Raises:
        TypeError: ``image`` or landmarks are not tensors.
        ValueError: bad shapes, dtype, batch mismatch, or invalid ``max_points`` /
            ``anchor_border``.
    """
    max_points = _validate_max_points(max_points)
    anchor_border = _validate_bool("anchor_border", anchor_border)
    smoothness = _validate_smoothness(smoothness)
    prepared = _prepare_image("image", image)
    prepared = prepared.to(torch.device(device) if device is not None else image.device)
    batch, height, width, _ = prepared.shape
    src_prepared = _prepare_landmarks("src_landmarks", src_landmarks, batch=batch)
    dst_prepared = _prepare_landmarks("dst_landmarks", dst_landmarks, batch=batch)

    outputs: list[torch.Tensor] = []
    for index in range(batch):
        src_px = torch.from_numpy(
            _to_pixel_landmarks(src_prepared[index], width=width, height=height)
        )
        dst_px = torch.from_numpy(
            _to_pixel_landmarks(dst_prepared[index], width=width, height=height)
        )
        warped = _warp_single(
            prepared[index].permute(2, 0, 1).contiguous(),
            src_px,
            dst_px,
            max_points=max_points,
            anchor_border=anchor_border,
            smoothness=smoothness,
        )
        outputs.append(warped.permute(1, 2, 0))
    return torch.stack(outputs, dim=0).clamp(0.0, 1.0).contiguous()
