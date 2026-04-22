"""Closed-form alpha-matte extraction via Levin 2008 matting Laplacian."""

from __future__ import annotations

import warnings
from numbers import Integral, Real
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as functional
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .mask_trimap import validate_trimap

# Reference:
# Levin, A., Lischinski, D., & Weiss, Y. (2008).
# "A Closed-Form Solution to Natural Image Matting." IEEE TPAMI 30(2), 228-242.


def _validate_image(guide: torch.Tensor) -> torch.Tensor:
    if not isinstance(guide, torch.Tensor):
        raise TypeError("guide must be a torch.Tensor.")
    if guide.ndim != 4 or guide.shape[-1] != 3:
        raise ValueError(f"guide must have shape (B,H,W,3), got {tuple(guide.shape)}.")
    if not torch.is_floating_point(guide):
        raise ValueError(f"guide must be floating point, got {guide.dtype}.")
    return guide.to(dtype=torch.float32).clamp(0.0, 1.0)


def _validate_int(name: str, value: int, *, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}].")
    value_int = int(value)
    if not lower <= value_int <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_int


def _validate_float(name: str, value: float, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    value_float = float(value)
    if not lower <= value_float <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_float


def _validate_compute_device(value: str) -> Literal["auto", "cuda", "cpu"]:
    if value not in {"auto", "cuda", "cpu"}:
        raise ValueError("compute_device must be one of: auto, cuda, cpu.")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("compute_device='cuda' but CUDA unavailable")
    return value


def _window_indices(height: int, width: int, y: int, x: int, radius: int) -> np.ndarray:
    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return (yy.reshape(-1) * width + xx.reshape(-1)).astype(np.int64, copy=False)


def _matting_laplacian(
    guide_np: np.ndarray,
    *,
    epsilon: float,
    window_radius: int,
) -> sparse.csr_matrix:
    """Build Levin's closed-form matting Laplacian for one RGB guide image."""

    height, width, channels = guide_np.shape
    if channels != 3:
        raise ValueError("guide_np must have 3 channels.")
    pixel_count = height * width
    flat_guide = guide_np.reshape(pixel_count, channels).astype(np.float64, copy=False)
    eye = np.eye(channels, dtype=np.float64)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []

    for y in range(height):
        for x in range(width):
            indices = _window_indices(height, width, y, x, window_radius)
            colors = flat_guide[indices]
            window_size = float(indices.size)
            mean = colors.mean(axis=0, keepdims=True)
            centered = colors - mean
            covariance = (centered.T @ centered) / window_size
            regularized = covariance + (epsilon / window_size) * eye
            inverse = np.linalg.pinv(regularized)
            affinity = (1.0 + centered @ inverse @ centered.T) / window_size
            values = np.eye(indices.size, dtype=np.float64) - affinity
            grid_rows = np.repeat(indices, indices.size)
            grid_cols = np.tile(indices, indices.size)
            rows.append(grid_rows)
            cols.append(grid_cols)
            data.append(values.reshape(-1))

    matrix = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(pixel_count, pixel_count),
        dtype=np.float64,
    )
    return matrix.tocsr()


def _cg_solve(matrix: sparse.csr_matrix, rhs: np.ndarray) -> np.ndarray:
    try:
        solved, info = sparse_linalg.cg(matrix, rhs, tol=1e-6, maxiter=2000)
    except TypeError:
        solved, info = sparse_linalg.cg(matrix, rhs, rtol=1e-6, maxiter=2000)
    if info != 0:
        raise RuntimeError(f"cg failed to converge: info={info}")
    return solved


def _solve_levin(
    laplacian: sparse.csr_matrix,
    trimap_np: np.ndarray,
    *,
    lambda_constraint: float,
) -> np.ndarray:
    """Solve ``(L + lambda D) alpha = lambda bs`` for one trimap."""

    flat_trimap = trimap_np.reshape(-1)
    known = (flat_trimap >= 0.95) | (flat_trimap <= 0.05)
    if not known.any():
        return np.full_like(flat_trimap, 0.5, dtype=np.float64)
    constraint = sparse.diags(known.astype(np.float64), format="csr")
    rhs = lambda_constraint * (flat_trimap >= 0.95).astype(np.float64)
    matrix = (laplacian + lambda_constraint * constraint).tocsr()
    try:
        solved = sparse_linalg.spsolve(matrix, rhs)
    except Exception:
        solved = _cg_solve(matrix, rhs)
    return np.nan_to_num(solved, nan=0.5, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def _solve_single(
    trimap_np: np.ndarray,
    guide_np: np.ndarray,
    *,
    epsilon: float,
    window_radius: int,
    lambda_constraint: float,
) -> np.ndarray:
    fg = trimap_np >= 0.95
    bg = trimap_np <= 0.05
    unknown = ~(fg | bg)
    alpha = np.zeros_like(trimap_np, dtype=np.float32)
    alpha[fg] = 1.0
    if not unknown.any():
        return alpha

    laplacian = _matting_laplacian(guide_np, epsilon=epsilon, window_radius=window_radius)
    solved = _solve_levin(laplacian, trimap_np, lambda_constraint=lambda_constraint)
    alpha = solved.reshape(trimap_np.shape).astype(np.float32, copy=False)
    alpha[fg] = 1.0
    alpha[bg] = 0.0
    return alpha


def _gpu_window_geometry(
    height: int,
    width: int,
    window_radius: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_size = 2 * window_radius + 1
    coords_y = torch.arange(height, device=device)
    coords_x = torch.arange(width, device=device)
    yy, xx = torch.meshgrid(coords_y, coords_x, indexing="ij")
    center_y = yy.reshape(-1)
    center_x = xx.reshape(-1)
    offsets = torch.arange(-window_radius, window_radius + 1, device=device)
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    neighbor_y = center_y.unsqueeze(1) + offset_y.reshape(1, kernel_size * kernel_size)
    neighbor_x = center_x.unsqueeze(1) + offset_x.reshape(1, kernel_size * kernel_size)
    valid = (neighbor_y >= 0) & (neighbor_y < height) & (neighbor_x >= 0) & (neighbor_x < width)
    neighbor_y = neighbor_y.clamp(0, height - 1)
    neighbor_x = neighbor_x.clamp(0, width - 1)
    neighbor_indices = neighbor_y * width + neighbor_x
    return neighbor_indices.to(torch.long), valid


def _build_matting_laplacian_gpu(
    guide: torch.Tensor,
    *,
    epsilon: float,
    window_radius: int,
    chunk_size: int = 65536,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Levin's matting Laplacian as CUDA CSR plus diagonal."""

    if guide.ndim != 3 or guide.shape[-1] != 3:
        raise ValueError("guide must have shape (H,W,3).")
    height, width, channels = guide.shape
    pixel_count = height * width
    kernel_size = 2 * window_radius + 1
    window_area = kernel_size * kernel_size
    dtype = torch.float64
    device = guide.device

    guide_nchw = guide.to(dtype=dtype).permute(2, 0, 1).unsqueeze(0)
    patches = functional.unfold(guide_nchw, kernel_size=kernel_size, padding=window_radius)
    patches = patches.squeeze(0).transpose(0, 1).reshape(pixel_count, channels, window_area)
    patches = patches.transpose(1, 2).contiguous()
    neighbor_indices, valid = _gpu_window_geometry(
        height,
        width,
        window_radius,
        device=device,
    )

    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    eye_channels = torch.eye(channels, dtype=dtype, device=device)
    eye_window = torch.eye(window_area, dtype=dtype, device=device)

    for start in range(0, pixel_count, chunk_size):
        stop = min(start + chunk_size, pixel_count)
        chunk_colors = patches[start:stop]
        chunk_indices = neighbor_indices[start:stop]
        chunk_valid = valid[start:stop]
        valid_f = chunk_valid.to(dtype=dtype)
        window_size = valid_f.sum(dim=1).clamp_min(1.0)
        mean = (chunk_colors * valid_f.unsqueeze(-1)).sum(dim=1) / window_size.unsqueeze(-1)
        centered = (chunk_colors - mean.unsqueeze(1)) * valid_f.unsqueeze(-1)
        covariance = torch.einsum("nki,nkj->nij", centered, centered) / window_size.view(-1, 1, 1)
        regularized = covariance + (epsilon / window_size).view(-1, 1, 1) * eye_channels
        inverse = torch.linalg.pinv(regularized)
        affinity = (
            1.0 + torch.einsum("nki,nij,nlj->nkl", centered, inverse, centered)
        ) / window_size.view(-1, 1, 1)
        local_values = eye_window.unsqueeze(0) - affinity
        pair_valid = chunk_valid.unsqueeze(2) & chunk_valid.unsqueeze(1)
        rows.append(
            chunk_indices.unsqueeze(2).expand(-1, -1, window_area)[pair_valid].to(torch.long)
        )
        cols.append(
            chunk_indices.unsqueeze(1).expand(-1, window_area, -1)[pair_valid].to(torch.long)
        )
        values.append(local_values[pair_valid])

    indices = torch.stack((torch.cat(rows), torch.cat(cols)), dim=0)
    data = torch.cat(values)
    coo = torch.sparse_coo_tensor(
        indices,
        data,
        size=(pixel_count, pixel_count),
        device=device,
        dtype=dtype,
    ).coalesce()
    coo_indices = coo.indices()
    coo_values = coo.values()
    diagonal = torch.zeros(pixel_count, dtype=dtype, device=device)
    diag_mask = coo_indices[0] == coo_indices[1]
    diagonal.index_add_(0, coo_indices[0, diag_mask], coo_values[diag_mask])
    return coo.to_sparse_csr(), diagonal


def _solve_levin_gpu(
    laplacian: torch.Tensor,
    laplacian_diagonal: torch.Tensor,
    trimap: torch.Tensor,
    *,
    lambda_constraint: float,
    rtol: float = 1e-6,
    maxiter: int = 2000,
) -> torch.Tensor:
    """Solve Levin constraints on CUDA via Jacobi-preconditioned CG."""

    flat_trimap = trimap.reshape(-1).to(dtype=torch.float64)
    known = (flat_trimap >= 0.95) | (flat_trimap <= 0.05)
    if not bool(known.any()):
        return torch.full_like(flat_trimap, 0.5)

    known_f = known.to(dtype=torch.float64)
    target = (flat_trimap >= 0.95).to(dtype=torch.float64)
    rhs = lambda_constraint * target
    diagonal = (laplacian_diagonal + lambda_constraint * known_f).clamp_min(1e-12)
    inv_diagonal = diagonal.reciprocal()

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        sparse_part = torch.sparse.mm(laplacian, vector.unsqueeze(1)).squeeze(1)
        return sparse_part + lambda_constraint * known_f * vector

    norm_rhs = torch.linalg.vector_norm(rhs).clamp_min(1.0)
    x = rhs * inv_diagonal
    residual = rhs - matvec(x)
    z = inv_diagonal * residual
    direction = z.clone()
    rz_old = torch.dot(residual, z)

    for _ in range(maxiter):
        if torch.linalg.vector_norm(residual) <= rtol * norm_rhs:
            return torch.nan_to_num(x, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        matrix_direction = matvec(direction)
        denominator = torch.dot(direction, matrix_direction)
        if denominator.abs() <= 1e-20:
            break
        alpha = rz_old / denominator
        x = x + alpha * direction
        residual = residual - alpha * matrix_direction
        z = inv_diagonal * residual
        rz_new = torch.dot(residual, z)
        if rz_new.abs() <= 1e-30:
            return torch.nan_to_num(x, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        beta = rz_new / rz_old
        direction = z + beta * direction
        rz_old = rz_new

    raise RuntimeError("gpu cg failed to converge")


def _solve_single_gpu(
    trimap: torch.Tensor,
    guide: torch.Tensor,
    *,
    epsilon: float,
    window_radius: int,
    lambda_constraint: float,
) -> torch.Tensor:
    fg = trimap >= 0.95
    bg = trimap <= 0.05
    unknown = ~(fg | bg)
    alpha = torch.zeros_like(trimap, dtype=torch.float32)
    alpha[fg] = 1.0
    if not bool(unknown.any()):
        return alpha

    laplacian, diagonal = _build_matting_laplacian_gpu(
        guide,
        epsilon=epsilon,
        window_radius=window_radius,
    )
    solved = _solve_levin_gpu(
        laplacian,
        diagonal,
        trimap,
        lambda_constraint=lambda_constraint,
    )
    alpha = solved.reshape(trimap.shape).to(dtype=torch.float32)
    alpha[fg] = 1.0
    alpha[bg] = 0.0
    return alpha


def alpha_matte_extract(
    trimap: torch.Tensor,
    guide: torch.Tensor,
    *,
    epsilon: float = 1e-7,
    window_radius: int = 1,
    lambda_constraint: float = 100.0,
    compute_device: Literal["auto", "cuda", "cpu"] = "auto",
) -> torch.Tensor:
    """Extract a soft alpha matte via Levin 2008 closed-form matting.

    Args:
        trimap: MASK tensor ``(B,H,W)`` encoded as ``0.0`` BG, ``0.5`` Unknown,
            and ``1.0`` FG. Tolerance is ±0.05.
        guide: RGB image tensor ``(B,H,W,3)`` used to build the local color
            covariance matting Laplacian.
        epsilon: Levin covariance regularizer. The default ``1e-7`` follows the
            paper's canonical setting for normalized RGB inputs.
        window_radius: Radius of local windows used by the matting Laplacian.
        lambda_constraint: Strength of known foreground/background trimap constraints.
        compute_device: ``"auto"`` uses CUDA when the guide tensor is already on CUDA;
            ``"cuda"`` forces the GPU path; ``"cpu"`` preserves the v1.2.0 CPU path.

    Returns:
        Soft alpha matte tensor ``(B,H,W)`` in ``[0,1]``.

    Raises:
        TypeError: If inputs are not tensors.
        ValueError: If shapes, trimap encoding, or parameters are invalid.
        RuntimeError: If both sparse direct solve and CG fallback fail.
    """

    trimap_prepared = validate_trimap(trimap, tolerance=0.05)
    guide_prepared = _validate_image(guide)
    if guide_prepared.shape[0] not in (1, trimap_prepared.shape[0]):
        raise ValueError(f"guide batch must be 1 or {trimap_prepared.shape[0]}.")
    if guide_prepared.shape[1:3] != trimap_prepared.shape[1:3]:
        raise ValueError("guide and trimap spatial shapes must match.")
    if guide_prepared.shape[0] == 1 and trimap_prepared.shape[0] > 1:
        guide_prepared = guide_prepared.expand(trimap_prepared.shape[0], -1, -1, -1)
    epsilon = _validate_float("epsilon", epsilon, lower=1e-8, upper=1e-2)
    window_radius = _validate_int("window_radius", window_radius, lower=1, upper=3)
    lambda_constraint = _validate_float(
        "lambda_constraint",
        lambda_constraint,
        lower=1.0,
        upper=10000.0,
    )
    compute_device = _validate_compute_device(compute_device)

    outputs = []
    use_cuda = compute_device == "cuda" or (
        compute_device == "auto" and guide_prepared.is_cuda and torch.cuda.is_available()
    )
    for batch_index in range(trimap_prepared.shape[0]):
        if use_cuda:
            try:
                alpha_gpu = _solve_single_gpu(
                    trimap_prepared[batch_index].detach().to("cuda"),
                    guide_prepared[batch_index].detach().to("cuda"),
                    epsilon=epsilon,
                    window_radius=window_radius,
                    lambda_constraint=lambda_constraint,
                )
                outputs.append(alpha_gpu.detach().cpu())
                continue
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                warnings.warn(f"GPU path fallback to CPU: {exc}", stacklevel=2)
        alpha = _solve_single(
            trimap_prepared[batch_index].detach().cpu().numpy(),
            guide_prepared[batch_index].detach().cpu().numpy(),
            epsilon=epsilon,
            window_radius=window_radius,
            lambda_constraint=lambda_constraint,
        )
        outputs.append(torch.from_numpy(alpha))
    return torch.stack(outputs, dim=0).to(device=trimap.device, dtype=torch.float32)
