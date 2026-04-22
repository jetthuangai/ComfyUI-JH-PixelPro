"""Closed-form alpha-matte extraction via Levin 2008 matting Laplacian."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import torch
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


def alpha_matte_extract(
    trimap: torch.Tensor,
    guide: torch.Tensor,
    *,
    epsilon: float = 1e-7,
    window_radius: int = 1,
    lambda_constraint: float = 100.0,
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

    outputs = []
    for batch_index in range(trimap_prepared.shape[0]):
        alpha = _solve_single(
            trimap_prepared[batch_index].detach().cpu().numpy(),
            guide_prepared[batch_index].detach().cpu().numpy(),
            epsilon=epsilon,
            window_radius=window_radius,
            lambda_constraint=lambda_constraint,
        )
        outputs.append(torch.from_numpy(alpha))
    return torch.stack(outputs, dim=0).to(device=trimap.device, dtype=torch.float32)
