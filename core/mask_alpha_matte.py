"""Classical sparse alpha-matte extraction from a 3-value trimap MASK."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import torch
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


def _validate_image(guide: torch.Tensor) -> torch.Tensor:
    if not isinstance(guide, torch.Tensor):
        raise TypeError("guide must be a torch.Tensor.")
    if guide.ndim != 4 or guide.shape[-1] != 3:
        raise ValueError(f"guide must have shape (B,H,W,3), got {tuple(guide.shape)}.")
    if not torch.is_floating_point(guide):
        raise ValueError(f"guide must be floating point, got {guide.dtype}.")
    return guide.to(dtype=torch.float32).clamp(0.0, 1.0)


def _validate_trimap_tensor(trimap: torch.Tensor) -> torch.Tensor:
    if not isinstance(trimap, torch.Tensor):
        raise TypeError("trimap must be a torch.Tensor.")
    if trimap.ndim != 3:
        raise ValueError(f"trimap must have shape (B,H,W), got {tuple(trimap.shape)}.")
    if not torch.is_floating_point(trimap) and trimap.dtype is not torch.bool:
        raise ValueError(f"trimap must be float or bool, got {trimap.dtype}.")
    prepared = trimap.to(dtype=torch.float32).clamp(0.0, 1.0)
    bg = (prepared - 0.0).abs() <= 0.05
    unknown = (prepared - 0.5).abs() <= 0.05
    fg = (prepared - 1.0).abs() <= 0.05
    if not (bg | unknown | fg).all():
        raise ValueError("trimap must use 0.0 BG / 0.5 Unknown / 1.0 FG values.")
    return prepared


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


def _solve_single(
    trimap_np: np.ndarray,
    guide_np: np.ndarray,
    *,
    epsilon: float,
    window_radius: int,
) -> np.ndarray:
    fg = trimap_np >= 0.95
    bg = trimap_np <= 0.05
    unknown = ~(fg | bg)
    alpha = np.zeros_like(trimap_np, dtype=np.float32)
    alpha[fg] = 1.0
    if not unknown.any():
        return alpha

    unknown_coords = np.argwhere(unknown)
    index_map = -np.ones(trimap_np.shape, dtype=np.int64)
    index_map[unknown] = np.arange(unknown_coords.shape[0], dtype=np.int64)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(unknown_coords.shape[0], dtype=np.float64)
    sigma = max(float(epsilon) * 255.0, 1e-4)
    height, width = trimap_np.shape

    for row_index, (y, x) in enumerate(unknown_coords):
        center_color = guide_np[y, x]
        weights: list[tuple[int, int, float]] = []
        for yy in range(max(0, y - window_radius), min(height, y + window_radius + 1)):
            for xx in range(max(0, x - window_radius), min(width, x + window_radius + 1)):
                if yy == y and xx == x:
                    continue
                diff = guide_np[yy, xx] - center_color
                weight = float(np.exp(-float(diff.dot(diff)) / (sigma + 1e-6)))
                weights.append((yy, xx, weight))
        total = sum(weight for _, _, weight in weights) or 1.0
        rows.append(row_index)
        cols.append(row_index)
        data.append(1.0)
        for yy, xx, weight in weights:
            normalized = weight / total
            unknown_index = index_map[yy, xx]
            if unknown_index >= 0:
                rows.append(row_index)
                cols.append(int(unknown_index))
                data.append(-normalized)
            elif fg[yy, xx]:
                rhs[row_index] += normalized

    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(unknown_coords.shape[0],) * 2)
    try:
        solved = sparse_linalg.spsolve(matrix, rhs)
    except Exception:
        solved, _info = sparse_linalg.cg(matrix, rhs, maxiter=500)
    alpha[unknown] = np.nan_to_num(solved, nan=0.5, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    return alpha.astype(np.float32, copy=False)


def alpha_matte_extract(
    trimap: torch.Tensor,
    guide: torch.Tensor,
    *,
    epsilon: float = 1e-4,
    window_radius: int = 1,
) -> torch.Tensor:
    """Extract a soft alpha matte from a 3-value trimap and RGB guide image.

    Args:
        trimap: MASK tensor ``(B,H,W)`` encoded as ``0.0`` BG, ``0.5`` Unknown,
            and ``1.0`` FG. Tolerance is ±0.05.
        guide: RGB image tensor ``(B,H,W,3)`` used for edge-aware sparse weights.
        epsilon: Color-distance regularizer for the sparse diffusion weights.
        window_radius: Neighborhood radius used to connect Unknown pixels.

    Returns:
        Soft alpha matte tensor ``(B,H,W)`` in ``[0,1]``.

    Raises:
        TypeError: If inputs are not tensors.
        ValueError: If shapes, trimap encoding, or parameters are invalid.
    """

    trimap_prepared = _validate_trimap_tensor(trimap)
    guide_prepared = _validate_image(guide)
    if guide_prepared.shape[0] not in (1, trimap_prepared.shape[0]):
        raise ValueError(f"guide batch must be 1 or {trimap_prepared.shape[0]}.")
    if guide_prepared.shape[1:3] != trimap_prepared.shape[1:3]:
        raise ValueError("guide and trimap spatial shapes must match.")
    if guide_prepared.shape[0] == 1 and trimap_prepared.shape[0] > 1:
        guide_prepared = guide_prepared.expand(trimap_prepared.shape[0], -1, -1, -1)
    epsilon = _validate_float("epsilon", epsilon, lower=1e-8, upper=1e-2)
    window_radius = _validate_int("window_radius", window_radius, lower=1, upper=3)

    outputs = []
    for batch_index in range(trimap_prepared.shape[0]):
        alpha = _solve_single(
            trimap_prepared[batch_index].detach().cpu().numpy(),
            guide_prepared[batch_index].detach().cpu().numpy(),
            epsilon=epsilon,
            window_radius=window_radius,
        )
        outputs.append(torch.from_numpy(alpha))
    return torch.stack(outputs, dim=0).to(device=trimap.device, dtype=torch.float32)
