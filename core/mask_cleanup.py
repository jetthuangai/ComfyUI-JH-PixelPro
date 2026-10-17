"""Mask cleanup: drop specks, fill small holes, optional distance-based feather."""

from __future__ import annotations

from numbers import Integral, Real

import cv2
import numpy as np
import torch

_MAX_AREA = 1_000_000
_MAX_FEATHER = 256.0


def _validate_mask(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor.")
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (B,H,W), got {tuple(mask.shape)}.")
    if mask.shape[0] < 1 or mask.shape[1] < 1 or mask.shape[2] < 1:
        raise ValueError(f"mask must not be empty, got {tuple(mask.shape)}.")
    if not torch.is_floating_point(mask) and mask.dtype is not torch.bool:
        raise ValueError(f"mask must be float or bool, got {mask.dtype}.")
    return mask


def _validate_float(name: str, value: float, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number in [{lower}, {upper}].")
    value_f = float(value)
    if not np.isfinite(value_f) or not lower <= value_f <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_f


def _validate_int(name: str, value: int, *, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}].")
    value_int = int(value)
    if not lower <= value_int <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_int


def _remove_small_components(binary: np.ndarray, min_area: int, connectivity: int) -> np.ndarray:
    """Zero out foreground components (value 1) whose area is below ``min_area``."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=connectivity, ltype=cv2.CV_32S
    )
    if count <= 1:
        return binary
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False  # label 0 is the background
    return keep[labels].astype(np.uint8)


def _fill_small_holes(binary: np.ndarray, max_area: int, connectivity: int) -> np.ndarray:
    """Fill background components smaller than ``max_area`` that do not touch the border."""
    height, width = binary.shape
    inverse = 1 - binary
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverse, connectivity=connectivity, ltype=cv2.CV_32S
    )
    if count <= 1:
        return binary
    left = stats[:, cv2.CC_STAT_LEFT]
    top = stats[:, cv2.CC_STAT_TOP]
    right = left + stats[:, cv2.CC_STAT_WIDTH]
    bottom = top + stats[:, cv2.CC_STAT_HEIGHT]
    touches_border = (left == 0) | (top == 0) | (right == width) | (bottom == height)
    fill = (stats[:, cv2.CC_STAT_AREA] < max_area) & ~touches_border
    fill[0] = False  # label 0 is the foreground here
    return (binary.astype(bool) | fill[labels]).astype(np.uint8)


def _feather(binary: np.ndarray, feather_px: float) -> np.ndarray:
    """Map the binary mask to alpha through a signed distance to its boundary."""
    # The boundary runs between pixel centres, so a pixel next to it is 0.5 px away.
    inside = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    signed = np.where(binary > 0, inside - 0.5, -(outside - 0.5))
    alpha = 0.5 + signed / (2.0 * feather_px)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def mask_cleanup(
    mask: torch.Tensor,
    *,
    threshold: float = 0.5,
    min_island_area: int = 64,
    max_hole_area: int = 64,
    feather_px: float = 0.0,
    connectivity: int = 8,
) -> torch.Tensor:
    """Clean a segmentation mask: remove specks, fill small holes, optionally feather.

    The mask is binarised at ``> threshold``. Foreground components smaller than
    ``min_island_area`` are removed, then background components smaller than
    ``max_hole_area`` that do not touch the image border are filled. Connected
    components use ``cv2.connectedComponentsWithStats`` (exact and fast);
    ``kornia.contrib.connected_components`` is an iterative label propagation that is
    slow on large masks. Holes use the complementary connectivity (8 -> 4, 4 -> 8), the
    standard digital-topology pairing: an 8-connected outline encloses its hole only when the
    background is traced 4-connected, otherwise the hole leaks out diagonally.

    If ``feather_px > 0`` the result is converted to a soft alpha from the signed
    distance ``sd`` to the binary boundary (positive inside):
    ``alpha = clip(0.5 + sd / (2 * feather_px), 0, 1)``. The 0.5 contour stays on the
    binary boundary and alpha reaches 0 / 1 at ``feather_px`` pixels outside / inside,
    so the full ramp spans ``2 * feather_px`` pixels.

    Args:
        mask: MASK tensor ``(B,H,W)``, float or bool.
        threshold: Binarisation threshold in ``[0, 1]`` (``mask > threshold``).
        min_island_area: Remove foreground components with fewer pixels; 0 disables.
        max_hole_area: Fill enclosed holes with fewer pixels; 0 disables.
        feather_px: Feather distance in pixels in ``[0, 256]``; 0 gives a hard mask.
        connectivity: 4 or 8.

    Returns:
        float32 MASK ``(B,H,W)`` in ``[0, 1]`` on the input device.

    Raises:
        TypeError: If mask is not a tensor.
        ValueError: If shape, dtype or any parameter is invalid.
    """

    mask = _validate_mask(mask)
    threshold = _validate_float("threshold", threshold, lower=0.0, upper=1.0)
    min_island_area = _validate_int("min_island_area", min_island_area, lower=0, upper=_MAX_AREA)
    max_hole_area = _validate_int("max_hole_area", max_hole_area, lower=0, upper=_MAX_AREA)
    feather_px = _validate_float("feather_px", feather_px, lower=0.0, upper=_MAX_FEATHER)
    connectivity = _validate_int("connectivity", connectivity, lower=4, upper=8)
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8.")

    batch = (mask.detach() > threshold).to("cpu").numpy().astype(np.uint8)
    outputs = []
    for item in batch:
        binary = np.ascontiguousarray(item)
        if min_island_area > 0:
            binary = _remove_small_components(binary, min_island_area, connectivity)
        if max_hole_area > 0:
            binary = _fill_small_holes(binary, max_hole_area, 12 - connectivity)
        result = _feather(binary, feather_px) if feather_px > 0.0 else binary.astype(np.float32)
        outputs.append(torch.from_numpy(result))
    return torch.stack(outputs, dim=0).to(device=mask.device, dtype=torch.float32)
