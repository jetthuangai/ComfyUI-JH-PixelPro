from __future__ import annotations

import pytest
import torch

from core.mask_morphology import MORPHOLOGY_OPERATIONS, mask_morphology


def _mask(size: int = 32) -> torch.Tensor:
    mask = torch.zeros((1, size, size), dtype=torch.float32)
    mask[:, 10:22, 10:22] = 1.0
    return mask


def test_all_operations_preserve_shape_and_range() -> None:
    mask = _mask()
    for operation in MORPHOLOGY_OPERATIONS:
        out = mask_morphology(mask, operation=operation, radius=2, iterations=1)
        assert out.shape == mask.shape
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0


def test_dilate_grows_mask_area() -> None:
    mask = _mask()
    out = mask_morphology(mask, operation="dilate", radius=2, iterations=1)
    assert out.sum().item() > mask.sum().item()


def test_erode_shrinks_mask_area() -> None:
    mask = _mask()
    out = mask_morphology(mask, operation="erode", radius=2, iterations=1)
    assert out.sum().item() < mask.sum().item()


def test_gradient_extracts_boundary() -> None:
    mask = _mask()
    out = mask_morphology(mask, operation="gradient", radius=1, iterations=1)
    assert out.sum().item() > 0.0
    assert out[:, 14:18, 14:18].sum().item() == 0.0


def test_rejects_invalid_operation() -> None:
    with pytest.raises(ValueError, match="operation"):
        mask_morphology(_mask(), operation="invalid", radius=2, iterations=1)


def test_rejects_invalid_radius() -> None:
    with pytest.raises(ValueError, match="radius"):
        mask_morphology(_mask(), operation="dilate", radius=0, iterations=1)
