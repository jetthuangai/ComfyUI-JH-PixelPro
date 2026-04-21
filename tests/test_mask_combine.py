from __future__ import annotations

import pytest
import torch

from core.mask_combine import MASK_COMBINE_OPERATIONS, combine_masks


def _masks() -> tuple[torch.Tensor, torch.Tensor]:
    mask_a = torch.zeros((1, 16, 16), dtype=torch.float32)
    mask_b = torch.zeros((1, 16, 16), dtype=torch.float32)
    mask_a[:, 4:12, 4:12] = 1.0
    mask_b[:, 8:14, 8:14] = 1.0
    return mask_a, mask_b


def test_all_operations_preserve_shape_and_range() -> None:
    mask_a, mask_b = _masks()
    for operation in MASK_COMBINE_OPERATIONS:
        out = combine_masks(mask_a, mask_b, operation=operation)
        assert out.shape == mask_a.shape
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0


def test_union_matches_maximum() -> None:
    mask_a, mask_b = _masks()
    out = combine_masks(mask_a, mask_b, operation="union")
    assert torch.equal(out, torch.maximum(mask_a, mask_b))


def test_difference_removes_overlap() -> None:
    mask_a, mask_b = _masks()
    out = combine_masks(mask_a, mask_b, operation="difference")
    assert out[:, 8:12, 8:12].sum().item() == 0.0
    assert out.sum().item() < mask_a.sum().item()


def test_opacity_blends_from_mask_a() -> None:
    mask_a, mask_b = _masks()
    out = combine_masks(mask_a, mask_b, operation="union", opacity=0.5)
    assert torch.allclose(out, mask_a * 0.5 + torch.maximum(mask_a, mask_b) * 0.5)


def test_soft_feather_preserves_shape() -> None:
    mask_a, mask_b = _masks()
    out = combine_masks(
        mask_a,
        mask_b,
        operation="union",
        blend_mode="soft_feather",
        feather_sigma=1.0,
    )
    assert out.shape == mask_a.shape
    assert 0.0 < out[0, 3, 8].item() < 1.0


def test_rejects_invalid_operation() -> None:
    mask_a, mask_b = _masks()
    with pytest.raises(ValueError, match="operation"):
        combine_masks(mask_a, mask_b, operation="invalid")
