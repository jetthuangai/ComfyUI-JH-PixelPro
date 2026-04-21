from __future__ import annotations

import pytest
import torch

from core.mask_edge_smooth import mask_edge_smooth


def _mask(size: int = 32) -> torch.Tensor:
    mask = torch.zeros((1, size, size), dtype=torch.float32)
    mask[:, 8:24, 8:24] = 1.0
    return mask


def _guide(size: int = 32) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, size)
    guide = x.view(1, 1, size, 1).expand(1, size, size, 3).contiguous()
    return guide


def test_smooth_without_guide_preserves_shape_and_range() -> None:
    mask = _mask()
    out = mask_edge_smooth(mask, sigma_spatial=3.0, sigma_range=0.1, iterations=1)
    assert out.shape == mask.shape
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_smooth_with_guide_preserves_shape_and_range() -> None:
    mask = _mask()
    out = mask_edge_smooth(mask, _guide(), sigma_spatial=3.0, sigma_range=0.1, iterations=1)
    assert out.shape == mask.shape
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_softens_binary_edge() -> None:
    mask = _mask()
    out = mask_edge_smooth(mask, sigma_spatial=5.0, sigma_range=1.0, iterations=1)
    assert out[0, 8, 16].item() < 1.0
    assert out[0, 7, 16].item() > 0.0


def test_broadcasts_single_guide() -> None:
    mask = _mask().expand(2, -1, -1).clone()
    out = mask_edge_smooth(mask, _guide(), sigma_spatial=3.0, sigma_range=0.1, iterations=1)
    assert out.shape == mask.shape


def test_rejects_bad_guide_shape() -> None:
    with pytest.raises(ValueError, match="guide"):
        mask_edge_smooth(_mask(), torch.zeros((1, 32, 32)))


def test_rejects_bad_iterations() -> None:
    with pytest.raises(ValueError, match="iterations"):
        mask_edge_smooth(_mask(), iterations=0)
