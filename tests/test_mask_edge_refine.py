from __future__ import annotations

import pytest
import torch

from core.mask_edge_refine import edge_aware_refine


def _mask(size: int = 64) -> torch.Tensor:
    out = torch.zeros((1, size, size), dtype=torch.float32)
    out[:, size // 4 : size * 3 // 4, size // 4 : size * 3 // 4] = 1.0
    return out


def _guide(size: int = 64) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, size).view(1, 1, size, 1).expand(1, size, size, 1)
    return torch.cat((x, x.flip(2), torch.ones_like(x) * 0.5), dim=-1)


def test_edge_aware_refine_preserves_shape_and_range() -> None:
    mask = _mask()
    out = edge_aware_refine(mask, _guide(), radius=4, eps=1e-3)
    assert out.shape == mask.shape
    assert out.dtype == torch.float32
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_edge_aware_refine_preserves_bc1hw_rank() -> None:
    mask = _mask().unsqueeze(1)
    out = edge_aware_refine(mask, _guide(), radius=3, eps=1e-3)
    assert out.shape == mask.shape
    assert out.ndim == 4


def test_edge_aware_refine_supports_batch_broadcast_guide() -> None:
    mask = _mask().repeat(2, 1, 1)
    out = edge_aware_refine(mask, _guide(), radius=3, eps=1e-3)
    assert out.shape == mask.shape


def test_edge_aware_refine_feather_changes_output() -> None:
    mask = _mask()
    guide = _guide()
    plain = edge_aware_refine(mask, guide, radius=4, eps=1e-3, feather_sigma=0.0)
    feathered = edge_aware_refine(mask, guide, radius=4, eps=1e-3, feather_sigma=1.5)
    assert not torch.allclose(plain, feathered)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"radius": 0}, "radius"),
        ({"eps": 0.0}, "eps"),
        ({"feather_sigma": -1.0}, "feather_sigma"),
    ],
)
def test_edge_aware_refine_validates_params(kwargs: dict[str, float | int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        edge_aware_refine(_mask(), _guide(), **kwargs)


def test_edge_aware_refine_rejects_spatial_mismatch() -> None:
    with pytest.raises(ValueError, match="spatial"):
        edge_aware_refine(_mask(64), _guide(32))
