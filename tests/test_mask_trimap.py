from __future__ import annotations

import pytest
import torch

from core.mask_trimap import build_trimap, validate_trimap


def _mask(size: int = 32) -> torch.Tensor:
    mask = torch.zeros((1, size, size), dtype=torch.float32)
    mask[:, 8:24, 8:24] = 1.0
    return mask


def test_build_trimap_uses_three_values() -> None:
    trimap = build_trimap(_mask(), fg_radius=2, bg_radius=4)
    assert trimap.shape == _mask().shape
    assert set(torch.unique(trimap).tolist()) <= {0.0, 0.5, 1.0}
    assert (trimap == 0.5).any()


def test_validate_trimap_accepts_tolerance() -> None:
    trimap = torch.tensor([[[0.0, 0.48, 1.0]]], dtype=torch.float32)
    out = validate_trimap(trimap, tolerance=0.05)
    assert torch.equal(out, trimap)


def test_validate_trimap_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="trimap"):
        validate_trimap(torch.tensor([[[0.25]]], dtype=torch.float32))


def test_build_trimap_smoothing_preserves_shape() -> None:
    trimap = build_trimap(_mask(), fg_radius=2, bg_radius=4, smoothing=1.0)
    assert trimap.shape == _mask().shape


def test_build_trimap_rejects_bad_radius() -> None:
    with pytest.raises(ValueError, match="fg_radius"):
        build_trimap(_mask(), fg_radius=0, bg_radius=4)
