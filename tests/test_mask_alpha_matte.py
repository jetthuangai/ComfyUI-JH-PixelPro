from __future__ import annotations

import pytest
import torch

from core.mask_alpha_matte import alpha_matte_extract


def _trimap(size: int = 24) -> torch.Tensor:
    trimap = torch.zeros((1, size, size), dtype=torch.float32)
    trimap[:, 6:18, 6:18] = 0.5
    trimap[:, 9:15, 9:15] = 1.0
    return trimap


def _guide(size: int = 24) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, size).view(1, size, 1, 1).expand(1, size, size, 1)
    x = torch.linspace(0.0, 1.0, size).view(1, 1, size, 1).expand(1, size, size, 1)
    return torch.cat((x, y, torch.ones_like(x) * 0.5), dim=-1)


def test_alpha_matte_preserves_shape_and_range() -> None:
    trimap = _trimap()
    alpha = alpha_matte_extract(trimap, _guide(), epsilon=1e-4, window_radius=1)
    assert alpha.shape == trimap.shape
    assert alpha.min().item() >= 0.0
    assert alpha.max().item() <= 1.0


def test_known_foreground_and_background_are_pinned() -> None:
    alpha = alpha_matte_extract(_trimap(), _guide(), epsilon=1e-4, window_radius=1)
    assert alpha[0, 12, 12].item() == pytest.approx(1.0, abs=1e-6)
    assert alpha[0, 0, 0].item() == pytest.approx(0.0, abs=1e-6)


def test_no_unknown_returns_binary_alpha() -> None:
    trimap = torch.zeros((1, 12, 12), dtype=torch.float32)
    trimap[:, 3:9, 3:9] = 1.0
    alpha = alpha_matte_extract(trimap, _guide(12))
    assert torch.equal(alpha, trimap)


def test_alpha_matte_broadcasts_single_guide() -> None:
    trimap = _trimap().repeat(2, 1, 1)
    alpha = alpha_matte_extract(trimap, _guide(), epsilon=1e-4, window_radius=1)
    assert alpha.shape == trimap.shape


def test_alpha_matte_rejects_bad_trimap_values() -> None:
    trimap = _trimap()
    trimap[:, 4, 4] = 0.25
    with pytest.raises(ValueError, match="trimap"):
        alpha_matte_extract(trimap, _guide())


def test_alpha_matte_rejects_spatial_mismatch() -> None:
    with pytest.raises(ValueError, match="spatial"):
        alpha_matte_extract(_trimap(24), _guide(20))
