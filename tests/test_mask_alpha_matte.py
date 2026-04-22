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


def _circle_case(size: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = torch.arange(size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    distance = torch.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2)
    target = torch.clamp((22.0 - distance) / 6.0, 0.0, 1.0)
    trimap = torch.full((1, size, size), 0.5, dtype=torch.float32)
    trimap[:, target >= 0.95] = 1.0
    trimap[:, target <= 0.05] = 0.0
    guide = target.view(1, size, size, 1).expand(1, size, size, 3).contiguous()
    return trimap, guide, target.unsqueeze(0)


def _alpha_gradient_norm(alpha: torch.Tensor) -> float:
    dx = alpha[:, :, 1:] - alpha[:, :, :-1]
    dy = alpha[:, 1:, :] - alpha[:, :-1, :]
    return float(torch.sqrt((dx.square().mean() + dy.square().mean()).clamp_min(0.0)))


def test_alpha_matte_preserves_shape_and_range() -> None:
    trimap = _trimap()
    alpha = alpha_matte_extract(trimap, _guide(), epsilon=1e-7, window_radius=1)
    assert alpha.shape == trimap.shape
    assert alpha.min().item() >= 0.0
    assert alpha.max().item() <= 1.0


def test_known_foreground_and_background_are_pinned() -> None:
    alpha = alpha_matte_extract(_trimap(), _guide(), epsilon=1e-7, window_radius=1)
    assert alpha[0, 12, 12].item() == pytest.approx(1.0, abs=1e-6)
    assert alpha[0, 0, 0].item() == pytest.approx(0.0, abs=1e-6)


def test_no_unknown_returns_binary_alpha() -> None:
    trimap = torch.zeros((1, 12, 12), dtype=torch.float32)
    trimap[:, 3:9, 3:9] = 1.0
    alpha = alpha_matte_extract(trimap, _guide(12))
    assert torch.equal(alpha, trimap)


def test_alpha_matte_broadcasts_single_guide() -> None:
    trimap = _trimap().repeat(2, 1, 1)
    alpha = alpha_matte_extract(trimap, _guide(), epsilon=1e-7, window_radius=1)
    assert alpha.shape == trimap.shape


def test_alpha_matte_rejects_bad_trimap_values() -> None:
    trimap = _trimap()
    trimap[:, 4, 4] = 0.25
    with pytest.raises(ValueError, match="trimap"):
        alpha_matte_extract(trimap, _guide())


def test_alpha_matte_rejects_spatial_mismatch() -> None:
    with pytest.raises(ValueError, match="spatial"):
        alpha_matte_extract(_trimap(24), _guide(20))


def test_alpha_matte_rejects_bad_lambda_constraint() -> None:
    with pytest.raises(ValueError, match="lambda_constraint"):
        alpha_matte_extract(_trimap(), _guide(), lambda_constraint=0.5)


def test_levin_synthetic_circle_mse() -> None:
    trimap, guide, target = _circle_case()
    alpha = alpha_matte_extract(
        trimap, guide, epsilon=1e-7, window_radius=1, lambda_constraint=100.0
    )
    mse = torch.mean((alpha - target).square()).item()
    assert mse <= 0.02


def test_levin_linear_gradient_bg_is_color_aware() -> None:
    size = 64
    x = torch.linspace(0.0, 0.5, size).view(1, 1, size, 1).expand(1, size, size, 1)
    guide = torch.cat((x, x, x), dim=-1).clone()
    guide[:, 22:42, 22:42] = 1.0
    trimap = torch.zeros((1, size, size), dtype=torch.float32)
    trimap[:, 18:46, 18:46] = 0.5
    trimap[:, 24:40, 24:40] = 1.0
    unknown = trimap == 0.5
    alpha = alpha_matte_extract(
        trimap, guide, epsilon=1e-7, window_radius=1, lambda_constraint=100.0
    )
    assert alpha[unknown].std().item() > 0.05
    assert alpha[unknown].max().item() - alpha[unknown].min().item() > 0.3


def test_epsilon_regularization_monotonic() -> None:
    trimap, guide, _target = _circle_case()
    norms = [
        _alpha_gradient_norm(
            alpha_matte_extract(
                trimap,
                guide,
                epsilon=epsilon,
                window_radius=1,
                lambda_constraint=100.0,
            )
        )
        for epsilon in (1e-8, 1e-6, 1e-4)
    ]
    assert norms[0] >= norms[1] - 1e-5
    assert norms[1] >= norms[2] - 1e-5
