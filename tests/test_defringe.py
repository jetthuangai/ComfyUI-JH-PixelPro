"""Defringe core (``core/defringe.py``)."""

from __future__ import annotations

import pytest
import torch

from core.defringe import correct_lateral_ca, defringe


def _board(h: int = 600, w: int = 800) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    board = (((yy // 40) + (xx // 40)) % 2).float() * 0.8 + 0.1
    return board[None, None].expand(1, 3, h, w).contiguous()


def test_lateral_ca_is_measured_and_undone() -> None:
    clean = _board()
    shifted = correct_lateral_ca(clean, 1 / 1.0035, 1 / 0.9965)
    out, scales = defringe(shifted.permute(0, 2, 3, 1), purple=0.0)
    red, blue = scales[0]
    assert red == pytest.approx(1.0035, abs=3e-4) and blue == pytest.approx(0.9965, abs=3e-4)
    corner = (slice(None), slice(0, 120), slice(0, 160))
    before = (shifted.permute(0, 2, 3, 1) - clean.permute(0, 2, 3, 1))[corner].abs().mean()
    after = (out - clean.permute(0, 2, 3, 1))[corner].abs().mean()
    assert after < 0.4 * before


def test_clean_image_is_untouched() -> None:
    image = _board().permute(0, 2, 3, 1)
    out, scales = defringe(image)
    assert scales == [(1.0, 1.0)]
    assert torch.equal(out, image)


def _branch_on_sky() -> torch.Tensor:
    h, w = 200, 300
    xx = torch.arange(w).view(1, w).expand(h, w)
    sky = torch.tensor([0.75, 0.85, 0.95]).view(3, 1, 1).expand(3, h, w).clone()
    branch = (xx - 150).abs() < 6
    halo = ((xx - 150).abs() < 10) & ~branch
    sky[:, branch] = 0.08
    sky[:, halo] = torch.tensor([0.55, 0.25, 0.75]).view(3, 1)
    return sky[None].permute(0, 2, 3, 1)


def test_purple_fringe_is_neutralised_lightness_kept() -> None:
    image = _branch_on_sky()
    out, _ = defringe(image, lateral_ca=False)
    halo = out[0, 100, 142]
    assert float(halo[2] - halo[1]) < 0.1  # no longer purple (blue far above green)
    lum = lambda p: float(0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])  # noqa: E731
    assert abs(lum(halo) - lum(image[0, 100, 142])) < 0.08
    assert torch.allclose(out[0, :, :60], image[0, :, :60])  # far sky unchanged
    assert torch.allclose(out[0, :, 146:155], image[0, :, 146:155])  # dark branch unchanged


def test_strength_zero_does_nothing() -> None:
    image = _branch_on_sky()
    out, _ = defringe(image, purple=0.0, green=0.0, lateral_ca=False)
    assert torch.equal(out, image)


def test_purple_flower_away_from_edges_kept() -> None:
    flower = torch.tensor([0.6, 0.3, 0.8]).view(1, 1, 1, 3).expand(1, 64, 64, 3).contiguous()
    out, _ = defringe(flower, lateral_ca=False)
    assert torch.allclose(out, flower, atol=1e-5)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"purple": 2.0}, "purple"), ({"green": -1.0}, "green"), ({"radius": 99}, "radius")],
)
def test_validates(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        defringe(torch.rand(1, 16, 16, 3), **kwargs)
