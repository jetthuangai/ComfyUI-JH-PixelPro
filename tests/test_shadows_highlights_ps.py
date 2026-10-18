"""Photoshop 25.4 reference values below were measured by PhotoCraft (16-bit levels of 32768)."""

from __future__ import annotations

import pytest
import torch

from core.shadows_highlights_ps import (
    _lift,
    _strength,
    _tonal_mask_table,
    shadows_highlights_photoshop,
)


def _probe(bg: list[float], px: list[float], **kwargs) -> torch.Tensor:
    """Flat 121x121 image with one different pixel in the middle; returns that pixel."""

    img = torch.tensor(bg, dtype=torch.float32).view(1, 1, 1, 3).expand(1, 121, 121, 3).clone()
    img[0, 60, 60] = torch.tensor(px)
    return shadows_highlights_photoshop(img, **kwargs)[0, 60, 60]


def _levels(v: list[int]) -> torch.Tensor:
    return torch.tensor(v, dtype=torch.float32) / 32768.0


def _off(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).abs() * 255.0).max().item()


@pytest.mark.parametrize(
    ("bg", "want"), [(0.0, [11389, 19422, 28451, 31931]), (0.5, [5211, 9909, 18229, 25700])]
)
def test_grey_matches_photoshop_default_shadows(bg: float, want: list[int]) -> None:
    # Default Shadows (35 %, Tone 50 %, Radius 30 px), Color 0, no clipping.
    for value, level in zip([0.125, 0.25, 0.5, 0.749969], want, strict=True):
        got = _probe([bg] * 3, [value] * 3, color=0.0, black_clip=0.0, white_clip=0.0)
        assert _off(got, _levels([level] * 3)) < 2.0, (bg, value, got * 32768)


@pytest.mark.parametrize(
    ("bg", "px", "want"),
    [
        (0.0, [3277, 1638, 0], [8235, 5202, 2170]),
        (0.0, [8192, 6554, 4915], [19350, 16514, 13676]),
        (4915.0 / 32768.0, [4915, 3277, 1638], [10124, 7424, 4723]),
        (9830.0 / 32768.0, [11469, 9830, 8192], [16540, 14376, 12214]),
    ],
)
def test_colour_matches_photoshop(bg: float, px: list[int], want: list[int]) -> None:
    # Amount 35 %, Radius 10 px, Color +20.
    got = _probe([bg] * 3, _levels(px).tolist(), shadow_radius=10.0, black_clip=0.0, white_clip=0.0)
    assert _off(got, _levels(want)) < 2.0, got * 32768


def test_edge_matches_photoshop() -> None:
    # Amount 100 %, Tone 100 %, Radius 10 px across a 0.1 | 0.9 edge.
    a, b = 3277.0 / 32768.0, 29491.0 / 32768.0
    img = torch.where(torch.arange(200).view(1, 1, 200, 1) < 100, a, b).expand(1, 24, 200, 3)
    out = shadows_highlights_photoshop(
        img.contiguous(),
        shadow_amount=100.0,
        shadow_tone=100.0,
        shadow_radius=10.0,
        color=0.0,
        black_clip=0.0,
        white_clip=0.0,
    )
    row = out[0, 12, :, 0] * 32768.0
    for x, want in [
        (92, 16370),
        (98, 14439),
        (99, 12146),
        (100, 30891),
        (101, 30403),
        (102, 30146),
        (108, 29979),
    ]:
        assert abs(row[x].item() - want) / 32768.0 * 255.0 < 6.0, (x, row[x].item(), want)


def test_strength_and_curve() -> None:
    assert abs(_strength(50.0) - 2.0) < 2e-3 and abs(_strength(100.0) - 3.0) < 1e-4
    assert abs(_strength(15.0) - 1.02) < 0.01 and abs(_strength(1.0) - 0.19) < 0.01
    v = torch.tensor([0.1, 0.3, 0.6])
    assert torch.allclose(_lift(2.0, v), 1.0 - (1.0 - v) ** 4, atol=1e-6)


def test_tonal_mask_tables() -> None:
    m95 = _tonal_mask_table(95.0)
    assert abs(m95[128] - (1.0 - 128.0 / 255.0)) < 1e-3
    assert _tonal_mask_table(100.0) == m95
    m0 = _tonal_mask_table(0.0)
    assert m0[0] > 0.999 and abs(m0[18] - 0.5) < 0.03 and m0[60] < 0.02


def test_highlights_darken_bright_area() -> None:
    img = torch.full((1, 64, 64, 3), 0.3)
    img[:, 16:48, 16:48] = 0.9
    out = shadows_highlights_photoshop(
        img, shadow_amount=0.0, highlight_amount=60.0, black_clip=0.0, white_clip=0.0
    )
    assert out[0, 32, 32, 0].item() < 0.9 - 0.03


def test_zero_amounts_keep_image_without_clip() -> None:
    img = torch.rand(2, 20, 30, 3)
    out = shadows_highlights_photoshop(img, shadow_amount=0.0, black_clip=0.0, white_clip=0.0)
    assert (out - img).abs().max().item() < 1e-5


def test_validation() -> None:
    with pytest.raises(ValueError):
        shadows_highlights_photoshop(torch.rand(1, 8, 8, 3), shadow_amount=120.0)
    with pytest.raises(ValueError):
        shadows_highlights_photoshop(torch.rand(1, 8, 8))
