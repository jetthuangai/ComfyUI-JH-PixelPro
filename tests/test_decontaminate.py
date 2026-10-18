"""Decontaminate Colors core (``core/decontaminate.py``)."""

from __future__ import annotations

import pytest
import torch

from core.decontaminate import decontaminate_colors


def _red_on_green(size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Red disc on green with a soft 6 px edge; edge pixels mix red and green by alpha."""
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    dist = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2).float().sqrt()
    alpha = ((size / 4 + 3 - dist) / 6).clamp(0, 1)
    red, green = torch.tensor([0.8, 0.1, 0.1]), torch.tensor([0.1, 0.8, 0.1])
    image = alpha[..., None] * red + (1 - alpha[..., None]) * green
    return image.unsqueeze(0), alpha.unsqueeze(0)


def test_edge_loses_background_spill() -> None:
    image, alpha = _red_on_green()
    out = decontaminate_colors(image, alpha, radius=16)
    edge = (alpha > 0.1) & (alpha < 0.9)
    green_before = image[..., 1][edge].mean()
    green_after = out[..., 1][edge].mean()
    assert green_after < 0.5 * green_before
    # Edge colour now close to the subject's red.
    assert (out[edge] - torch.tensor([0.8, 0.1, 0.1])).abs().max() < 0.15


def test_solid_and_background_untouched() -> None:
    image, alpha = _red_on_green()
    out = decontaminate_colors(image, alpha)
    solid = (alpha == 0) | (alpha == 1)
    assert torch.equal(out[solid], image[solid])


def test_strength_zero_and_flat_image() -> None:
    image, alpha = _red_on_green()
    assert torch.equal(decontaminate_colors(image, alpha, strength=0.0), image)
    flat = torch.full_like(image, 0.4)
    assert torch.allclose(decontaminate_colors(flat, alpha), flat, atol=1e-6)


def test_hard_mask_is_identity() -> None:
    image, alpha = _red_on_green()
    hard = (alpha > 0.5).float()
    assert torch.equal(decontaminate_colors(image, hard), image)


def test_no_solid_foreground_keeps_colour() -> None:
    image, _ = _red_on_green()
    weak = torch.full(image.shape[:3], 0.5)
    assert torch.equal(decontaminate_colors(image, weak), image)


def test_far_edge_uses_wider_pass() -> None:
    # Subject 2 px wide strip at the left; soft ramp 40 px wide -> radius 4 must still fill.
    alpha = torch.zeros(1, 32, 96)
    alpha[..., :2] = 1.0
    alpha[..., 2:42] = torch.linspace(0.97, 0.03, 40)
    image = alpha[..., None] * torch.tensor([0.9, 0.2, 0.2]) + (1 - alpha[..., None]) * 0.5
    out = decontaminate_colors(image, alpha, radius=4)
    assert (out[0, :, 30, 0] - 0.9).abs().max() < 0.05


@pytest.mark.parametrize(
    ("kwargs", "match"), [({"radius": 0}, "radius"), ({"strength": 1.5}, "strength")]
)
def test_validates(kwargs: dict, match: str) -> None:
    image, alpha = _red_on_green(16)
    with pytest.raises(ValueError, match=match):
        decontaminate_colors(image, alpha, **kwargs)


def test_rejects_mismatched_mask() -> None:
    image, _ = _red_on_green(16)
    with pytest.raises(ValueError, match="spatial"):
        decontaminate_colors(image, torch.ones(1, 8, 8))


def test_edge_detail_is_kept() -> None:
    # Textured subject (hair-like stripes) over flat green: the exact solve recovers the
    # stripes instead of a blurred average.
    gen = torch.Generator().manual_seed(4)
    fg = torch.rand(1, 48, 96, 3, generator=gen)
    bg = torch.tensor([0.1, 0.8, 0.1])
    alpha = torch.zeros(1, 48, 96)
    alpha[..., :30] = 1.0
    alpha[..., 30:66] = torch.linspace(0.98, 0.02, 36)
    image = alpha[..., None] * fg + (1 - alpha[..., None]) * bg
    out = decontaminate_colors(image, alpha, radius=16)
    strong = (alpha > 0.6) & (alpha < 1.0)
    assert (out[strong] - fg[strong]).abs().max() < 1e-3


def test_faint_mask_tail_does_not_repaint_background() -> None:
    image, alpha = _red_on_green()
    tail = torch.where(alpha == 0, torch.full_like(alpha, 1e-4), alpha)
    out = decontaminate_colors(image, tail)
    background = alpha == 0
    assert (out[background] - image[background]).abs().max() < 0.01


def test_no_background_keeps_detail_where_mostly_subject() -> None:
    gen = torch.Generator().manual_seed(9)
    image = torch.rand(1, 32, 32, 3, generator=gen)
    alpha = torch.full((1, 32, 32), 0.9)
    alpha[:, :, :8] = 1.0  # solid subject, but nothing below alpha 0.05 anywhere
    out = decontaminate_colors(image, alpha)
    assert torch.allclose(out, image, atol=1e-6)
