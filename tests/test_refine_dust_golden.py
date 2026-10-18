"""Edge-Aware Refiner and Dust & Spot: defaults reproduce v1.7.0 exactly; new modes work."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from core.mask_edge_refine import edge_aware_refine

sys.path.insert(0, str(Path(__file__).with_name("fixtures")))
import make_refine_dust_golden as golden_maker  # noqa: E402

GOLDEN = torch.load(golden_maker.GOLDEN)
OUTPUTS = golden_maker.outputs()


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_matches_v170(key: str) -> None:
    # Spot masks are binary and must match exactly; images may differ by float rounding
    # across platforms (the golden is made on Windows, CI runs Linux).
    if key.endswith("_spots"):
        assert torch.equal(OUTPUTS[key], GOLDEN[key])
    else:
        diff = (OUTPUTS[key] - GOLDEN[key]).abs().max()
        assert torch.allclose(OUTPUTS[key], GOLDEN[key], atol=1e-5), diff


def test_rgb_guide_follows_isoluminant_edge() -> None:
    # Red | green halves with equal luma: the grey guide sees no edge, the RGB guide does.
    red, green = torch.tensor([0.6, 0.2, 0.4]), torch.tensor([0.2, 0.4013, 0.2])
    guide = torch.zeros(1, 32, 64, 3)
    guide[:, :, :32] = red
    guide[:, :, 32:] = green
    mask = torch.zeros(1, 32, 64)
    mask[:, :, :32] = 1.0
    blurry = edge_aware_refine(mask, guide, radius=6, guide_mode="luma")
    sharp = edge_aware_refine(mask, guide, radius=6, guide_mode="rgb")
    err = lambda m: (m - mask).abs().mean().item()  # noqa: E731
    assert err(sharp) < 0.5 * err(blurry)


def test_rgb_guide_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="guide_mode"):
        edge_aware_refine(torch.ones(1, 8, 8), torch.ones(1, 8, 8, 3), guide_mode="hsv")


def _textured_with_spots() -> tuple[torch.Tensor, torch.Tensor]:
    """Grainy mid-grey with a gentle gradient and dark dust specks; also the clean copy."""
    gen = torch.Generator().manual_seed(11)
    h, w = 160, 200
    ramp = torch.linspace(0.45, 0.6, w).view(1, 1, w, 1)
    clean = (ramp + 0.05 * torch.randn(1, h, w, 1, generator=gen)).clamp(0, 1).expand(1, h, w, 3)
    clean = clean.contiguous()
    dirty = clean.clone()
    for y, x in ((40, 50), (90, 120), (120, 60), (60, 160)):
        dirty[:, y - 3 : y + 4, x - 3 : x + 4] *= 0.35
    return dirty, clean


def test_texture_fill_keeps_grain() -> None:
    from core.dust_spot import dust_spot_removal

    dirty, clean = _textured_with_spots()
    blur, spots = dust_spot_removal(dirty, spot_size=12, sensitivity=0.6, fill_mode="blur")
    texture, spots_t = dust_spot_removal(dirty, spot_size=12, sensitivity=0.6, fill_mode="texture")
    assert torch.equal(spots, spots_t)
    region = spots[0] > 0.5
    assert int(region.sum()) > 100
    ref_std = clean[0, ..., 0][region].std()
    blur_std = blur[0, ..., 0][region].std()
    texture_std = texture[0, ..., 0][region].std()
    assert blur_std < 0.6 * ref_std  # blur fill is flat
    assert 0.7 * ref_std < texture_std < 1.4 * ref_std  # texture fill keeps the grain
    # Brightness follows the surroundings (no dark residue, no bright patch).
    assert abs(texture[0, ..., 0][region].mean() - clean[0, ..., 0][region].mean()) < 0.02
    # Outside the spots nothing changes.
    assert torch.equal(texture[0][~region], dirty[0][~region])


def test_texture_fill_clean_image_unchanged() -> None:
    from core.dust_spot import dust_spot_removal

    _, clean = _textured_with_spots()
    out, spots = dust_spot_removal(clean, fill_mode="texture")
    assert not spots.any()
    assert torch.equal(out, clean)


def test_fill_mode_validated() -> None:
    from core.dust_spot import dust_spot_removal

    with pytest.raises(ValueError, match="fill_mode"):
        dust_spot_removal(torch.rand(1, 16, 16, 3), fill_mode="clone")
