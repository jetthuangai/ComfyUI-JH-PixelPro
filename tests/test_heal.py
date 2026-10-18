"""Heal / Content-Aware Fill core (``core/heal.py``)."""

from __future__ import annotations

import pytest
import torch

from core.heal import heal


def _stripes(h: int = 120, w: int = 160) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    stripes = (0.5 + 0.4 * torch.sin((xx + yy) * 0.35)).float()
    return stripes[None, ..., None].expand(1, h, w, 3).contiguous()


def _hole(h: int = 120, w: int = 160) -> torch.Tensor:
    mask = torch.zeros(1, h, w)
    mask[:, 45:75, 60:100] = 1.0
    return mask


def test_content_aware_continues_texture() -> None:
    image, mask = _stripes(), _hole()
    damaged = image * (1 - mask[..., None])  # what is under the mask must not matter
    out = heal(damaged, mask, grow=0, feather=0)
    region = mask[0] > 0
    assert (out[0][region] - image[0][region]).abs().mean() < 0.03
    smooth = heal(damaged, mask, mode="smooth", grow=0, feather=0)
    assert (smooth[0][region] - image[0][region]).abs().mean() > 0.15  # flat fill loses it


def test_outside_mask_unchanged_and_empty_mask_identity() -> None:
    image, mask = _stripes(), _hole()
    out = heal(image, mask, grow=0, feather=0)
    assert torch.equal(out[0][mask[0] == 0], image[0][mask[0] == 0])
    assert torch.equal(heal(image, torch.zeros_like(mask)), image)


def test_flat_and_gradient_fill() -> None:
    xx = torch.linspace(0.2, 0.8, 160).view(1, 1, 160, 1).expand(1, 120, 160, 3).contiguous()
    mask = _hole()
    for mode in ("content_aware", "smooth"):
        out = heal(xx, mask, mode=mode, grow=0, feather=0)
        assert (out - xx).abs().max() < 0.05


def test_deterministic_and_batch() -> None:
    image = torch.cat([_stripes(), _stripes().flip(2)])
    mask = _hole()
    a, b = heal(image, mask), heal(image, mask)
    assert torch.equal(a, b)
    single = heal(image[1:], mask)
    assert (a[1] - single[0]).abs().max() < 0.05


def test_whole_frame_mask_is_left_alone() -> None:
    image = _stripes(40, 40)
    assert torch.equal(heal(image, torch.ones(1, 40, 40)), image)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mode": "clone"}, "mode"),
        ({"grow": -1}, "grow"),
        ({"feather": 40.0}, "feather"),
        ({"context": 0.0}, "context"),
    ],
)
def test_validates(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        heal(torch.rand(1, 16, 16, 3), torch.zeros(1, 16, 16), **kwargs)


def test_scattered_and_dense_masks_do_not_crash() -> None:
    image = _stripes(128, 128)
    dots = torch.zeros(1, 128, 128)
    dots[:, ::12, ::12] = 1.0
    out = heal(image, dots, grow=1, feather=0)
    region = dots[0] > 0
    assert (out[0][region] - image[0][region]).abs().mean() < 0.1
    checker = ((torch.arange(64)[:, None] + torch.arange(64)[None, :]) % 2).float()[None]
    out = heal(_stripes(64, 64), checker, grow=0, feather=0)
    assert torch.isfinite(out).all()


def test_feather_blends_a_band_outside_the_mask() -> None:
    image, mask = _stripes(), _hole()
    damaged = image.clone()
    damaged[0][mask[0] > 0] = torch.tensor([1.0, 0.0, 0.0])
    hard = heal(damaged, mask, grow=0, feather=0)
    soft = heal(damaged, mask, grow=0, feather=6)
    band = torch.zeros_like(mask[0], dtype=torch.bool)
    band[41:79, 56:104] = True
    band &= mask[0] == 0
    assert torch.equal(hard[0][band], damaged[0][band])
    assert not torch.equal(soft[0][band], damaged[0][band])
    assert torch.equal(soft[0][:, :40], damaged[0][:, :40])  # far away untouched


def test_result_does_not_depend_on_batch_position() -> None:
    image, mask = _stripes(), _hole()
    alone = heal(image, mask)
    second = heal(torch.cat([image.flip(1), image]), mask)
    assert torch.equal(alone[0], second[1])
