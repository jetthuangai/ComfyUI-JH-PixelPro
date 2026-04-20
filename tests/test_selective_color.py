from __future__ import annotations

import torch

from core.selective_color import apply_hue_sat_shift, hue_range_mask, saturation_range_mask


def _solid_rgb(color: tuple[float, float, float], *, batch: int = 1, size: int = 4) -> torch.Tensor:
    return torch.tensor(color, dtype=torch.float32).view(1, 1, 1, 3).expand(batch, size, size, 3)


def _approx_rgb(tensor: torch.Tensor, expected: tuple[float, float, float], *, atol: float) -> None:
    target = torch.tensor(expected, dtype=tensor.dtype).view(1, 1, 1, 3).expand_as(tensor)
    assert torch.allclose(tensor, target, atol=atol)


def test_hue_range_mask_marks_red_center() -> None:
    image = _solid_rgb((1.0, 0.0, 0.0))

    mask = hue_range_mask(image, hue_center=0.0, band_width=30.0)

    assert mask.shape == (1, 4, 4)
    assert torch.allclose(mask, torch.ones_like(mask), atol=1e-6)


def test_hue_range_mask_rejects_opposite_hues() -> None:
    image = torch.cat([_solid_rgb((1.0, 0.0, 0.0)), _solid_rgb((0.0, 1.0, 0.0))], dim=0)

    mask = hue_range_mask(image, hue_center=0.0, band_width=30.0)

    assert torch.allclose(mask[0], torch.ones_like(mask[0]), atol=1e-6)
    assert torch.allclose(mask[1], torch.zeros_like(mask[1]), atol=1e-6)


def test_hue_range_mask_grayscale_is_zero() -> None:
    grayscale = _solid_rgb((0.5, 0.5, 0.5))

    mask = hue_range_mask(grayscale, hue_center=0.0, band_width=180.0)

    assert torch.count_nonzero(mask) == 0


def test_apply_hue_sat_shift_identity_when_mask_zero() -> None:
    image = _solid_rgb((0.2, 0.4, 0.6), batch=2)
    mask = torch.zeros((2, 4, 4), dtype=torch.float32)

    output = apply_hue_sat_shift(image, mask, hue_shift=45.0, sat_mult=1.5, sat_add=0.25)

    assert torch.allclose(output, image, atol=1e-6)


def test_apply_hue_sat_shift_round_trip_noop() -> None:
    image = torch.rand((2, 8, 8, 3), dtype=torch.float32)
    mask = torch.ones((2, 8, 8), dtype=torch.float32)

    output = apply_hue_sat_shift(image, mask, hue_shift=0.0, sat_mult=1.0, sat_add=0.0)

    assert output.shape == image.shape
    assert output.dtype == image.dtype
    assert torch.max(torch.abs(output - image)).item() < 1e-6


def test_apply_hue_sat_shift_rotates_red_to_yellow() -> None:
    image = _solid_rgb((1.0, 0.0, 0.0))
    mask = torch.ones((1, 4, 4), dtype=torch.float32)

    output = apply_hue_sat_shift(image, mask, hue_shift=60.0, sat_mult=1.0, sat_add=0.0)

    _approx_rgb(output, (1.0, 1.0, 0.0), atol=2e-4)


def test_saturation_range_mask_returns_zero_for_grayscale() -> None:
    image = _solid_rgb((0.5, 0.5, 0.5))

    mask = saturation_range_mask(image, sat_min=0.3, sat_max=1.0, feather=0.1)

    assert torch.allclose(mask, torch.zeros_like(mask), atol=1e-6)


def test_saturation_range_mask_returns_one_for_saturated_input() -> None:
    image = _solid_rgb((1.0, 0.0, 0.0))

    mask = saturation_range_mask(image, sat_min=0.5, sat_max=1.0, feather=0.0)

    assert torch.allclose(mask, torch.ones_like(mask), atol=1e-6)


def test_saturation_range_mask_soft_feather() -> None:
    image = torch.tensor(
        [[[[0.6, 0.4, 0.4], [0.8, 0.2, 0.2], [1.0, 0.0, 0.0]]]],
        dtype=torch.float32,
    )

    mask = saturation_range_mask(image, sat_min=0.3, sat_max=1.0, feather=0.2)

    assert 0.0 < float(mask[0, 0, 0]) < 1.0
    assert float(mask[0, 0, 1]) > float(mask[0, 0, 0])
    assert torch.allclose(mask[0, 0, 2], torch.tensor(1.0), atol=1e-6)
