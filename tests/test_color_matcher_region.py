from __future__ import annotations

import pytest
import torch
from kornia.color import rgb_to_lab

from core import color_matcher_region


def _solid_bchw(
    red: float,
    green: float,
    blue: float,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    image = torch.empty((1, 3, height, width), dtype=torch.float32)
    image[:, 0] = red
    image[:, 1] = green
    image[:, 2] = blue
    return image


def test_accepts_reference_with_different_hw() -> None:
    target = _solid_bchw(0.8, 0.2, 0.2, height=32, width=48)
    reference = _solid_bchw(0.2, 0.8, 0.2, height=19, width=27)

    output = color_matcher_region(target, reference, channels="lab")

    assert output.shape == target.shape
    assert output[:, 1].mean().item() > output[:, 0].mean().item()


def test_target_mask_limits_apply_region() -> None:
    target = _solid_bchw(0.8, 0.2, 0.2, height=32, width=32)
    reference = _solid_bchw(0.2, 0.8, 0.2, height=20, width=20)
    target_mask = torch.zeros((1, 32, 32), dtype=torch.float32)
    target_mask[:, :, :16] = 1.0

    output = color_matcher_region(
        target,
        reference,
        channels="lab",
        target_mask=target_mask,
    )

    assert output[:, 1, :, :16].mean().item() > output[:, 0, :, :16].mean().item()
    assert torch.equal(output[:, :, :, 16:], target[:, :, :, 16:])


def test_reference_mask_controls_reference_stats() -> None:
    target = _solid_bchw(0.8, 0.2, 0.2, height=32, width=32)
    reference = torch.zeros((1, 3, 24, 24), dtype=torch.float32)
    reference[:, 1, :, :12] = 0.9
    reference[:, 2, :, 12:] = 0.9
    green_reference_mask = torch.zeros((1, 24, 24), dtype=torch.float32)
    green_reference_mask[:, :, :12] = 1.0

    masked = color_matcher_region(
        target,
        reference,
        channels="lab",
        reference_mask=green_reference_mask,
    )
    unmasked = color_matcher_region(target, reference, channels="lab")

    assert masked[:, 1].mean().item() > masked[:, 2].mean().item()
    assert not torch.allclose(masked, unmasked, atol=1e-4)


def test_no_masks_matches_full_image_stats() -> None:
    xx = torch.linspace(0.0, 1.0, 40, dtype=torch.float32).view(1, 1, 1, 40)
    yy = torch.linspace(0.0, 1.0, 32, dtype=torch.float32).view(1, 1, 32, 1)
    target = torch.cat(
        [
            (0.2 + 0.5 * xx).expand(1, 1, 32, 40),
            (0.3 + 0.3 * yy).expand(1, 1, 32, 40),
            (0.1 + 0.2 * ((xx + yy) / 2.0)).expand(1, 1, 32, 40),
        ],
        dim=1,
    )
    reference = torch.rand((1, 3, 18, 22), dtype=torch.float32)

    output = color_matcher_region(target, reference, channels="lab")
    output_lab = rgb_to_lab(output)
    reference_lab = rgb_to_lab(reference)

    output_mean = output_lab.mean(dim=(-1, -2))
    reference_mean = reference_lab.mean(dim=(-1, -2))
    assert torch.allclose(output_mean, reference_mean, rtol=0.10, atol=2.0)


def test_empty_target_mask_raises() -> None:
    target = _solid_bchw(0.8, 0.2, 0.2, height=16, width=16)
    reference = _solid_bchw(0.2, 0.8, 0.2, height=8, width=8)
    target_mask = torch.zeros((1, 16, 16), dtype=torch.float32)

    with pytest.raises(ValueError, match="at least one positive pixel"):
        color_matcher_region(target, reference, target_mask=target_mask)
