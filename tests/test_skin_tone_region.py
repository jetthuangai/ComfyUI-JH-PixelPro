from __future__ import annotations

import pytest
import torch

from core.skin_tone_region import skin_tone_tri_region


def _luminance_gradient(height: int = 16, width: int = 96) -> torch.Tensor:
    ramp = torch.linspace(0.0, 1.0, width, dtype=torch.float32)
    image = ramp.view(1, 1, 1, width).expand(1, 3, height, width).clone()
    return image


def test_gradient_partition_sums_to_one() -> None:
    image = _luminance_gradient()
    shadow, midtone, highlight = skin_tone_tri_region(
        image,
        shadow_cutoff=0.33,
        highlight_cutoff=0.66,
        soft_sigma=1.0,
    )

    total = shadow + midtone + highlight
    assert torch.allclose(total, torch.ones_like(total), atol=1e-4)
    assert shadow[..., :24].mean().item() > 0.75
    assert midtone[..., 40:56].mean().item() > 0.75
    assert highlight[..., -24:].mean().item() > 0.75


def test_skin_mask_limits_output_and_preserves_soft_mask_sum() -> None:
    image = _luminance_gradient(height=12, width=48)
    skin_mask = torch.zeros((1, 12, 48), dtype=torch.float32)
    skin_mask[:, :, 8:40] = 0.8

    masks = skin_tone_tri_region(image, skin_mask=skin_mask, soft_sigma=1.0)
    total = sum(masks)

    assert torch.allclose(total, skin_mask, atol=1e-4)
    assert all(mask[:, :, :8].max().item() == 0.0 for mask in masks)
    assert all(mask[:, :, 40:].max().item() == 0.0 for mask in masks)


def test_none_skin_mask_treats_whole_image_as_skin() -> None:
    image = torch.full((2, 3, 10, 10), 0.5, dtype=torch.float32)
    shadow, midtone, highlight = skin_tone_tri_region(image, skin_mask=None)

    assert shadow.shape == (2, 10, 10)
    assert torch.allclose(midtone, torch.ones_like(midtone), atol=1e-4)
    assert torch.allclose(shadow, torch.zeros_like(shadow), atol=1e-4)
    assert torch.allclose(highlight, torch.zeros_like(highlight), atol=1e-4)


@pytest.mark.parametrize(
    ("shadow_cutoff", "highlight_cutoff"),
    [(0.5, 0.5), (0.6, 0.8), (0.2, 0.4), (-0.1, 0.7), (0.2, 1.1)],
)
def test_invalid_cutoffs_raise(shadow_cutoff: float, highlight_cutoff: float) -> None:
    image = _luminance_gradient()
    with pytest.raises(ValueError, match="cutoff"):
        skin_tone_tri_region(
            image,
            shadow_cutoff=shadow_cutoff,
            highlight_cutoff=highlight_cutoff,
        )


def test_batch_one_mask_broadcasts() -> None:
    image = torch.rand((2, 3, 12, 14), dtype=torch.float32)
    skin_mask = torch.ones((1, 12, 14), dtype=torch.float32) * 0.5
    masks = skin_tone_tri_region(image, skin_mask=skin_mask)

    for mask in masks:
        assert mask.shape == (2, 12, 14)
    assert torch.allclose(sum(masks), torch.full((2, 12, 14), 0.5), atol=1e-4)
