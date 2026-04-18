from __future__ import annotations

import logging

import pytest
import torch
from kornia.color import rgb_to_lab

from core import luminosity_masks
from core.luminosity import _luminance


def _gradient_image(height: int, width: int) -> torch.Tensor:
    ramp = torch.linspace(0.0, 1.0, width, dtype=torch.float32)
    row = ramp.view(1, 1, 1, width).repeat(1, 3, height, 1)
    return row.clone()


def test_shape_output_3_bhw(rng: torch.Generator) -> None:
    image = torch.rand((2, 3, 48, 64), generator=rng, dtype=torch.float32)

    shadows, midtones, highlights = luminosity_masks(image)

    assert shadows.shape == (2, 48, 64)
    assert midtones.shape == (2, 48, 64)
    assert highlights.shape == (2, 48, 64)


def test_partition_of_unity(rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 64, 64), generator=rng, dtype=torch.float32)

    shadows, midtones, highlights = luminosity_masks(image, soft_edge=0.3)
    total = shadows + midtones + highlights

    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


@pytest.mark.parametrize("source", ["lab_l", "ycbcr_y", "hsv_v"])
def test_luminance_sources_all_valid(source: str, rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 64, 64), generator=rng, dtype=torch.float32)

    shadows, midtones, highlights = luminosity_masks(image, luminance_source=source)

    assert torch.all((shadows >= 0.0) & (shadows <= 1.0))
    assert torch.all((midtones >= 0.0) & (midtones <= 1.0))
    assert torch.all((highlights >= 0.0) & (highlights <= 1.0))


def test_synthetic_gradient_monotonic() -> None:
    image = _gradient_image(32, 256)

    shadows, midtones, highlights = luminosity_masks(image, soft_edge=0.1)
    shadows_line = shadows[0].mean(dim=0)
    highlights_line = highlights[0].mean(dim=0)
    midtones_line = midtones[0].mean(dim=0)
    peak_index = int(midtones_line.argmax().item())

    assert torch.all(torch.diff(shadows_line) <= 1e-6)
    assert torch.all(torch.diff(highlights_line) >= -1e-6)
    assert 80 <= peak_index <= 176


def test_soft_edge_sharp_vs_smooth() -> None:
    image = _gradient_image(8, 512)

    shadows_sharp, _, _ = luminosity_masks(image, soft_edge=0.01)
    shadows_smooth, _, _ = luminosity_masks(image, soft_edge=0.3)

    sharp_transition = ((shadows_sharp > 0.05) & (shadows_sharp < 0.95)).sum().item()
    smooth_transition = ((shadows_smooth > 0.05) & (shadows_smooth < 0.95)).sum().item()

    assert smooth_transition > sharp_transition


def test_shadow_end_lt_highlight_start(rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 32, 32), generator=rng, dtype=torch.float32)

    with pytest.raises(ValueError, match="shadow_end"):
        luminosity_masks(image, shadow_end=0.5, highlight_start=0.5)


def test_raise_on_invalid_source(rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 32, 32), generator=rng, dtype=torch.float32)

    with pytest.raises(ValueError, match="luminance_source"):
        luminosity_masks(image, luminance_source="bad")


def test_hsv_v_shape_valid(rng: torch.Generator) -> None:
    image = torch.rand((3, 3, 40, 52), generator=rng, dtype=torch.float32)

    shadows, midtones, highlights = luminosity_masks(image, luminance_source="hsv_v")

    assert shadows.shape == (3, 40, 52)
    assert midtones.shape == (3, 40, 52)
    assert highlights.shape == (3, 40, 52)


def test_out_of_range_clamp_warns(caplog: pytest.LogCaptureFixture, rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 32, 32), generator=rng, dtype=torch.float32)
    image[0, 0, 0, 0] = -0.2
    image[0, 1, 0, 1] = 1.3

    with caplog.at_level(logging.WARNING):
        masks = luminosity_masks(image)

    assert "clamped to [0,1]" in caplog.text
    assert all(torch.all(mask >= 0.0) for mask in masks)


def test_lab_l_matches_kornia_reference(rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 48, 48), generator=rng, dtype=torch.float32)

    fast = _luminance(image, "lab_l")
    reference = (rgb_to_lab(image)[:, :1] / 100.0).clamp(0.0, 1.0)

    assert torch.allclose(fast, reference, atol=1e-4)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_device_cpu_cuda_parity(device_name: str, rng: torch.Generator) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this runner.")

    base = torch.rand((1, 3, 96, 96), generator=rng, dtype=torch.float32)
    reference = luminosity_masks(base, luminance_source="lab_l", soft_edge=0.1)

    image = base.to(device=device_name)
    output = luminosity_masks(image, luminance_source="lab_l", soft_edge=0.1)

    for expected, actual in zip(reference, output, strict=True):
        assert actual.device.type == device_name
        assert torch.allclose(actual.cpu(), expected, atol=1e-5)
