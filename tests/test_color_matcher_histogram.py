"""Histogram (quantile mapping) mode of Color Matcher / Color Matcher Region."""

from __future__ import annotations

import time

import pytest
import torch
from kornia.color import rgb_to_lab

from core import color_matcher, color_matcher_region


def _random_image(seed: int, height: int = 64, width: int = 64, gain: float = 1.0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    base = torch.rand((1, 3, height, width), generator=gen) ** 1.5
    return (base * gain).clamp(0.0, 1.0)


def _quantiles(lab: torch.Tensor, channel: int) -> torch.Tensor:
    q = torch.linspace(0.05, 0.95, 10)
    return torch.quantile(lab[:, channel].flatten(), q)


def test_histogram_matches_reference_quantiles_lab() -> None:
    target = _random_image(1, gain=0.6)
    reference = _random_image(2)
    out = color_matcher(target, reference, channels="lab", method="histogram")
    out_lab, ref_lab = rgb_to_lab(out), rgb_to_lab(reference)
    for ch in range(3):
        diff = (_quantiles(out_lab, ch) - _quantiles(ref_lab, ch)).abs().max().item()
        span = (ref_lab[:, ch].max() - ref_lab[:, ch].min()).item()
        assert diff < 0.03 * span, (ch, diff)


def test_histogram_ab_preserves_luminance() -> None:
    target = _random_image(3)
    reference = _random_image(4, gain=0.7)
    out = color_matcher(target, reference, channels="ab", method="histogram")
    # Gamut clipping of saturated random colours moves L slightly; mean error stays small.
    assert (rgb_to_lab(out)[:, 0] - rgb_to_lab(target)[:, 0]).abs().mean().item() < 2.0


def test_stats_is_default_and_unchanged() -> None:
    target = _random_image(5)
    reference = _random_image(6)
    default = color_matcher(target, reference)
    explicit = color_matcher(target, reference, method="stats")
    assert torch.equal(default, explicit)
    assert not torch.allclose(default, color_matcher(target, reference, method="histogram"))


def test_strength_zero_is_identity() -> None:
    target = _random_image(7)
    reference = _random_image(8)
    out = color_matcher(target, reference, strength=0.0, method="histogram")
    assert torch.equal(out, target)


def test_identical_images_are_nearly_unchanged() -> None:
    image = _random_image(9)
    out = color_matcher(image, image.clone(), channels="lab", method="histogram")
    assert (out - image).abs().max().item() < 0.03


def test_invalid_method_raises() -> None:
    image = _random_image(10)
    with pytest.raises(ValueError, match="method"):
        color_matcher(image, image, method="nope")
    with pytest.raises(ValueError, match="method"):
        color_matcher_region(image, image, method="nope")


def test_mask_changes_histogram_reference() -> None:
    target = _random_image(11)
    reference = torch.cat(
        [torch.full((1, 3, 64, 32), 0.2), torch.full((1, 3, 64, 32), 0.8)], dim=-1
    )
    reference[:, 0, :, :32] = 0.8
    reference[:, 1, :, :32] = 0.2
    mask_left = torch.zeros((1, 64, 64))
    mask_left[:, :, :32] = 1.0
    mask_right = 1.0 - mask_left
    left = color_matcher(target, reference, method="histogram", mask=mask_left)
    right = color_matcher(target, reference, method="histogram", mask=mask_right)
    assert not torch.allclose(left, right, atol=1e-3)


def test_region_different_sizes_and_masks() -> None:
    target = _random_image(12, 48, 64)
    reference = _random_image(13, 30, 20, gain=0.7)
    target_mask = torch.zeros((1, 48, 64))
    target_mask[:, :, :32] = 1.0
    reference_mask = torch.zeros((1, 30, 20))
    reference_mask[:, :15] = 1.0
    out = color_matcher_region(
        target,
        reference,
        channels="lab",
        method="histogram",
        target_mask=target_mask,
        reference_mask=reference_mask,
    )
    assert out.shape == target.shape
    assert torch.equal(out[..., 32:], target[..., 32:])
    assert not torch.allclose(out[..., :32], target[..., :32], atol=1e-3)


def test_region_histogram_matches_masked_reference_quantiles() -> None:
    target = _random_image(14, 64, 64, gain=0.5)
    reference = _random_image(15, 40, 50)
    reference_mask = torch.zeros((1, 40, 50))
    reference_mask[:, :, 25:] = 1.0
    out = color_matcher_region(
        target, reference, channels="lab", method="histogram", reference_mask=reference_mask
    )
    ref_lab = rgb_to_lab(reference)[:, :, :, 25:]
    out_lab = rgb_to_lab(out)
    for ch in range(3):
        diff = (_quantiles(out_lab, ch) - _quantiles(ref_lab, ch)).abs().max().item()
        span = (ref_lab[:, ch].max() - ref_lab[:, ch].min()).item()
        assert diff < 0.04 * span, (ch, diff)


def test_batch_reference_broadcast() -> None:
    target = torch.cat([_random_image(16), _random_image(17)], dim=0)
    reference = _random_image(18)
    out = color_matcher_region(target, reference, method="histogram")
    assert out.shape == target.shape
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_histogram_2k_cuda_memory_and_speed() -> None:
    target = torch.rand((1, 3, 2048, 2048), device="cuda")
    reference = torch.rand((1, 3, 2048, 2048), device="cuda") * 0.7
    color_matcher(target, reference, method="histogram")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    color_matcher(target, reference, method="histogram")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 1024**3
    assert peak < 2.0, peak
    assert elapsed < 2.0, elapsed


def test_histogram_target_minimum_maps_to_reference_minimum() -> None:
    gen = torch.Generator().manual_seed(5)
    target = 0.05 + torch.rand(1, 3, 64, 64, generator=gen) * 0.5
    target[:, :, :2, :2] = 0.0  # a few clipped blacks at the global minimum
    reference = 0.2 + torch.rand(1, 3, 64, 64, generator=gen) * 0.6
    out = color_matcher(target, reference, channels="lab", method="histogram")
    lab_ref = rgb_to_lab(reference)[:, 0]
    lab_out = rgb_to_lab(out.clamp(0, 1))[:, 0]
    clipped = lab_out[:, :2, :2]
    # blacks land near the reference minimum luminance, not on the shared range start (0)
    assert float(clipped.min()) > float(lab_ref.min()) - 2.0
    assert float(clipped.min()) > 5.0
    # continuous: the darkest non-clipped output is close to the clipped one
    rest = lab_out.clone()
    rest[:, :2, :2] = 1e9
    assert float(rest.min()) - float(clipped.max()) < 3.0
