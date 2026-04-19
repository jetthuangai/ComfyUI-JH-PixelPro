from __future__ import annotations

import logging

import pytest
import torch
from kornia.color import rgb_to_lab

from core import color_matcher


def _gradient_image(size: int) -> torch.Tensor:
    xx = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, 1, 1, size)
    yy = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, 1, size, 1)
    return torch.cat(
        [
            (0.2 + (0.4 * xx)).expand(1, 1, size, size),
            (0.25 + (0.3 * yy)).expand(1, 1, size, size),
            (0.3 + (0.2 * ((xx + yy) / 2.0))).expand(1, 1, size, size),
        ],
        dim=1,
    )


def _two_patch_target(size: int = 64) -> torch.Tensor:
    image = torch.zeros((1, 3, size, size), dtype=torch.float32)
    half = size // 2
    image[:, 0, :, :half] = 0.9
    image[:, 2, :, half:] = 0.8
    return image


def _two_patch_reference(size: int = 64) -> torch.Tensor:
    image = torch.zeros((1, 3, size, size), dtype=torch.float32)
    half = size // 2
    image[:, 1, :, :half] = 0.85
    image[:, 0, :, half:] = 0.9
    image[:, 1, :, half:] = 0.85
    return image


def _stats(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = values.mean(dim=(-1, -2))
    std = values.std(dim=(-1, -2), correction=0)
    return mean, std


def test_shape_bchw() -> None:
    target = _gradient_image(64)
    reference = torch.flip(target, dims=[-1])

    output = color_matcher(target, reference)

    assert output.shape == target.shape


def test_strength_zero_identity() -> None:
    target = _gradient_image(64)
    reference = torch.flip(target, dims=[-2])

    output = color_matcher(target, reference, strength=0.0)

    assert torch.equal(output, target)


def test_strength_one_same_img() -> None:
    target = _gradient_image(64)

    output = color_matcher(target, target, strength=1.0)

    assert torch.allclose(output, target, atol=1e-3)


def test_channels_ab_preserves_l() -> None:
    target = _gradient_image(64)
    reference = torch.flip(target, dims=[-1]).roll(shifts=8, dims=-2)

    output = color_matcher(target, reference, channels="ab")
    output_lab = rgb_to_lab(output)
    target_lab = rgb_to_lab(target)

    assert torch.allclose(output_lab[:, :1], target_lab[:, :1], atol=1.0)


def test_channels_lab_full_match() -> None:
    target = _gradient_image(96)
    reference = _gradient_image(96).roll(shifts=11, dims=-1).flip(-2)

    output = color_matcher(target, reference, channels="lab")
    output_lab = rgb_to_lab(output)
    reference_lab = rgb_to_lab(reference)

    output_mean, output_std = _stats(output_lab)
    reference_mean, reference_std = _stats(reference_lab)

    assert torch.allclose(output_mean, reference_mean, rtol=0.05, atol=1.0)
    assert torch.allclose(output_std, reference_std, rtol=0.10, atol=1.0)


def test_strength_half_matches_linear_blend() -> None:
    target = _gradient_image(64)
    reference = torch.flip(target, dims=[-2])
    full = color_matcher(target, reference, strength=1.0)
    blended = color_matcher(target, reference, strength=0.5)

    expected = (0.5 * target) + (0.5 * full)
    assert torch.allclose(blended, expected, atol=1e-5)


def test_mask_stat_gate() -> None:
    target = _two_patch_target()
    reference = _two_patch_reference()
    mask = torch.zeros((1, 64, 64), dtype=torch.float32)
    mask[:, :, :32] = 1.0

    masked = color_matcher(target, reference, mask=mask)
    unmasked = color_matcher(target, reference)

    left_half = masked[:, :, :, :32]
    right_half = masked[:, :, :, 32:]

    assert left_half[:, 1].mean().item() > left_half[:, 0].mean().item()
    assert torch.mean(torch.abs(right_half - target[:, :, :, 32:])).item() > 0.01
    assert not torch.allclose(masked, unmasked, atol=1e-4)


def test_mask_none_full_img() -> None:
    target = _gradient_image(64)
    reference = torch.flip(target, dims=[-1])
    mask = torch.ones((1, 64, 64), dtype=torch.float32)

    masked = color_matcher(target, reference, mask=mask)
    unmasked = color_matcher(target, reference, mask=None)

    assert torch.allclose(masked, unmasked, atol=1e-5)


def test_shape_mismatch_raises() -> None:
    target = _gradient_image(64)
    reference = _gradient_image(32)

    with pytest.raises(ValueError, match="same HxW"):
        color_matcher(target, reference)


def test_invalid_channels_raises() -> None:
    with pytest.raises(ValueError, match="channels must be one of"):
        color_matcher(_gradient_image(32), _gradient_image(32), channels="xyz")


@pytest.mark.parametrize("strength", [-0.1, 1.5])
def test_invalid_strength_raises(strength: float) -> None:
    with pytest.raises(ValueError, match="strength must be in"):
        color_matcher(_gradient_image(32), _gradient_image(32), strength=strength)


def test_nan_clean_constant_color_region() -> None:
    target = torch.full((1, 3, 64, 64), 0.4, dtype=torch.float32)
    reference = torch.full((1, 3, 64, 64), 0.7, dtype=torch.float32)

    output = color_matcher(target, reference, channels="lab")

    assert torch.isfinite(output).all()
    assert output.min().item() >= 0.0
    assert output.max().item() <= 1.0


def test_warn_clamp_out_of_range(caplog: pytest.LogCaptureFixture) -> None:
    target = _gradient_image(32)
    reference = _gradient_image(32)
    target[0, 0, 0, 0] = -0.2
    reference[0, 1, 0, 1] = 1.2

    with caplog.at_level(logging.WARNING):
        output = color_matcher(target, reference)

    assert "outside [0,1]" in caplog.text
    assert output.min().item() >= 0.0
    assert output.max().item() <= 1.0


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_device_cpu_cuda_parity(device_name: str) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this runner.")

    target = _gradient_image(96)
    reference = torch.flip(target, dims=[-1]).roll(shifts=13, dims=-2)
    mask = torch.ones((1, 96, 96), dtype=torch.float32)
    mask[:, :, 48:] = 0.5

    expected = color_matcher(target, reference, channels="lab", strength=0.8, mask=mask)
    actual = color_matcher(
        target.to(device=device_name),
        reference.to(device=device_name),
        channels="lab",
        strength=0.8,
        mask=mask.to(device=device_name),
    )

    assert actual.device.type == device_name
    assert torch.allclose(actual.cpu(), expected, atol=1e-4)
