from __future__ import annotations

import logging

import pytest
import torch

from core import high_freq_detail_mask


def _make_image(batch: int, height: int, width: int, *, seed: int = 20260418) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.rand((batch, 3, height, width), generator=generator, dtype=torch.float32)


def test_shape_output_bhw(rng: torch.Generator) -> None:
    image = torch.rand((2, 3, 64, 96), generator=rng, dtype=torch.float32)

    output = high_freq_detail_mask(image, sensitivity=0.5)

    assert output.shape == (2, 64, 96)
    assert output.dtype == torch.float32


def test_sensitivity_zero_all_off(rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 64, 64), generator=rng, dtype=torch.float32)

    output = high_freq_detail_mask(image, sensitivity=0.0)

    assert torch.equal(output, torch.zeros_like(output))


def test_sensitivity_one_all_on(rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 64, 64), generator=rng, dtype=torch.float32)

    output = high_freq_detail_mask(image, sensitivity=1.0)

    assert torch.equal(output, torch.ones_like(output))


@pytest.mark.parametrize("kernel_type", ["laplacian", "sobel", "fs_gaussian"])
def test_kernel_paths_valid(kernel_type: str, rng: torch.Generator) -> None:
    image = torch.rand((2, 3, 64, 64), generator=rng, dtype=torch.float32)

    output = high_freq_detail_mask(image, kernel_type=kernel_type, sensitivity=0.5)

    assert output.shape == (2, 64, 64)
    assert torch.all((output >= 0.0) & (output <= 1.0))


def test_threshold_absolute_mode_deterministic(rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 64, 64), generator=rng, dtype=torch.float32)

    first = high_freq_detail_mask(
        image,
        sensitivity=0.35,
        kernel_type="laplacian",
        threshold_mode="absolute",
    )
    second = high_freq_detail_mask(
        image,
        sensitivity=0.35,
        kernel_type="laplacian",
        threshold_mode="absolute",
    )

    assert torch.equal(first, second)


def test_threshold_relative_percentile_mode_stable(rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 64, 64), generator=rng, dtype=torch.float32)

    first = high_freq_detail_mask(
        image,
        sensitivity=0.5,
        kernel_type="sobel",
        threshold_mode="relative_percentile",
    )
    second = high_freq_detail_mask(
        image,
        sensitivity=0.5,
        kernel_type="sobel",
        threshold_mode="relative_percentile",
    )

    assert torch.equal(first, second)


def test_mask_in_limits_region(rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 64, 64), generator=rng, dtype=torch.float32)
    mask = torch.zeros((1, 64, 64), dtype=torch.float32)
    mask[:, 16:48, 16:48] = 1.0

    output = high_freq_detail_mask(image, sensitivity=0.5, mask_bchw=mask)

    assert torch.count_nonzero(output[:, :16, :]) == 0
    assert torch.count_nonzero(output[:, :, :16]) == 0
    assert torch.count_nonzero(output[:, 48:, :]) == 0
    assert torch.count_nonzero(output[:, :, 48:]) == 0


def test_grain_uniform_noise_uniform_output() -> None:
    image = _make_image(1, 128, 128, seed=20260419)

    output = high_freq_detail_mask(image, sensitivity=0.5, kernel_type="laplacian")
    density = output.mean().item()

    assert 0.35 <= density <= 0.65


def test_out_of_range_clamp_warns(caplog: pytest.LogCaptureFixture, rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 32, 32), generator=rng, dtype=torch.float32)
    image[0, 0, 0, 0] = -0.1
    image[0, 1, 0, 1] = 1.3

    with caplog.at_level(logging.WARNING):
        output = high_freq_detail_mask(image, sensitivity=0.4)

    assert "clamped to [0,1]" in caplog.text
    assert torch.all((output >= 0.0) & (output <= 1.0))


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_device_cpu_cuda_parity(device_name: str, rng: torch.Generator) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this runner.")

    base = torch.rand((1, 3, 96, 96), generator=rng, dtype=torch.float32)
    reference = high_freq_detail_mask(base, sensitivity=0.5, kernel_type="laplacian")

    image = base.to(device=device_name)
    output = high_freq_detail_mask(image, sensitivity=0.5, kernel_type="laplacian")

    assert output.device.type == device_name
    assert torch.equal(output.cpu(), reference)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sensitivity": -0.1}, "sensitivity"),
        ({"sensitivity": 1.1}, "sensitivity"),
        ({"kernel_type": "bad"}, "kernel_type"),
        ({"threshold_mode": "bad"}, "threshold_mode"),
    ],
)
def test_raise_on_invalid_params(
    kwargs: dict[str, object], match: str, rng: torch.Generator
) -> None:
    image = torch.rand((1, 3, 32, 32), generator=rng, dtype=torch.float32)

    with pytest.raises(ValueError, match=match):
        high_freq_detail_mask(image, **kwargs)


def test_raise_on_invalid_mask_shape(rng: torch.Generator) -> None:
    image = torch.rand((2, 3, 32, 32), generator=rng, dtype=torch.float32)
    mask = torch.rand((3, 32, 32), generator=rng, dtype=torch.float32)

    with pytest.raises(ValueError, match="mask batch"):
        high_freq_detail_mask(image, mask_bchw=mask)
