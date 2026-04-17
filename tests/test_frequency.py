from __future__ import annotations

import logging

import pytest
import torch
from kornia.filters import gaussian_blur2d

from core import frequency_separation


def _make_case_image(
    case_name: str,
    shape: tuple[int, int, int, int],
    rng: torch.Generator,
) -> torch.Tensor:
    if case_name == "random":
        return torch.rand(shape, generator=rng)
    if case_name == "zeros":
        return torch.zeros(shape)
    if case_name == "ones":
        return torch.ones(shape)
    if case_name == "one-pixel-differ":
        image = torch.zeros(shape)
        image[0, 0, shape[-2] // 2, shape[-1] // 2] = 1.0
        return image
    raise ValueError(f"Unknown case {case_name!r}.")


def _assert_reconstruction(
    original: torch.Tensor, low: torch.Tensor, high: torch.Tensor, atol: float
) -> None:
    reconstructed = low + high
    assert torch.allclose(reconstructed.float(), original.float(), atol=atol)


@pytest.mark.parametrize(
    ("shape", "case_name"),
    [
        ((1, 3, 512, 512), "random"),
        ((1, 3, 2048, 2048), "random"),
        ((4, 3, 512, 512), "random"),
        ((1, 3, 512, 512), "zeros"),
        ((1, 3, 512, 512), "ones"),
        ((1, 3, 512, 512), "one-pixel-differ"),
    ],
    ids=[
        "b1-512-random",
        "b1-2048-random",
        "b4-512-random",
        "b1-512-zeros",
        "b1-512-ones",
        "b1-512-one-pixel-differ",
    ],
)
def test_reconstruction_invariant_float32(
    shape: tuple[int, int, int, int], case_name: str, rng: torch.Generator
) -> None:
    image = _make_case_image(case_name, shape, rng)
    low, high = frequency_separation(image, radius=8, sigma=0.0, precision="float32")

    assert low.shape == image.shape
    assert high.shape == image.shape
    assert low.dtype == torch.float32
    assert high.dtype == torch.float32
    _assert_reconstruction(image, low, high, atol=1e-5)


@pytest.mark.parametrize(
    ("shape", "case_name"),
    [
        ((1, 3, 512, 512), "random"),
        ((1, 3, 2048, 2048), "random"),
        ((4, 3, 512, 512), "random"),
        ((1, 3, 512, 512), "zeros"),
        ((1, 3, 512, 512), "ones"),
        ((1, 3, 512, 512), "one-pixel-differ"),
    ],
    ids=[
        "b1-512-random",
        "b1-2048-random",
        "b4-512-random",
        "b1-512-zeros",
        "b1-512-ones",
        "b1-512-one-pixel-differ",
    ],
)
def test_reconstruction_invariant_float16(
    shape: tuple[int, int, int, int], case_name: str, rng: torch.Generator
) -> None:
    image = _make_case_image(case_name, shape, rng)
    low, high = frequency_separation(image, radius=8, sigma=0.0, precision="float16")

    assert low.shape == image.shape
    assert high.shape == image.shape
    assert low.dtype == torch.float16
    assert high.dtype == torch.float16
    _assert_reconstruction(image, low, high, atol=1e-3)


def test_reject_grayscale() -> None:
    image = torch.rand(1, 1, 64, 64)

    with pytest.raises(ValueError, match="3-channel"):
        frequency_separation(image, radius=4)


def test_reject_rgba() -> None:
    image = torch.rand(1, 4, 64, 64)

    with pytest.raises(ValueError, match="SplitAlpha|4"):
        frequency_separation(image, radius=4)


@pytest.mark.parametrize("radius", [0, -1])
def test_reject_invalid_radius(radius: int) -> None:
    image = torch.rand(1, 3, 64, 64)

    with pytest.raises(ValueError, match="radius must be >= 1"):
        frequency_separation(image, radius=radius)


def test_reject_invalid_sigma() -> None:
    image = torch.rand(1, 3, 64, 64)

    with pytest.raises(ValueError, match="sigma must be >= 0"):
        frequency_separation(image, radius=4, sigma=-0.5)


def test_sigma_auto_compute(rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 64, 64), generator=rng)
    low, high = frequency_separation(image, radius=8, sigma=0.0, precision="float32")

    expected_low = gaussian_blur2d(
        image,
        kernel_size=(17, 17),
        sigma=(4.0, 4.0),
        border_type="replicate",
    )

    assert torch.allclose(low, expected_low, atol=1e-6)
    _assert_reconstruction(image, low, high, atol=1e-5)


def test_sigma_override_warning(caplog: pytest.LogCaptureFixture) -> None:
    image = torch.rand(1, 3, 64, 64)

    with caplog.at_level(logging.WARNING):
        frequency_separation(image, radius=4, sigma=10.0, precision="float32")

    assert "sigma > radius/2" in caplog.text


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_device_parity(device_name: str, rng: torch.Generator) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this runner.")

    base = torch.rand((1, 3, 128, 128), generator=rng)
    reference_low, reference_high = frequency_separation(base, radius=6, precision="float32")

    image = base.to(device=device_name)
    low, high = frequency_separation(image, radius=6, precision="float32")

    assert low.device.type == device_name
    assert high.device.type == device_name
    _assert_reconstruction(base, low.cpu(), high.cpu(), atol=1e-5)
    assert torch.allclose(low.cpu(), reference_low, atol=1e-5)
    assert torch.allclose(high.cpu(), reference_high, atol=1e-5)


def test_float16_on_cpu_warns(caplog: pytest.LogCaptureFixture, rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 128, 128), generator=rng)

    with caplog.at_level(logging.WARNING):
        low, high = frequency_separation(image, radius=6, precision="float16")

    cpu_messages = [
        record.message for record in caplog.records if "float16 on CPU" in record.message
    ]
    assert len(cpu_messages) == 1
    _assert_reconstruction(image, low, high, atol=1e-3)
