from __future__ import annotations

import logging

import pytest
import torch
from kornia.filters import bilateral_blur, sobel
from kornia.metrics import ssim

from core import edge_aware_smooth


def _line_art_image(size: int = 128) -> torch.Tensor:
    image = torch.zeros((1, 3, size, size), dtype=torch.float32)
    image[:, :, 24:26, 12:-12] = 1.0
    image[:, :, 48:50, 12:-12] = 1.0
    image[:, :, 72:74, 12:-12] = 1.0
    image[:, :, 24:-24, 32:34] = 1.0
    image[:, :, 24:-24, 64:66] = 1.0
    image[:, :, 24:-24, 96:98] = 1.0
    return image


def _noise_image(rng: torch.Generator, size: int = 128) -> torch.Tensor:
    base = torch.full((1, 3, size, size), 0.5, dtype=torch.float32)
    noise = torch.randn((1, 3, size, size), generator=rng, dtype=torch.float32) * 0.05
    return (base + noise).clamp(0.0, 1.0)


def test_strength_zero_identity(rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 64, 64), generator=rng, dtype=torch.float32)

    output = edge_aware_smooth(image, strength=0.0)

    assert output.device.type == "cpu"
    assert torch.equal(output, image)


def test_mask_zero_identity(rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 64, 64), generator=rng, dtype=torch.float32)
    mask = torch.zeros((1, 1, 64, 64), dtype=torch.float32)

    output = edge_aware_smooth(image, strength=1.0, mask_bchw=mask)

    assert output.device.type == "cpu"
    assert torch.equal(output, image)


def test_mask_ones_vs_none_equiv(rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 64, 64), generator=rng, dtype=torch.float32)
    mask = torch.ones((1, 1, 64, 64), dtype=torch.float32)

    masked = edge_aware_smooth(image, strength=1.0, mask_bchw=mask)
    unmasked = edge_aware_smooth(image, strength=1.0)

    assert torch.allclose(masked, unmasked, atol=1e-6)


def test_edge_preservation_ssim() -> None:
    image = _line_art_image()
    output = edge_aware_smooth(image, strength=1.0, sigma_color=0.1, sigma_space=6.0)

    edge_band = sobel(image.mean(dim=1, keepdim=True)) > 0.1
    ssim_map = ssim(output, image, window_size=5)

    assert edge_band.any()
    assert ssim_map[edge_band.expand_as(ssim_map)].mean().item() >= 0.95


def test_smoothing_effectiveness_stddev(rng: torch.Generator) -> None:
    image = _noise_image(rng)
    output = edge_aware_smooth(image, strength=1.0, sigma_color=0.1, sigma_space=3.0)

    assert output.std().item() <= 0.5 * image.std().item()


def test_output_range(rng: torch.Generator) -> None:
    image = torch.rand((2, 3, 96, 96), generator=rng, dtype=torch.float32)
    mask = torch.rand((2, 1, 96, 96), generator=rng, dtype=torch.float32)

    output = edge_aware_smooth(
        image,
        strength=0.7,
        sigma_color=0.15,
        sigma_space=5.0,
        mask_bchw=mask,
    )

    assert output.device.type == "cpu"
    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)


@pytest.mark.parametrize(
    ("image", "kwargs", "match"),
    [
        (torch.rand((1, 1, 32, 32), dtype=torch.float32), {}, "3-channel"),
        (torch.rand((1, 4, 32, 32), dtype=torch.float32), {}, "3-channel"),
        (torch.rand((1, 3, 32, 32), dtype=torch.float32), {"sigma_color": 0.0}, "sigma_color"),
        (torch.rand((1, 3, 32, 32), dtype=torch.float32), {"sigma_space": -1.0}, "sigma_space"),
        (torch.rand((1, 3, 32, 32), dtype=torch.float32), {"strength": 1.5}, "strength"),
    ],
)
def test_reject_invalid_input(
    image: torch.Tensor, kwargs: dict[str, float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        edge_aware_smooth(image, **kwargs)


def test_mask_shape_mismatch(rng: torch.Generator) -> None:
    image = torch.rand((4, 3, 64, 64), generator=rng, dtype=torch.float32)

    with pytest.raises(ValueError, match="mask HxW"):
        edge_aware_smooth(image, mask_bchw=torch.ones((4, 1, 32, 64), dtype=torch.float32))

    with pytest.raises(ValueError, match="mask batch"):
        edge_aware_smooth(image, mask_bchw=torch.ones((2, 1, 64, 64), dtype=torch.float32))


def test_mask_broadcast(rng: torch.Generator) -> None:
    image = torch.rand((4, 3, 64, 64), generator=rng, dtype=torch.float32)
    mask = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    mask[:, :, 16:48, 16:48] = 1.0

    output = edge_aware_smooth(image, strength=1.0, mask_bchw=mask)
    expected = mask * bilateral_blur(
        image,
        kernel_size=(37, 37),
        sigma_color=0.1,
        sigma_space=(6.0, 6.0),
        border_type="replicate",
        color_distance_type="l1",
    ) + (1.0 - mask) * image

    assert output.shape == image.shape
    assert torch.allclose(output, expected.clamp(0.0, 1.0), atol=1e-6)


@pytest.mark.parametrize("shape", [(1, 3, 512, 512), (4, 3, 1024, 1024)])
def test_shapes(shape: tuple[int, int, int, int], rng: torch.Generator) -> None:
    image = torch.rand(shape, generator=rng, dtype=torch.float32)
    tile_mode = shape[-1] >= 1024

    output = edge_aware_smooth(
        image,
        strength=0.5,
        sigma_color=0.1,
        sigma_space=2.0,
        tile_mode=tile_mode,
    )

    assert output.shape == image.shape
    assert output.dtype == torch.float32
    assert output.device.type == "cpu"


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_device_parity(device_name: str, rng: torch.Generator) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this runner.")

    image = torch.rand((1, 3, 64, 64), generator=rng, dtype=torch.float32).to(device_name)
    zero_mask = torch.zeros((1, 1, 64, 64), dtype=torch.float32, device=device_name)

    strength_zero = edge_aware_smooth(image, strength=0.0, device=device_name)
    mask_zero = edge_aware_smooth(image, strength=1.0, mask_bchw=zero_mask, device=device_name)

    assert strength_zero.device.type == "cpu"
    assert mask_zero.device.type == "cpu"
    assert torch.equal(strength_zero, image.cpu())
    assert torch.equal(mask_zero, image.cpu())


def test_warning_paths(caplog: pytest.LogCaptureFixture, rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 32, 32), generator=rng, dtype=torch.float32)
    mask = torch.ones((1, 1, 32, 32), dtype=torch.float32) * 1.5
    image[0, 0, 0, 0] = 1.2

    with caplog.at_level(logging.WARNING):
        output = edge_aware_smooth(image, sigma_color=1.5, mask_bchw=mask)

    assert "clamped to [0,1]" in caplog.text
    assert "values > 1 will smooth across most edges" in caplog.text
    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)


def test_sigma_space_cap(rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 32, 32), generator=rng, dtype=torch.float32)

    with pytest.raises(ValueError, match=r"must be in \[1.0, 8.0\]"):
        edge_aware_smooth(image, sigma_space=9.0)


def test_sigma_space_at_max(rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 32, 32), generator=rng, dtype=torch.float32)

    output = edge_aware_smooth(image, sigma_space=8.0, tile_mode=True)

    assert output.shape == image.shape
    assert output.device.type == "cpu"


def test_memory_budget_error() -> None:
    image = torch.zeros((1, 3, 4096, 4096), dtype=torch.float32)

    with pytest.raises(RuntimeError, match=r"would allocate.*GB.*enable tile_mode"):
        edge_aware_smooth(image, sigma_space=6.0, tile_mode=False)


def test_tile_mode_correctness(rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 512, 512), generator=rng, dtype=torch.float32)

    tiled = edge_aware_smooth(
        image,
        strength=1.0,
        sigma_color=0.1,
        sigma_space=1.0,
        tile_mode=True,
    )
    direct = edge_aware_smooth(
        image,
        strength=1.0,
        sigma_color=0.1,
        sigma_space=1.0,
        tile_mode=False,
    )

    assert torch.allclose(tiled, direct, atol=1e-4)


def test_device_auto_resolves(rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 64, 64), generator=rng, dtype=torch.float32)

    output = edge_aware_smooth(image, device="auto")

    assert output.device.type == "cpu"
    assert output.shape == image.shape


def test_device_cuda_unavailable_raises(rng: torch.Generator) -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this runner.")

    image = torch.rand((1, 3, 64, 64), generator=rng, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        edge_aware_smooth(image, device="cuda")


def test_device_cpu_forced(rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 64, 64), generator=rng, dtype=torch.float32)
    image_before = image.clone()

    output = edge_aware_smooth(image, device="cpu")

    assert output.device.type == "cpu"
    assert torch.equal(image, image_before)


def test_invalid_device_value(rng: torch.Generator) -> None:
    image = torch.rand((1, 3, 64, 64), generator=rng, dtype=torch.float32)

    with pytest.raises(ValueError, match="device must be one of"):
        edge_aware_smooth(image, device="mps")


@pytest.mark.parametrize("tile_mode", [1, "true"])
def test_invalid_tile_mode_type(rng: torch.Generator, tile_mode: object) -> None:
    image = torch.rand((1, 3, 64, 64), generator=rng, dtype=torch.float32)

    with pytest.raises(ValueError, match="tile_mode must be a bool"):
        edge_aware_smooth(image, tile_mode=tile_mode)  # type: ignore[arg-type]
