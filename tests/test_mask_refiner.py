from __future__ import annotations

import logging

import pytest
import torch
from kornia.filters import gaussian_blur2d
from kornia.morphology import dilation, erosion

from core import subpixel_mask_refine


def _disk_mask(batch: int, size: int, radius: int) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    center = (size - 1) / 2.0
    mask = ((yy - center) ** 2 + (xx - center) ** 2 <= radius**2).to(dtype=torch.float32)
    return mask.unsqueeze(0).repeat(batch, 1, 1)


def _binary_mask(mask_bhw: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (mask_bhw.unsqueeze(1) > threshold).to(dtype=torch.float32)


def _inside_core_reference(mask_bhw: torch.Tensor, erosion_radius: int) -> torch.Tensor:
    mask_binary = _binary_mask(mask_bhw)
    if erosion_radius == 0:
        return mask_binary.squeeze(1).bool()

    kernel = torch.ones((2 * erosion_radius + 1, 2 * erosion_radius + 1), dtype=torch.float32)
    return erosion(mask_binary, kernel).squeeze(1).bool()


def _outside_core_reference(mask_bhw: torch.Tensor, dilation_radius: int) -> torch.Tensor:
    mask_binary = _binary_mask(mask_bhw)
    if dilation_radius == 0:
        return (~mask_binary.bool()).squeeze(1)

    kernel = torch.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1), dtype=torch.float32)
    return (1.0 - dilation(mask_binary, kernel)).squeeze(1).bool()


def test_inside_core_invariant() -> None:
    mask = _disk_mask(batch=1, size=65, radius=20)
    refined = subpixel_mask_refine(mask, erosion_radius=3, dilation_radius=4, feather_sigma=2.0)
    inside_core = _inside_core_reference(mask, erosion_radius=3)

    assert inside_core.any()
    assert torch.all(refined[inside_core] == 1.0)


def test_outside_core_invariant() -> None:
    mask = _disk_mask(batch=1, size=65, radius=20)
    refined = subpixel_mask_refine(mask, erosion_radius=3, dilation_radius=4, feather_sigma=2.0)
    outside_core = _outside_core_reference(mask, dilation_radius=4)

    assert outside_core.any()
    assert torch.all(refined[outside_core] == 0.0)


def test_band_values_in_range() -> None:
    mask = _disk_mask(batch=1, size=65, radius=20)
    refined = subpixel_mask_refine(mask, erosion_radius=3, dilation_radius=4, feather_sigma=2.0)
    inside_core = _inside_core_reference(mask, erosion_radius=3)
    outside_core = _outside_core_reference(mask, dilation_radius=4)
    band = ~(inside_core | outside_core)

    assert band.any()
    assert torch.all(refined[band] >= 0.0)
    assert torch.all(refined[band] <= 1.0)


def test_output_range() -> None:
    mask = _disk_mask(batch=2, size=64, radius=18)
    refined = subpixel_mask_refine(mask, erosion_radius=8, dilation_radius=2, feather_sigma=2.0)

    assert torch.all(refined >= 0.0)
    assert torch.all(refined <= 1.0)


@pytest.mark.parametrize("shape", [(1, 512, 512), (4, 1024, 1024)])
def test_shapes(shape: tuple[int, int, int]) -> None:
    mask = torch.zeros(shape, dtype=torch.float32)
    refined = subpixel_mask_refine(mask)

    assert refined.shape == mask.shape
    assert refined.dtype == torch.float32


def test_edge_cases() -> None:
    all_zero = torch.zeros((1, 64, 64), dtype=torch.float32)
    all_one = torch.ones((1, 64, 64), dtype=torch.float32)
    one_pixel = torch.zeros((1, 33, 33), dtype=torch.float32)
    one_pixel[0, 16, 16] = 1.0

    refined_zero = subpixel_mask_refine(
        all_zero,
        erosion_radius=2,
        dilation_radius=4,
        feather_sigma=2.0,
    )
    refined_one = subpixel_mask_refine(
        all_one,
        erosion_radius=2,
        dilation_radius=4,
        feather_sigma=2.0,
    )
    refined_dot = subpixel_mask_refine(
        one_pixel,
        erosion_radius=0,
        dilation_radius=0,
        feather_sigma=1.5,
    )

    assert torch.equal(refined_zero, all_zero)
    assert torch.allclose(refined_one, all_one, atol=1e-5)
    assert torch.allclose(refined_dot, refined_dot.flip(-1), atol=1e-6)
    assert torch.allclose(refined_dot, refined_dot.flip(-2), atol=1e-6)


def test_erosion_zero() -> None:
    mask = _disk_mask(batch=1, size=65, radius=20)
    refined = subpixel_mask_refine(mask, erosion_radius=0, dilation_radius=4, feather_sigma=2.0)
    mask_binary = _binary_mask(mask).squeeze(1).bool()

    assert torch.all(refined[mask_binary] == 1.0)


def test_dilation_zero() -> None:
    mask = _disk_mask(batch=1, size=65, radius=20)
    refined = subpixel_mask_refine(mask, erosion_radius=3, dilation_radius=0, feather_sigma=2.0)
    outside_core = (~_binary_mask(mask).bool()).squeeze(1)

    assert torch.all(refined[outside_core] == 0.0)


def test_erosion_dilation_both_zero(caplog: pytest.LogCaptureFixture) -> None:
    mask = _disk_mask(batch=1, size=33, radius=10)
    mask_binary = _binary_mask(mask)

    with caplog.at_level(logging.WARNING):
        refined = subpixel_mask_refine(
            mask,
            erosion_radius=0,
            dilation_radius=0,
            feather_sigma=2.0,
        )

    expected = (
        gaussian_blur2d(
            mask_binary,
            kernel_size=(3, 3),
            sigma=(2.0, 2.0),
            border_type="replicate",
        )
        .squeeze(1)
        .clamp(0.0, 1.0)
    )

    assert "no protected zone" in caplog.text
    assert torch.allclose(refined, expected, atol=1e-6)


def test_erosion_greater_than_dilation(caplog: pytest.LogCaptureFixture) -> None:
    mask = _disk_mask(batch=1, size=65, radius=20)

    with caplog.at_level(logging.WARNING):
        refined = subpixel_mask_refine(mask, erosion_radius=8, dilation_radius=2, feather_sigma=2.0)

    assert "unusual config" in caplog.text
    assert torch.all(refined >= 0.0)
    assert torch.all(refined <= 1.0)


def test_grayscale_warning(caplog: pytest.LogCaptureFixture) -> None:
    ramp = torch.linspace(0.0, 1.0, 64, dtype=torch.float32).view(1, 1, 64).repeat(1, 64, 1)

    with caplog.at_level(logging.WARNING):
        subpixel_mask_refine(ramp, erosion_radius=2, dilation_radius=4, feather_sigma=2.0)

    messages = [
        record.message for record in caplog.records if "not really binary" in record.message
    ]
    assert len(messages) == 1


def test_out_of_range_clamp(caplog: pytest.LogCaptureFixture) -> None:
    mask = torch.zeros((1, 32, 32), dtype=torch.float32)
    mask[0, 0, 0] = -0.1
    mask[0, 0, 1] = 1.5

    with caplog.at_level(logging.WARNING):
        refined = subpixel_mask_refine(mask, erosion_radius=2, dilation_radius=4, feather_sigma=2.0)

    messages = [record.message for record in caplog.records if "clamped to [0,1]" in record.message]
    assert len(messages) == 1
    assert torch.all(refined >= 0.0)
    assert torch.all(refined <= 1.0)


def test_reject_invalid_shape() -> None:
    mask = torch.rand((1, 3, 64, 64), dtype=torch.float32)

    with pytest.raises(ValueError, match="BHW|BC1HW|C=3"):
        subpixel_mask_refine(mask)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"erosion_radius": -1}, "erosion_radius"),
        ({"dilation_radius": 65}, "dilation_radius"),
        ({"feather_sigma": 0.0}, "feather_sigma"),
        ({"threshold": -0.5}, "threshold"),
        ({"threshold": 1.5}, "threshold"),
    ],
)
def test_reject_invalid_params(kwargs: dict[str, float | int], match: str) -> None:
    mask = torch.zeros((1, 32, 32), dtype=torch.float32)

    with pytest.raises(ValueError, match=match):
        subpixel_mask_refine(mask, **kwargs)


def test_reject_bool_input(caplog: pytest.LogCaptureFixture) -> None:
    mask = torch.zeros((1, 64, 64), dtype=torch.bool)
    mask[0, 16:48, 16:48] = True

    with caplog.at_level(logging.WARNING):
        refined = subpixel_mask_refine(mask, erosion_radius=2, dilation_radius=4, feather_sigma=2.0)

    messages = [
        record.message for record in caplog.records if "bool mask input cast" in record.message
    ]
    assert len(messages) == 1
    assert refined.dtype == torch.float32


def test_reject_uint8_input() -> None:
    mask = torch.zeros((1, 64, 64), dtype=torch.uint8)

    with pytest.raises(ValueError, match="Expected float tensor"):
        subpixel_mask_refine(mask)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_device_parity(device_name: str) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this runner.")

    base = _disk_mask(batch=1, size=65, radius=20)
    reference = subpixel_mask_refine(base, erosion_radius=3, dilation_radius=4, feather_sigma=2.0)

    mask = base.to(device=device_name)
    refined = subpixel_mask_refine(mask, erosion_radius=3, dilation_radius=4, feather_sigma=2.0)

    assert refined.device.type == device_name
    assert torch.allclose(refined.cpu(), reference, atol=1e-5)


def test_accepts_bc1hw() -> None:
    mask_bhw = _disk_mask(batch=1, size=64, radius=18)
    mask_bc1hw = mask_bhw.unsqueeze(1)

    refined_bhw = subpixel_mask_refine(
        mask_bhw,
        erosion_radius=2,
        dilation_radius=4,
        feather_sigma=2.0,
    )
    refined_bc1hw = subpixel_mask_refine(
        mask_bc1hw,
        erosion_radius=2,
        dilation_radius=4,
        feather_sigma=2.0,
    )

    assert refined_bc1hw.shape == mask_bc1hw.shape
    assert torch.allclose(refined_bc1hw.squeeze(1), refined_bhw, atol=1e-6)
