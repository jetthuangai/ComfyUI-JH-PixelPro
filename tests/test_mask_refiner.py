from __future__ import annotations

import logging

import pytest
import torch
from kornia.filters import gaussian_blur2d
from kornia.morphology import dilation, erosion

from core import subpixel_mask_refine
from core.mask_refiner import _feather_kernel_size


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
            kernel_size=(13, 13),
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


def _structured_masks() -> list[torch.Tensor]:
    border_rect = torch.zeros((1, 1, 41, 53), dtype=torch.float32)
    border_rect[..., :17, 9:40] = 1.0
    return [
        _disk_mask(batch=2, size=49, radius=15).unsqueeze(1),
        border_rect,
        torch.ones((1, 1, 23, 31), dtype=torch.float32),
        torch.zeros((1, 1, 3, 5), dtype=torch.float32).index_fill_(3, torch.tensor([2]), 1.0),
    ]


@pytest.mark.parametrize("radius", [1, 3, 16, 64])
def test_square_morphology_matches_kornia_exactly(radius: int) -> None:
    # Regression: square max-pool morphology must equal kornia geodesic erosion/dilation.
    from core.mask_refiner import _dilate_square, _erode_square

    kernel = torch.ones((2 * radius + 1, 2 * radius + 1), dtype=torch.float32)
    for mask in _structured_masks():
        assert torch.equal(_dilate_square(mask, radius), dilation(mask, kernel))
        assert torch.equal(_erode_square(mask, radius), erosion(mask, kernel))


@pytest.mark.parametrize(("erosion_radius", "dilation_radius"), [(2, 4), (5, 3), (0, 6), (6, 0)])
def test_refine_output_matches_kornia_reference(erosion_radius: int, dilation_radius: int) -> None:
    mask = _disk_mask(batch=2, size=65, radius=20)
    refined = subpixel_mask_refine(
        mask, erosion_radius=erosion_radius, dilation_radius=dilation_radius, feather_sigma=2.0
    )
    binary = _binary_mask(mask)
    kernel_size = max(2 * max(erosion_radius, dilation_radius, 1) + 1, 13)
    feather = gaussian_blur2d(
        binary, (kernel_size, kernel_size), (2.0, 2.0), border_type="replicate"
    )
    inside = _inside_core_reference(mask, erosion_radius).unsqueeze(1).float()
    outside = _outside_core_reference(mask, dilation_radius).unsqueeze(1).float()
    expected = (inside + (1.0 - inside - outside) * feather).clamp(0.0, 1.0).squeeze(1)
    assert torch.equal(refined, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA memory check")
def test_large_radius_memory_stays_bounded() -> None:
    # Regression: kornia.morphology unfold engine needed ~H*W*k^2 floats (5.5 GB at r=16, 1024^2).
    mask = (torch.rand((1, 1024, 1024), device="cuda") > 0.5).to(dtype=torch.float32)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    subpixel_mask_refine(mask, erosion_radius=64, dilation_radius=64, feather_sigma=2.0)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - baseline < 512 * 1024**2


@pytest.mark.parametrize(("feather_sigma", "expected"), [(2.0, 13), (0.1, 9), (8.0, 49)])
def test_feather_kernel_covers_three_sigma(feather_sigma: float, expected: int) -> None:
    assert _feather_kernel_size(feather_sigma, 4, 256, 256) == expected


def test_feather_kernel_capped_by_image_extent() -> None:
    assert _feather_kernel_size(500.0, 2, 16, 24) == 49


def test_large_feather_with_small_radius_is_not_truncated() -> None:
    # Regression: the kernel used to be sized by radius only (5 taps for r=2), turning a
    # sigma=8 feather into a narrow box. The band must now match an untruncated Gaussian.
    mask = _disk_mask(batch=1, size=97, radius=30)
    refined = subpixel_mask_refine(mask, erosion_radius=2, dilation_radius=2, feather_sigma=8.0)
    binary = _binary_mask(mask)
    feather = gaussian_blur2d(binary, (49, 49), (8.0, 8.0), border_type="replicate")
    truncated = gaussian_blur2d(binary, (5, 5), (8.0, 8.0), border_type="replicate")
    inside = _inside_core_reference(mask, 2).unsqueeze(1).float()
    outside = _outside_core_reference(mask, 2).unsqueeze(1).float()
    band = 1.0 - inside - outside
    expected = (inside + band * feather).clamp(0.0, 1.0).squeeze(1)
    old = (inside + band * truncated).clamp(0.0, 1.0).squeeze(1)
    assert torch.equal(refined, expected)
    assert (refined - old).abs().max() > 0.05
