from __future__ import annotations

import pytest
import torch
from kornia.color import rgb_to_lab

from core.clahe import clahe_local_contrast


def _lab(image: torch.Tensor) -> torch.Tensor:
    return rgb_to_lab(image.permute(0, 3, 1, 2))


def _low_contrast(batch: int = 1, size: int = 128, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    ramp = torch.linspace(0.35, 0.55, size).view(1, 1, size, 1).expand(batch, size, size, 3)
    tint = torch.tensor([1.0, 0.9, 0.8]).view(1, 1, 1, 3)
    noise = torch.randn(batch, size, size, 1, generator=generator) * 0.01
    return (ramp * tint + noise).clamp(0.0, 1.0).contiguous()


def test_output_shape_dtype_range() -> None:
    out = clahe_local_contrast(_low_contrast(2))
    assert out.shape == (2, 128, 128, 3)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_strength_zero_is_identity() -> None:
    image = _low_contrast()
    out = clahe_local_contrast(image, strength=0.0)
    assert torch.allclose(out, image, atol=1e-6)


def test_mask_zero_is_identity() -> None:
    image = _low_contrast()
    out = clahe_local_contrast(image, mask=torch.zeros(1, 128, 128))
    assert torch.allclose(out, image, atol=1e-6)


def test_flat_grey_stays_flat() -> None:
    image = torch.full((1, 128, 128, 3), 0.5)
    out = clahe_local_contrast(image)
    assert out.std() < 1e-3
    assert torch.allclose(out, image, atol=0.02)


def test_low_contrast_gains_local_contrast() -> None:
    image = _low_contrast()
    out = clahe_local_contrast(image, clip_limit=8.0, grid_size=4)
    assert _lab(out)[:, 0].std() > _lab(image)[:, 0].std()


def test_chroma_preserved() -> None:
    # Mid-tone saturated image: no clamp at the gamut edge, so a*/b* must survive.
    image = _low_contrast() * torch.tensor([0.9, 0.6, 0.5]).view(1, 1, 1, 3)
    out = clahe_local_contrast(image, clip_limit=4.0)
    assert not torch.allclose(out, image, atol=1e-3)
    lab_in, lab_out = _lab(image), _lab(out)
    assert torch.allclose(lab_in[:, 1:], lab_out[:, 1:], atol=0.5)


def test_strength_blend_is_between() -> None:
    image = _low_contrast()
    full = clahe_local_contrast(image, clip_limit=8.0)
    half = clahe_local_contrast(image, clip_limit=8.0, strength=0.5)
    assert torch.allclose(half, image + (full - image) * 0.5, atol=1e-5)


def test_mask_restricts_effect() -> None:
    image = _low_contrast()
    mask = torch.zeros(1, 128, 128)
    mask[:, :, 64:] = 1.0
    full = clahe_local_contrast(image, clip_limit=8.0)
    out = clahe_local_contrast(image, clip_limit=8.0, mask=mask)
    assert torch.allclose(out[:, :, :64], image[:, :, :64], atol=1e-6)
    assert torch.allclose(out[:, :, 64:], full[:, :, 64:], atol=1e-6)


def test_mask_batch_one_broadcasts() -> None:
    image = _low_contrast(2)
    out = clahe_local_contrast(image, mask=torch.ones(1, 128, 128))
    assert out.shape == image.shape
    assert torch.allclose(out, clahe_local_contrast(image), atol=1e-6)


def test_non_divisible_size() -> None:
    generator = torch.Generator().manual_seed(1)
    image = torch.rand(1, 101, 77, 3, generator=generator)
    out = clahe_local_contrast(image, grid_size=7)
    assert out.shape == image.shape
    assert torch.isfinite(out).all()


def test_input_not_mutated() -> None:
    image = _low_contrast()
    before = image.clone()
    clahe_local_contrast(image)
    assert torch.equal(image, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_matches_cpu() -> None:
    image = _low_contrast()
    cpu = clahe_local_contrast(image, clip_limit=4.0)
    gpu = clahe_local_contrast(image.cuda(), clip_limit=4.0)
    assert gpu.device.type == "cuda"
    assert torch.allclose(cpu, gpu.cpu(), atol=0.02)


def test_validation_errors() -> None:
    image = _low_contrast()
    with pytest.raises(TypeError):
        clahe_local_contrast("x")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        clahe_local_contrast(image, mask="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"\(B,H,W,3\)"):
        clahe_local_contrast(torch.rand(1, 8, 8, 4))
    with pytest.raises(ValueError, match="floating"):
        clahe_local_contrast(torch.zeros(1, 64, 64, 3, dtype=torch.uint8))
    for kwargs in (
        {"clip_limit": 0.05},
        {"clip_limit": 41.0},
        {"clip_limit": True},
        {"grid_size": 1},
        {"grid_size": 33},
        {"grid_size": 8.0},
        {"grid_size": True},
        {"strength": -0.1},
        {"strength": 1.1},
    ):
        with pytest.raises(ValueError):
            clahe_local_contrast(image, **kwargs)
    with pytest.raises(ValueError, match="mask must have shape"):
        clahe_local_contrast(image, mask=torch.ones(128, 128))
    with pytest.raises(ValueError, match="spatial"):
        clahe_local_contrast(image, mask=torch.ones(1, 64, 64))
    with pytest.raises(ValueError, match="batch"):
        clahe_local_contrast(image, mask=torch.ones(3, 128, 128))


@pytest.mark.parametrize("size", [6, 64, 100, 256, 300, 777])
def test_small_image_flat_stays_flat(size: int) -> None:
    # Regression: small tiles bias kornia's clip threshold (flat 0.5 -> up to ~0.98).
    image = torch.full((1, size, size, 3), 0.5)
    out = clahe_local_contrast(image, clip_limit=2.0, grid_size=8)
    assert (out - image).abs().max() < 0.02


def test_effective_grid_keeps_tiles_at_least_80_px() -> None:
    from core.clahe import _effective_grid

    assert _effective_grid(8, 2048, 1024) == 8
    assert _effective_grid(8, 400, 640) == 5
    assert _effective_grid(8, 79, 79) == 0
