from __future__ import annotations

import pytest
import torch
from kornia.color import rgb_to_ycbcr
from kornia.filters import unsharp_mask

from core.sharpen import smart_sharpen


def _rand(b: int = 1, h: int = 48, w: int = 48, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, h, w, 3, generator=g) * 0.6 + 0.2


def _step() -> torch.Tensor:
    img = torch.full((1, 32, 32, 3), 0.3)
    img[:, :, 16:] = 0.7
    return img


def test_type_and_shape_errors() -> None:
    with pytest.raises(TypeError):
        smart_sharpen("x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        smart_sharpen(torch.rand(1, 8, 8))
    with pytest.raises(ValueError):
        smart_sharpen(torch.rand(1, 8, 8, 4))
    with pytest.raises(ValueError):
        smart_sharpen(torch.zeros(1, 8, 8, 3, dtype=torch.int64))
    with pytest.raises(TypeError):
        smart_sharpen(_rand(), mask="m")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        smart_sharpen(_rand(), mask=torch.ones(1, 8, 8))
    with pytest.raises(ValueError):
        smart_sharpen(_rand(2), mask=torch.ones(3, 48, 48))
    with pytest.raises(ValueError):
        smart_sharpen(torch.rand(1, 8, 8, 3), radius=10.0)  # image too small


@pytest.mark.parametrize(
    "kwargs",
    [
        {"amount": -0.1},
        {"amount": 5.1},
        {"radius": 0.2},
        {"radius": 10.5},
        {"threshold": -0.01},
        {"threshold": 0.6},
        {"amount": True},
        {"amount": float("nan")},
        {"luminance_only": 1},
    ],
)
def test_range_errors(kwargs) -> None:
    with pytest.raises(ValueError):
        smart_sharpen(_rand(), **kwargs)


def test_shape_dtype() -> None:
    out = smart_sharpen(_rand(2))
    assert out.shape == (2, 48, 48, 3)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_amount_zero_identity() -> None:
    img = _rand()
    assert torch.equal(smart_sharpen(img, amount=0.0), img)


def test_flat_stays_flat() -> None:
    img = torch.full((1, 32, 32, 3), 0.4)
    img[..., 1] = 0.6
    for lum in (True, False):
        out = smart_sharpen(img, amount=3.0, luminance_only=lum)
        assert torch.allclose(out, img, atol=1e-5)


def test_matches_kornia_unsharp_mask() -> None:
    img = _rand(seed=1)
    radius = 1.5
    k = 2 * 5 + 1  # ceil(4.5) = 5
    ref = unsharp_mask(img.permute(0, 3, 1, 2), (k, k), (radius, radius), border_type="reflect")
    ref = ref.clamp(0, 1).permute(0, 2, 3, 1)
    out = smart_sharpen(img, amount=1.0, radius=radius, threshold=0.0, luminance_only=False)
    assert torch.allclose(out, ref, atol=1e-6)


def test_edge_overshoot() -> None:
    img = _step()
    out = smart_sharpen(img, amount=1.5, radius=1.0)
    assert out[0, 16, 15, 0] < 0.3  # dark side undershoots
    assert out[0, 16, 16, 0] > 0.7  # bright side overshoots
    assert out[0, 16, 4, 0] == pytest.approx(0.3, abs=1e-5)


def test_threshold_suppresses_noise_keeps_edges() -> None:
    g = torch.Generator().manual_seed(3)
    noise = (torch.rand(1, 32, 32, 1, generator=g) - 0.5) * 0.02
    img = (_step() + noise).clamp(0, 1)
    plain = smart_sharpen(img, amount=2.0, threshold=0.0)
    gated = smart_sharpen(img, amount=2.0, threshold=0.05)
    flat_plain = (plain[:, :, :8] - img[:, :, :8]).abs().max()
    flat_gated = (gated[:, :, :8] - img[:, :, :8]).abs().max()
    assert flat_gated < 1e-6 < flat_plain
    assert gated[0, 16, 16, 0] > img[0, 16, 16, 0] + 0.05  # strong edge still sharpened


def test_threshold_is_continuous() -> None:
    img = _rand(seed=5)
    a = smart_sharpen(img, amount=1.0, threshold=0.100)
    b = smart_sharpen(img, amount=1.0, threshold=0.101)
    assert (a - b).abs().max() < 0.01


def test_luminance_only_keeps_chroma() -> None:
    img = _rand(seed=7) * 0.8 + 0.1
    out = smart_sharpen(img, amount=0.5, radius=1.0, luminance_only=True)
    c0 = rgb_to_ycbcr(img.permute(0, 3, 1, 2))[:, 1:]
    c1 = rgb_to_ycbcr(out.permute(0, 3, 1, 2))[:, 1:]
    assert torch.allclose(c0, c1, atol=1e-5)
    assert not torch.allclose(out, img, atol=1e-3)


def test_mask_zero_identity_and_partial() -> None:
    img = _rand(seed=9)
    assert torch.allclose(smart_sharpen(img, mask=torch.zeros(1, 48, 48)), img, atol=1e-6)
    full = smart_sharpen(img, amount=1.0)
    half = smart_sharpen(img, amount=1.0, mask=torch.full((1, 48, 48), 0.5))
    expected = smart_sharpen(img, amount=0.5)
    assert torch.allclose(half, expected, atol=1e-5)
    assert not torch.allclose(half, full, atol=1e-4)


def test_mask_bool_and_batch_broadcast() -> None:
    img = _rand(3, seed=11)
    m = torch.zeros(1, 48, 48, dtype=torch.bool)
    m[:, :, 24:] = True
    out = smart_sharpen(img, mask=m)
    assert out.shape == img.shape
    assert torch.allclose(out[:, :, :20], img[:, :, :20], atol=1e-6)


def test_input_not_modified() -> None:
    img = _rand()
    before = img.clone()
    smart_sharpen(img)
    assert torch.equal(img, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu() -> None:
    img = _rand(seed=13)
    mask = torch.rand(1, 48, 48)
    cpu = smart_sharpen(img, amount=1.2, threshold=0.02, mask=mask)
    gpu = smart_sharpen(img.cuda(), amount=1.2, threshold=0.02, mask=mask.cuda())
    assert gpu.device.type == "cuda"
    assert torch.allclose(cpu, gpu.cpu(), atol=1e-4)
