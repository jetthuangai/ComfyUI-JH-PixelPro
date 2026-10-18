from __future__ import annotations

import pytest
import torch

from core.guided import (
    box_filter,
    guided_filter,
    guided_upsample,
    make_proxy,
    proxy_size,
    resize_to,
)


def _smooth(height: int = 64, width: int = 96, channels: int = 1) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, height).view(1, 1, height, 1)
    x = torch.linspace(0.0, 1.0, width).view(1, 1, 1, width)
    base = 0.25 + 0.5 * (0.5 * y + 0.5 * x)
    return base.expand(1, channels, height, width).contiguous()


def _step(height: int = 64, width: int = 64) -> torch.Tensor:
    out = torch.full((1, 1, height, width), 0.2)
    out[..., width // 2 :] = 0.8
    return out


def test_box_filter_keeps_constant_and_any_size() -> None:
    x = torch.full((2, 3, 5, 7), 0.4)
    out = box_filter(x, 9)
    assert out.shape == x.shape
    assert torch.allclose(out, x)


def test_flat_image_unchanged() -> None:
    flat = torch.full((1, 3, 40, 50), 0.37)
    out = guided_filter(flat, flat, radius=6, eps=1e-2)
    assert torch.allclose(out, flat, atol=1e-6)


def test_small_eps_preserves_step_edge() -> None:
    step = _step()
    out = guided_filter(step, step, radius=4, eps=1e-6)
    assert (out - step).abs().max().item() < 1e-3


def test_large_eps_smooths_noise() -> None:
    torch.manual_seed(0)
    noise = 0.5 + 0.1 * torch.randn(1, 1, 64, 64)
    out = guided_filter(noise, noise, radius=4, eps=1.0)
    assert out.std().item() < noise.std().item() * 0.3


@pytest.mark.parametrize("channels", [1, 3])
def test_subsample_handles_non_divisible_sizes(channels: int) -> None:
    guide = _smooth(101, 203, channels)
    src = _smooth(101, 203, 2)[:, :1].expand(1, 2, 101, 203).contiguous()
    full = guided_filter(guide, src, radius=8, eps=1e-3)
    fast = guided_filter(guide, src, radius=8, eps=1e-3, subsample=4)
    assert fast.shape == src.shape
    assert (fast - full).abs().mean().item() < 0.01


def test_rgb_guide_keeps_isoluminant_colour_edge() -> None:
    # Red and green halves with equal luma: a grey guide cannot see this edge.
    height, width = 32, 64
    left = torch.tensor([0.587, 0.0, 0.0]).view(1, 3, 1, 1)
    right = torch.tensor([0.0, 0.299, 0.0]).view(1, 3, 1, 1)
    rgb = torch.where(torch.arange(width).view(1, 1, 1, width) < width // 2, left, right)
    rgb = rgb.expand(1, 3, height, width).contiguous()
    mask = (
        (torch.arange(width).view(1, 1, 1, width) < width // 2).float().expand(1, 1, height, width)
    )
    luma = (rgb * torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(1, keepdim=True)

    rgb_out = guided_filter(rgb, mask, radius=6, eps=1e-4)
    grey_out = guided_filter(luma, mask, radius=6, eps=1e-4)
    edge_cols = slice(width // 2 - 3, width // 2 + 3)
    rgb_err = (rgb_out - mask)[..., edge_cols].abs().mean().item()
    grey_err = (grey_out - mask)[..., edge_cols].abs().mean().item()
    assert rgb_err < 0.02
    assert grey_err > 0.2


def test_guided_upsample_recovers_linear_relation() -> None:
    guide_full = _smooth(120, 160)
    guide_low = make_proxy(guide_full, 40)
    result_low = guide_low * 0.5 + 0.1
    out = guided_upsample(guide_full, guide_low, result_low, radius=2, eps=1e-6)
    assert out.shape == guide_full.shape
    assert (out - (guide_full * 0.5 + 0.1)).abs().max().item() < 1e-3


def test_guided_upsample_keeps_full_res_edge() -> None:
    step_full = _step(128, 128)
    step_low = make_proxy(step_full, 32)
    target = step_full * 2.0
    out = guided_upsample(step_full, step_low, step_low * 2.0, radius=2, eps=1e-5)
    naive = resize_to(step_low * 2.0, (128, 128))
    # Fast guided upsampling leaves a small seam at the edge (bilinear a/b), far
    # below a plain bilinear upsample, and is exact away from it.
    assert (out - target).abs().max().item() < 0.1
    assert (out - target).abs().max().item() < 0.25 * (naive - target).abs().max().item()
    assert (out - target)[..., :, :56].abs().max().item() < 1e-4


def test_proxy_size_and_noop() -> None:
    assert proxy_size(3000, 4000, 1024) == (768, 1024)
    assert proxy_size(500, 300, 1024) == (500, 300)
    small = torch.rand(1, 3, 20, 30)
    assert torch.equal(make_proxy(small, 64), small)


def test_validation_errors() -> None:
    img = torch.rand(1, 1, 8, 8)
    with pytest.raises(ValueError):
        guided_filter(img, img, radius=0, eps=1e-3)
    with pytest.raises(ValueError):
        guided_filter(img, img, radius=2, eps=0.0)
    with pytest.raises(ValueError):
        guided_filter(torch.rand(1, 2, 8, 8), img, radius=2, eps=1e-3)
    with pytest.raises(ValueError):
        guided_filter(img, torch.rand(1, 1, 9, 8), radius=2, eps=1e-3)
    with pytest.raises(TypeError):
        guided_filter(img.numpy(), img, radius=2, eps=1e-3)


def test_colour_strips_match_untiled(monkeypatch: pytest.MonkeyPatch) -> None:
    import core.guided as guided

    torch.manual_seed(1)
    guide = torch.rand(1, 3, 70, 33)
    src = torch.rand(1, 2, 70, 33)
    whole = guided_filter(guide, src, radius=3, eps=1e-3)
    monkeypatch.setattr(guided, "_STRIP_BUDGET_BYTES", 400 * 33 * 9)  # ~9-row strips
    tiled = guided_filter(guide, src, radius=3, eps=1e-3)
    assert torch.allclose(tiled, whole, atol=1e-5)
