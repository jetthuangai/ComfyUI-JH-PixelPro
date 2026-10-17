from __future__ import annotations

import time

import kornia.filters as kf
import pytest
import torch

from core.mask_edge_smooth import _diameter, _reflect101_index, mask_edge_smooth


def _mask(size: int = 32) -> torch.Tensor:
    mask = torch.zeros((1, size, size), dtype=torch.float32)
    mask[:, 8:24, 8:24] = 1.0
    return mask


def _guide(size: int = 32) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, size)
    guide = x.view(1, 1, size, 1).expand(1, size, size, 3).contiguous()
    return guide


def test_smooth_without_guide_preserves_shape_and_range() -> None:
    mask = _mask()
    out = mask_edge_smooth(mask, sigma_spatial=3.0, sigma_range=0.1, iterations=1)
    assert out.shape == mask.shape
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_smooth_with_guide_preserves_shape_and_range() -> None:
    mask = _mask()
    out = mask_edge_smooth(mask, _guide(), sigma_spatial=3.0, sigma_range=0.1, iterations=1)
    assert out.shape == mask.shape
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_softens_binary_edge() -> None:
    mask = _mask()
    out = mask_edge_smooth(mask, sigma_spatial=5.0, sigma_range=1.0, iterations=1)
    assert out[0, 8, 16].item() < 1.0
    assert out[0, 7, 16].item() > 0.0


def test_broadcasts_single_guide() -> None:
    mask = _mask().expand(2, -1, -1).clone()
    out = mask_edge_smooth(mask, _guide(), sigma_spatial=3.0, sigma_range=0.1, iterations=1)
    assert out.shape == mask.shape


def test_rejects_bad_guide_shape() -> None:
    with pytest.raises(ValueError, match="guide"):
        mask_edge_smooth(_mask(), torch.zeros((1, 32, 32)))


def test_rejects_bad_iterations() -> None:
    with pytest.raises(ValueError, match="iterations"):
        mask_edge_smooth(_mask(), iterations=0)


def _soft_box(size: int = 64) -> torch.Tensor:
    mask = torch.zeros((1, size, size), dtype=torch.float32)
    mask[:, 16:48, 16:48] = 1.0
    return torch.nn.functional.avg_pool2d(mask[None], 5, 1, 2)[0]


def test_rejects_bad_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        mask_edge_smooth(_mask(), backend="cuda")


def test_gpu_matches_opencv_without_guide() -> None:
    mask = _soft_box()
    ref = mask_edge_smooth(mask, sigma_spatial=3.0, sigma_range=0.2)
    out = mask_edge_smooth(mask, sigma_spatial=3.0, sigma_range=0.2, backend="gpu")
    diff = (ref - out).abs()
    print(f"gpu vs opencv: max={diff.max().item():.5f} mean={diff.mean().item():.6f}")
    assert out.shape == mask.shape
    assert out.dtype == torch.float32
    assert diff.mean().item() < 5e-3
    assert diff.max().item() < 0.05


@pytest.mark.parametrize(
    ("height", "width", "sigma_spatial", "sigma_range", "use_guide"),
    [
        (24, 24, 1.0, 0.2, False),
        (24, 24, 3.0, 0.1, True),
        (31, 17, 2.0, 0.3, True),
        (33, 20, 4.0, 0.05, False),
        (40, 25, 0.5, 0.5, True),
        (13, 13, 3.0, 0.2, True),  # kernel radius 3 on a small image
    ],
)
def test_gpu_matches_kornia(height, width, sigma_spatial, sigma_range, use_guide) -> None:
    gen = torch.Generator().manual_seed(1)
    mask = torch.rand((2, height, width), generator=gen)
    guide = torch.rand((2, height, width, 3), generator=gen)
    kernel = _diameter(sigma_spatial)
    sigma = (sigma_spatial, sigma_spatial)
    if use_guide:
        ref = kf.joint_bilateral_blur(
            mask[:, None],
            guide.permute(0, 3, 1, 2),
            kernel,
            sigma_range,
            sigma,
            border_type="reflect",
            color_distance_type="l1",
        )
    else:
        ref = kf.bilateral_blur(mask[:, None], kernel, sigma_range, sigma, border_type="reflect")
    out = mask_edge_smooth(
        mask,
        guide if use_guide else None,
        sigma_spatial=sigma_spatial,
        sigma_range=sigma_range,
        backend="gpu",
    )
    assert (out - ref[:, 0].clamp(0, 1)).abs().max().item() < 1e-5


def test_reflect101_index_handles_pad_larger_than_size() -> None:
    idx = _reflect101_index(4, 7, torch.device("cpu"))
    assert idx.tolist() == [1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2]
    assert _reflect101_index(1, 3, torch.device("cpu")).tolist() == [0] * 7


def test_gpu_guide_follows_edges() -> None:
    mask = _soft_box(64)
    guide = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    guide[:, :, 20:, :] = 1.0  # strong guide edge that the mask does not have
    plain = mask_edge_smooth(mask, sigma_spatial=4.0, sigma_range=0.1, backend="gpu")
    guided = mask_edge_smooth(mask, guide, sigma_spatial=4.0, sigma_range=0.1, backend="gpu")
    assert (plain - guided).abs().max().item() > 0.02
    assert guided.min().item() >= 0.0
    assert guided.max().item() <= 1.0


def test_gpu_small_image_large_sigma() -> None:
    out = mask_edge_smooth(_mask(16), sigma_spatial=12.0, backend="gpu")
    assert out.shape == (1, 16, 16)
    assert torch.isfinite(out).all()


def test_gpu_batch_iterations_guide_broadcast() -> None:
    mask = _soft_box(32).expand(2, -1, -1).clone()
    out = mask_edge_smooth(
        mask, _guide(32), sigma_spatial=2.0, sigma_range=0.2, iterations=2, backend="gpu"
    )
    assert out.shape == mask.shape


def test_gpu_cpu_fallback_warns_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with caplog.at_level("WARNING"):
        out = mask_edge_smooth(_soft_box(32), sigma_spatial=2.0, backend="gpu")
    assert out.device.type == "cpu"
    assert out.shape == (1, 32, 32)
    assert "running on CPU" in caplog.text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_cuda_2048_sigma16_fast_and_low_memory() -> None:
    mask = torch.rand((1, 2048, 2048))
    guide = torch.rand((1, 2048, 2048, 3))
    mask_edge_smooth(mask, guide, sigma_spatial=2.0, backend="gpu")  # warm up
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start = time.perf_counter()
    out = mask_edge_smooth(mask, guide, sigma_spatial=16.0, backend="gpu")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_mb = (torch.cuda.max_memory_allocated() - base) / 2**20
    assert out.device.type == "cpu"
    assert out.shape == mask.shape
    assert peak_mb < 1024, f"peak {peak_mb:.0f} MB"
    assert elapsed < 10.0, f"{elapsed:.1f} s"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_cuda_512_sigma64_finishes() -> None:
    mask = torch.rand((1, 512, 512))
    start = time.perf_counter()
    out = mask_edge_smooth(mask, sigma_spatial=64.0, backend="gpu")
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    assert time.perf_counter() - start < 30.0
