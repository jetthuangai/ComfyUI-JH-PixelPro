from __future__ import annotations

import pytest
import torch

from core import depth_anything, depth_lens_blur
from core.depth_lens_blur import circle_of_confusion, resolve_focus


def _scene(size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Noisy image + depth ramp (left far = 0, right near = 1)."""
    gen = torch.Generator().manual_seed(0)
    image = torch.rand(1, size, size, 3, generator=gen)
    ramp = torch.linspace(0.0, 1.0, size).view(1, 1, size).expand(1, size, size).contiguous()
    return image, ramp


def _variance(image: torch.Tensor, cols: slice) -> float:
    return float(image[:, :, cols].var())


def test_focus_plane_unchanged() -> None:
    image = torch.rand(1, 32, 32, 3)
    depth = torch.full((1, 32, 32), 0.5)
    out, coc = depth_lens_blur.depth_lens_blur(image, depth, focus_depth=0.5, depth_of_field=0.1)
    assert float(coc.max()) == 0.0
    assert torch.allclose(out, image, atol=1e-5)


def test_far_pixels_blurrier_and_focus_region_sharp() -> None:
    image, depth = _scene()
    out, coc = depth_lens_blur.depth_lens_blur(
        image, depth, focus_depth=1.0, depth_of_field=0.1, max_blur=8
    )
    assert float(coc[0, 0, 0]) == pytest.approx(1.0)
    assert torch.allclose(out[:, :, -3:], image[:, :, -3:], atol=1e-5)
    assert _variance(out, slice(0, 8)) < 0.5 * _variance(image, slice(0, 8))


def test_monotonic_in_max_blur() -> None:
    image, depth = _scene()
    variances = []
    for max_blur in (1, 4, 12, 30):
        out, _ = depth_lens_blur.depth_lens_blur(
            image, depth, focus_depth=1.0, depth_of_field=0.0, max_blur=max_blur
        )
        variances.append(_variance(out, slice(0, 10)))
    assert variances == sorted(variances, reverse=True)
    assert variances[-1] < variances[0]


def test_disc_vs_gaussian_differ_and_both_blur() -> None:
    image, depth = _scene()
    kwargs = dict(focus_depth=1.0, depth_of_field=0.0, max_blur=10)
    disc, _ = depth_lens_blur.depth_lens_blur(image, depth, bokeh="disc", **kwargs)
    gauss, _ = depth_lens_blur.depth_lens_blur(image, depth, bokeh="gaussian", **kwargs)
    assert not torch.allclose(disc, gauss, atol=1e-3)
    for out in (disc, gauss):
        assert _variance(out, slice(0, 8)) < 0.5 * _variance(image, slice(0, 8))


def test_large_radius_downscaled_path() -> None:
    image, depth = _scene(96)
    out, _ = depth_lens_blur.depth_lens_blur(
        image, depth, focus_depth=1.0, depth_of_field=0.0, max_blur=60, layers=4
    )
    assert out.shape == image.shape
    assert _variance(out, slice(0, 8)) < 0.1 * _variance(image, slice(0, 8))


def test_no_halo_from_sharp_foreground() -> None:
    # sharp bright foreground block on dark blurred background: background near the block must
    # not pick up much of the block colour.
    image = torch.zeros(1, 64, 64, 3)
    image[:, :, 32:] = 1.0
    depth = torch.zeros(1, 64, 64)
    depth[:, :, 32:] = 1.0
    out, _ = depth_lens_blur.depth_lens_blur(
        image, depth, focus_depth=1.0, depth_of_field=0.2, max_blur=12
    )
    assert float(out[0, 32, 30].mean()) < 0.1
    assert torch.allclose(out[:, :, 32:], image[:, :, 32:], atol=1e-5)


def test_blur_foreground_false_keeps_near_sharp() -> None:
    image, depth = _scene()
    kw = dict(focus_depth=0.5, depth_of_field=0.0, max_blur=8)
    both, _ = depth_lens_blur.depth_lens_blur(image, depth, blur_foreground=True, **kw)
    behind, coc = depth_lens_blur.depth_lens_blur(image, depth, blur_foreground=False, **kw)
    assert float(coc[:, :, 40:].max()) == 0.0
    assert torch.allclose(behind[:, :, 40:], image[:, :, 40:], atol=1e-5)
    assert _variance(both, slice(50, 64)) < _variance(image, slice(50, 64))


def test_focus_mask_uses_median_depth() -> None:
    _, depth = _scene()
    mask = torch.zeros(1, 64, 64)
    mask[:, :, 10:21] = 1.0
    focus = resolve_focus(depth, 0.9, mask)
    assert float(focus) == pytest.approx(float(depth[0, 0, 15]), abs=1e-6)
    assert float(resolve_focus(depth, 0.9, torch.zeros(1, 64, 64))) == pytest.approx(0.9)
    assert float(resolve_focus(depth, 0.9, None)) == pytest.approx(0.9)


def test_focus_mask_drives_coc() -> None:
    image, depth = _scene()
    mask = torch.zeros(1, 64, 64)
    mask[:, :, 10:21] = 1.0
    _, coc = depth_lens_blur.depth_lens_blur(
        image, depth, focus_depth=1.0, depth_of_field=0.3, focus_mask=mask
    )
    assert float(coc[0, 0, 15]) == pytest.approx(0.0, abs=1e-6)
    assert float(coc[0, 0, 63]) > 0.5


def test_circle_of_confusion_formula() -> None:
    depth = torch.tensor([[[0.0, 0.45, 0.5, 0.55, 1.0]]])
    coc = circle_of_confusion(depth, torch.full((1, 1, 1), 0.5), 0.2)
    assert coc[0, 0].tolist() == pytest.approx(
        [(0.5 - 0.1) / 0.9, 0.0, 0.0, 0.0, (0.5 - 0.1) / 0.9]
    )


def test_missing_depth_calls_estimator(monkeypatch) -> None:
    calls = []

    def fake(image, model="small", resolution=518):
        calls.append((model, resolution, tuple(image.shape)))
        return torch.zeros(image.shape[:3])

    monkeypatch.setattr(depth_anything, "estimate_depth", fake)
    image = torch.rand(2, 24, 24, 3)
    out, coc = depth_lens_blur.depth_lens_blur(image, None, focus_depth=0.0, max_blur=4)
    assert calls == [("small", 518, (2, 24, 24, 3))]
    assert torch.allclose(out, image, atol=1e-5)
    assert coc.shape == (2, 24, 24)


def test_shapes_batch_broadcast_and_depth_resize() -> None:
    image = torch.rand(3, 32, 40, 3)
    depth = torch.rand(1, 16, 20)
    mask = torch.ones(1, 32, 40)
    out, coc = depth_lens_blur.depth_lens_blur(image, depth, focus_mask=mask, max_blur=6)
    assert out.shape == image.shape and out.dtype == torch.float32
    assert coc.shape == (3, 32, 40)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_validation_errors() -> None:
    image, depth = _scene(16)
    f = depth_lens_blur.depth_lens_blur
    with pytest.raises(TypeError):
        f("x", depth)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="image"):
        f(torch.rand(16, 16, 3), depth)
    with pytest.raises(TypeError):
        f(image, "x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="depth"):
        f(image, torch.rand(1, 1, 16, 16))
    with pytest.raises(ValueError, match="batch"):
        f(image, torch.rand(2, 16, 16))
    with pytest.raises(ValueError, match="bokeh"):
        f(image, depth, bokeh="square")
    with pytest.raises(ValueError, match="focus_depth"):
        f(image, depth, focus_depth=1.5)
    with pytest.raises(ValueError, match="depth_of_field"):
        f(image, depth, depth_of_field=-0.1)
    with pytest.raises(ValueError, match="max_blur"):
        f(image, depth, max_blur=0)
    with pytest.raises(ValueError, match="max_blur"):
        f(image, depth, max_blur=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="layers"):
        f(image, depth, layers=1)
    with pytest.raises(ValueError, match="blur_foreground"):
        f(image, depth, blur_foreground=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("bokeh", ["disc", "gaussian"])
@pytest.mark.parametrize("max_blur", [1, 8, 24, 96])
@pytest.mark.parametrize("size", [(1, 1), (2, 2), (8, 8), (64, 64), (3, 500), (500, 3)])
def test_tiny_and_extreme_aspect_do_not_crash(size, max_blur, bokeh) -> None:
    h, w = size
    image = torch.rand(1, h, w, 3)
    depth = torch.linspace(0.0, 1.0, h * w).view(1, h, w)
    out, coc = depth_lens_blur.depth_lens_blur(
        image, depth, focus_depth=0.0, depth_of_field=0.0, max_blur=max_blur, bokeh=bokeh
    )
    assert out.shape == image.shape and coc.shape == (1, h, w)
    assert bool(torch.isfinite(out).all()) and bool(torch.isfinite(coc).all())


def test_nan_in_depth_and_focus_mask_stays_local() -> None:
    image, depth = _scene(32)
    depth = depth.clone()
    depth[0, 5, 5] = float("nan")
    mask = torch.zeros(1, 32, 32)
    mask[:, :, 20:30] = 1.0
    mask[0, 3, 3] = float("nan")
    out, coc = depth_lens_blur.depth_lens_blur(
        image, depth, focus_mask=mask, focus_depth=1.0, max_blur=6
    )
    assert bool(torch.isfinite(out).all()) and bool(torch.isfinite(coc).all())
    clean, _ = depth_lens_blur.depth_lens_blur(
        image, _scene(32)[1], focus_mask=mask.nan_to_num(), focus_depth=1.0, max_blur=6
    )
    assert torch.allclose(out[:, :, 16:], clean[:, :, 16:], atol=1e-5)  # far from the NaN


def test_no_sharp_seam_where_depth_crosses_focus_between_blurred_zones() -> None:
    size = 64
    image = torch.rand(1, size, size, 3, generator=torch.Generator().manual_seed(1))
    cols = torch.arange(size, dtype=torch.float32)
    depth = (0.5 + (cols - 32.0).abs() / 64.0).view(1, 1, size).expand(1, size, size)
    _, coc = depth_lens_blur.depth_lens_blur(
        image, depth.contiguous(), focus_depth=0.5, depth_of_field=0.02, max_blur=24
    )
    assert float(coc[0, :, 28:36].min()) > 0.0  # no zero-CoC (sharp) line across the crossing


def test_wide_focus_zone_stays_exactly_sharp() -> None:
    image, depth = _scene(64)
    out, coc = depth_lens_blur.depth_lens_blur(
        image, depth, focus_depth=0.5, depth_of_field=0.4, max_blur=16
    )
    sharp = slice(26, 38)
    assert float(coc[:, :, sharp].max()) == 0.0
    assert torch.allclose(out[:, :, sharp], image[:, :, sharp], atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cpu_input_is_returned_on_cpu() -> None:
    image, depth = _scene(32)
    out, coc = depth_lens_blur.depth_lens_blur(image, depth, focus_depth=1.0, max_blur=6)
    assert out.device.type == "cpu" and coc.device.type == "cpu"
    cuda_out, _ = depth_lens_blur.depth_lens_blur(
        image.cuda(), depth.cuda(), focus_depth=1.0, max_blur=6
    )
    assert cuda_out.device.type == "cuda"
    assert torch.allclose(out, cuda_out.cpu(), atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_2k_peak_memory_and_time() -> None:
    import time

    torch.cuda.empty_cache()
    image = torch.rand(1, 2048, 2048, 3, device="cuda")
    yy = torch.linspace(0, 1, 2048, device="cuda").view(1, 2048, 1).expand(1, 2048, 2048)
    depth_lens_blur.depth_lens_blur(image, yy, max_blur=24)  # warm-up
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start = time.perf_counter()
    out, _ = depth_lens_blur.depth_lens_blur(image, yy, max_blur=96, layers=16)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() - base
    assert out.shape == image.shape
    assert peak < 2 * 1024**3, f"peak {peak / 1024**3:.2f} GB"
    assert elapsed < 10.0, f"{elapsed:.2f}s"


def test_thin_subject_in_focus_mask_stays_sharp() -> None:
    image, _ = _scene()
    depth = torch.zeros(1, 64, 64)
    depth[:, :, 30:32] = 1.0  # 2 px pole at the focus depth, background far behind
    mask = torch.zeros(1, 64, 64)
    mask[:, :, 30:32] = 1.0
    _, coc = depth_lens_blur.depth_lens_blur(
        image, depth, depth_of_field=0.05, max_blur=48, focus_mask=mask
    )
    assert float(coc[0, :, 30:32].max()) == pytest.approx(0.0, abs=1e-6)
    _, coc_free = depth_lens_blur.depth_lens_blur(
        image, depth, focus_depth=1.0, depth_of_field=0.05, max_blur=48
    )
    assert float(coc_free[0, :, 30:32].min()) > 0.5  # without a mask the closing fills it
