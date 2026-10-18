from __future__ import annotations

import pytest
import torch

from core.local_laplacian import _blur, _mirror_index, _up, shadows_highlights_camera_raw


def _grey(width: int, height: int, value) -> torch.Tensor:
    """(1,H,W,3) grey image; ``value(x, y)`` gets broadcast index grids."""

    y = torch.arange(height).view(height, 1).expand(height, width)
    x = torch.arange(width).view(1, width).expand(height, width)
    plane = value(x, y).to(torch.float32).expand(height, width)
    return plane.unsqueeze(0).unsqueeze(-1).expand(1, height, width, 3).contiguous()


def _srgb_to_linear(v: torch.Tensor) -> torch.Tensor:
    return torch.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(v: torch.Tensor) -> torch.Tensor:
    return torch.where(v <= 0.0031308, v * 12.92, 1.055 * v.clamp_min(0) ** (1 / 2.4) - 0.055)


def test_mirror_index_and_pyramid_shapes() -> None:
    assert _mirror_index(5, torch.device("cpu")).tolist() == [2, 1, 0, 1, 2, 3, 4, 3, 2]
    assert _mirror_index(1, torch.device("cpu")).tolist() == [0] * 5
    for h, w in [(1, 1), (2, 3), (37, 23), (64, 64)]:
        x = torch.rand(1, 1, h, w)
        assert _blur(x, 2).shape[-2:] == ((h + 1) // 2, (w + 1) // 2)
        assert _up(_blur(x, 2), h, w).shape[-2:] == (h, w)


def test_flat_image_unchanged() -> None:
    img = _grey(64, 48, lambda x, y: torch.tensor(0.4))
    out = shadows_highlights_camera_raw(img, 100.0, -100.0)
    assert (out - 0.4).abs().max().item() < 1e-5


def test_zero_sliders_return_input() -> None:
    img = torch.rand(1, 16, 16, 3)
    assert torch.equal(shadows_highlights_camera_raw(img, 0.0, 0.0), img)


def test_shadows_raise_dark_area_without_halo() -> None:
    w, h = 256, 128
    img = _grey(w, h, lambda x, y: torch.where(x < w // 2, 0.1, 0.6))
    row = shadows_highlights_camera_raw(img, 100.0, 0.0)[0, 64, :, 0]
    assert abs(row[w // 2 + 1].item() - 0.6) < 0.002 and abs(row[w - 5].item() - 0.6) < 0.002
    far, near = row[10].item(), row[w // 2 - 1].item()
    assert abs(far - 0.1) < 0.004 and abs(far - near) < 0.004, (far, near)


@pytest.mark.parametrize(
    ("shadows", "highlights", "want"), [(100.0, 0.0, 165.3), (0.0, -100.0, 89.6)]
)
def test_grey_moves_as_photoshop_camera_raw(shadows: float, highlights: float, want: float) -> None:
    # Mid-grey with 1 % black and white corners; Photoshop 25.4 Camera Raw values (of 255),
    # measured by PhotoCraft.
    n = 256

    def value(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = torch.full_like(x, 0.5, dtype=torch.float32)
        out = torch.where((x >= n - 26) & (y >= n - 26), 0.97, out)
        return torch.where((x >= n - 26) & (y < 26), 0.015, out)

    got = shadows_highlights_camera_raw(_grey(n, n, value), shadows, highlights)[
        0, 175, 100, 0
    ].item()
    assert abs(got * 255.0 - want) < 5.0, got * 255.0


def test_highlights_lower_bright_area_and_keep_darkest() -> None:
    n = 192
    img = _grey(
        n, n, lambda x, y: torch.where((x >= 64) & (x < 128) & (y >= 64) & (y < 128), 0.8, 0.2)
    )
    out = shadows_highlights_camera_raw(img, 0.0, -100.0)
    assert out[0, 96, 96, 0].item() < 0.75
    assert abs(out[0, 5, 5, 0].item() - 0.2) < 0.01


@pytest.mark.parametrize(("shadows", "highlights"), [(100.0, 0.0), (0.0, -100.0), (-100.0, 100.0)])
def test_smooth_gradients_stay_smooth(shadows: float, highlights: float) -> None:
    # Eight-stop ramp: interpolating references too far apart draws contour lines.
    w, h = 640, 48
    ramp = _linear_to_srgb(torch.exp2(torch.arange(w) / (w - 1) * 8.0 - 9.0))
    img = _grey(w, h, lambda x, y: ramp[x])
    out = shadows_highlights_camera_raw(img, shadows, highlights)
    row = torch.log2(_srgb_to_linear(out[0, 24, :, 0].double()).clamp_min(1e-6))
    steps = row[1:] - row[:-1]
    assert (steps > 0).all(), "ramp reverses"
    ripple = (steps[1:] - steps[:-1]).abs()[16:-16].max().item()
    assert ripple < 0.004, ripple


def test_result_independent_of_size() -> None:
    def at(n: int) -> tuple[float, float]:
        img = _grey(n, n, lambda x, y: torch.where((x < n // 3) | (y < n // 4), 0.08, 0.7))
        out = shadows_highlights_camera_raw(img, 70.0, -60.0)
        return out[0, n // 8, n // 8, 0].item(), out[0, 3 * n // 4, 3 * n // 4, 0].item()

    a, b = at(96), at(384)
    assert abs(a[0] - b[0]) < 0.02 and abs(a[1] - b[1]) < 0.02, (a, b)


def test_proxy_path_matches_direct_on_large_image() -> None:
    # 3200 px wide runs on a 1600 px proxy + guided upsampling; compare with the
    # same scene at 1600 px (direct) at matching points.
    def scene(w: int, h: int) -> torch.Tensor:
        return _grey(w, h, lambda x, y: torch.where(x < w // 2, 0.12, 0.65))

    big = shadows_highlights_camera_raw(scene(3200, 200), 80.0, -50.0)
    small = shadows_highlights_camera_raw(scene(1600, 100), 80.0, -50.0)
    assert abs(big[0, 100, 400, 0].item() - small[0, 50, 200, 0].item()) < 0.01
    assert abs(big[0, 100, 2800, 0].item() - small[0, 50, 1400, 0].item()) < 0.01


def test_batch_and_validation() -> None:
    img = torch.rand(2, 40, 50, 3)
    out = shadows_highlights_camera_raw(img, 40.0, -40.0)
    assert out.shape == img.shape and out.min() >= 0.0 and out.max() <= 1.0
    with pytest.raises(ValueError):
        shadows_highlights_camera_raw(img, 120.0, 0.0)
    with pytest.raises(ValueError):
        shadows_highlights_camera_raw(torch.rand(1, 8, 8, 4), 10.0, 0.0)
