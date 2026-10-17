from __future__ import annotations

import time
from pathlib import Path

import cv2
import pytest
import torch

from core.dust_spot import dust_spot_removal

_PORTRAIT = Path(__file__).resolve().parents[1] / "workflows" / "sample_portrait.jpg"
_SPOTS = ((30, 40), (80, 100), (150, 60), (60, 180))


def _gradient(h: int = 200, w: int = 240) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    return (0.45 + 0.25 * yy * xx + 0.1 * xx)[None, :, :, None].expand(1, h, w, 3).clone()


def _with_dots(sign: float = -1.0) -> tuple[torch.Tensor, torch.Tensor]:
    base = _gradient()
    img = base.clone()
    for cy, cx in _SPOTS:
        img[:, cy - 3 : cy + 4, cx - 3 : cx + 4, :] += sign * 0.25
    return base, img


def test_dark_dots_removed() -> None:
    base, img = _with_dots(-1.0)
    out, spot_mask = dust_spot_removal(img)
    assert out.shape == img.shape and out.dtype == torch.float32
    assert spot_mask.shape == (1, 200, 240)
    assert (out - base).abs().max() < 0.01
    for cy, cx in _SPOTS:
        assert spot_mask[0, cy, cx] == 1.0


def test_bright_mode_and_dark_mode_are_exclusive() -> None:
    base, img = _with_dots(+1.0)
    out_dark, mask_dark = dust_spot_removal(img, mode="dark")
    assert mask_dark.sum() == 0
    assert torch.equal(out_dark, img.clamp(0, 1))
    out_bright, _ = dust_spot_removal(img, mode="bright")
    assert (out_bright - base).abs().max() < 0.01
    out_both, _ = dust_spot_removal(img, mode="both")
    assert (out_both - base).abs().max() < 0.01


def test_large_dark_shape_untouched() -> None:
    base = _gradient()
    img = base.clone()
    img[:, 60:120, 100:170, :] -= 0.3  # 60x70 px block, far larger than spot_size
    out, spot_mask = dust_spot_removal(img, spot_size=12)
    assert spot_mask[:, 62:118, 102:168].sum() == 0
    assert torch.equal(out[:, 60:120, 100:170], img[:, 60:120, 100:170])


def test_long_thin_line_untouched() -> None:
    img = _gradient()
    img[:, 100:102, 10:230, :] -= 0.3  # hair-like line, much longer than spot_size
    out, spot_mask = dust_spot_removal(img, spot_size=12)
    assert spot_mask.sum() == 0
    assert torch.equal(out, img)


def test_no_spot_image_is_identity() -> None:
    img = _gradient()
    out, spot_mask = dust_spot_removal(img)
    assert spot_mask.sum() == 0
    assert torch.allclose(out, img, atol=1e-5)


def test_identity_outside_spot_mask() -> None:
    _, img = _with_dots(-1.0)
    out, spot_mask = dust_spot_removal(img)
    outside = spot_mask[..., None] == 0
    assert torch.allclose(out * outside, img * outside, atol=1e-5)


def test_mask_limits_removal() -> None:
    base, img = _with_dots(-1.0)
    allow = torch.zeros(1, 200, 240)
    allow[:, :, :120] = 1.0  # spots at x=40,100 allowed; x=60?,180 -> see below
    out, spot_mask = dust_spot_removal(img, allow)
    assert spot_mask[:, :, 120:].sum() == 0
    assert (out[:, :, :120] - base[:, :, :120]).abs().max() < 0.01
    # spot at (60,180) lies outside the mask and must remain
    assert torch.equal(out[:, 57:64, 177:184], img[:, 57:64, 177:184])


def test_protect_mask_keeps_spot() -> None:
    base, img = _with_dots(-1.0)
    protect = torch.zeros(1, 200, 240)
    protect[:, 20:45, 30:55] = 1.0  # around spot (30, 40)
    out, spot_mask = dust_spot_removal(img, protect_mask=protect)
    assert spot_mask[:, 20:45, 30:55].sum() == 0
    assert torch.equal(out[:, 27:34, 37:44], img[:, 27:34, 37:44])
    assert (out[:, 70:90, 90:110] - base[:, 70:90, 90:110]).abs().max() < 0.01


def test_spot_size_controls_what_counts_as_spot() -> None:
    img = _gradient()
    img[:, 100:116, 100:116, :] -= 0.25  # 16 px blob
    _, small = dust_spot_removal(img, spot_size=8)
    _, large = dust_spot_removal(img, spot_size=24)
    assert small.sum() == 0
    assert large.sum() > 0


def test_sensitivity_controls_threshold() -> None:
    img = _gradient()
    img[:, 100:105, 100:105, :] -= 0.05  # faint spot
    _, low = dust_spot_removal(img, sensitivity=0.1)
    _, high = dust_spot_removal(img, sensitivity=1.0)
    assert low.sum() == 0
    assert high.sum() > 0


def test_batch_and_mask_broadcast() -> None:
    _, img = _with_dots(-1.0)
    batch = torch.cat([img, _gradient()])
    out, spot_mask = dust_spot_removal(batch, torch.ones(1, 200, 240))
    assert out.shape[0] == 2 and spot_mask.shape[0] == 2
    assert spot_mask[0].sum() > 0 and spot_mask[1].sum() == 0


def test_tiny_image_does_not_crash() -> None:
    out, _ = dust_spot_removal(torch.rand(1, 5, 7, 3) * 0.1 + 0.5, spot_size=4)
    assert out.shape == (1, 5, 7, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spot_size": 1},
        {"spot_size": 65},
        {"spot_size": 12.5},
        {"spot_size": True},
        {"sensitivity": -0.1},
        {"sensitivity": 1.1},
        {"sensitivity": "x"},
        {"mode": "light"},
    ],
)
def test_parameter_validation(kwargs) -> None:
    with pytest.raises(ValueError):
        dust_spot_removal(_gradient(), **kwargs)


def test_input_validation() -> None:
    with pytest.raises(TypeError):
        dust_spot_removal("x")
    with pytest.raises(ValueError):
        dust_spot_removal(torch.rand(1, 8, 8))
    with pytest.raises(ValueError):
        dust_spot_removal(torch.zeros(1, 8, 8, 3, dtype=torch.long))
    with pytest.raises(TypeError):
        dust_spot_removal(torch.rand(1, 8, 8, 3), "m")
    with pytest.raises(ValueError):
        dust_spot_removal(torch.rand(1, 8, 8, 3), torch.ones(1, 4, 4))
    with pytest.raises(ValueError):
        dust_spot_removal(torch.rand(2, 8, 8, 3), protect_mask=torch.ones(3, 8, 8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_peak_vram_and_speed_2k_cuda() -> None:
    size = 2048
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, size, device="cuda"),
        torch.linspace(0, 1, size, device="cuda"),
        indexing="ij",
    )
    img = (0.4 + 0.3 * yy * xx)[None, :, :, None].expand(1, size, size, 3).clone()
    for i in range(40):
        cy, cx = 100 + (i * 97) % 1800, 100 + (i * 211) % 1800
        img[:, cy - 3 : cy + 4, cx - 3 : cx + 4, :] -= 0.2
    dust_spot_removal(img)  # warm-up
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    out, spot_mask = dust_spot_removal(img, spot_size=12)
    torch.cuda.synchronize()
    assert time.perf_counter() - start < 2.0
    assert torch.cuda.max_memory_allocated() < 2 * 1024**3
    assert spot_mask.sum() > 0 and out.shape == img.shape


def _portrait() -> torch.Tensor:
    if not _PORTRAIT.exists():
        pytest.skip("sample portrait not available")
    bgr = cv2.imread(str(_PORTRAIT))
    return torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).float().div(255.0)[None]


def test_clean_portrait_defaults_change_almost_nothing() -> None:
    image = _portrait()
    out, spot_mask = dust_spot_removal(image)
    total = image.shape[1] * image.shape[2]
    assert spot_mask.sum() < 0.001 * total
    # eye region (eyelashes, iris, lid creases) and the hair block stay untouched
    eye = (slice(None), slice(430, 740), slice(1380, 1840))
    hair = (slice(None), slice(0, 700), slice(0, 1000))
    assert spot_mask[eye].sum() == 0
    assert torch.equal(out[eye], image[eye])
    assert spot_mask[hair].sum() < 400  # only isolated specks, no hair blotches


def test_synthetic_dust_on_portrait_skin_is_removed() -> None:
    image = _portrait()
    height, width = image.shape[1:3]
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    dirty = image.clone()
    truth = torch.zeros(height, width)
    centers = [(1500, 1400), (1600, 900), (2300, 800), (2600, 1300)]
    centers += [(2900, 700), (1200, 1700), (3100, 1500), (2200, 1900)]  # cheek, shoulder
    for i, (cy, cx) in enumerate(centers):
        radius = 1 + i % 3  # dark dots r = 1..3
        dist = ((yy - cy) ** 2 + (xx - cx) ** 2).float().sqrt()
        alpha = (radius + 0.5 - dist).clamp(0, 1) * 0.6
        dirty[0] = dirty[0] * (1 - alpha[..., None]) + 0.05 * alpha[..., None]
        truth = torch.maximum(truth, (alpha > 0.3).float())
    _, spot_mask = dust_spot_removal(dirty)
    recall = float((spot_mask[0] * truth).sum() / truth.sum())
    assert recall >= 0.9


def test_elongated_and_ring_busy_blobs_are_rejected() -> None:
    img = _gradient()
    img[:, 100:104, 100:110, :] -= 0.3  # 4x10 bar: elongated
    rows = torch.arange(200)[None, :, None, None]
    cols = torch.arange(240)[None, None, :, None]
    stripes = ((rows // 2 + cols // 2) % 2).float() * 0.2 - 0.1  # busy checker surroundings
    busy = _gradient() + stripes
    busy[:, 60:66, 100:106, :] -= 0.3
    _, bar_mask = dust_spot_removal(img)
    _, busy_mask = dust_spot_removal(busy)
    assert bar_mask.sum() == 0
    assert busy_mask[:, 60:66, 100:106].sum() == 0
