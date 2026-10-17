from __future__ import annotations

import time

import pytest
import torch

from core.palette import STRIP_HEIGHT, STRIP_WIDTH, palette_extract

_RED, _GREEN, _BLUE = (200, 30, 30), (30, 160, 60), (40, 60, 210)


def _three_color_image(h: int = 96, w: int = 128) -> torch.Tensor:
    """Left 50% red, next 30% green, last 20% blue."""
    img = torch.zeros(1, h, w, 3)
    a, b = int(w * 0.5), int(w * 0.8)
    for sl, col in ((slice(0, a), _RED), (slice(a, b), _GREEN), (slice(b, w), _BLUE)):
        img[:, :, sl, :] = torch.tensor(col, dtype=torch.float32) / 255.0
    return img


def test_recovers_known_colors_and_shares() -> None:
    img = _three_color_image()
    _, palettes, _ = palette_extract(img, colors=3)
    found = palettes[0]
    assert [c["share"] for c in found] == pytest.approx([0.5, 0.3, 0.2], abs=0.01)
    for color, want in zip(found, (_RED, _GREEN, _BLUE), strict=True):
        assert max(abs(a - b) for a, b in zip(color["rgb"], want, strict=True)) <= 2
        assert color["hex"] == "#{:02x}{:02x}{:02x}".format(*color["rgb"])


def test_output_shapes_and_strip_bands() -> None:
    img = _three_color_image()
    strip, palettes, quantized = palette_extract(img, colors=3)
    assert strip.shape == (1, STRIP_HEIGHT, STRIP_WIDTH, 3)
    assert strip.dtype == torch.float32
    assert quantized.shape == img.shape and quantized.dtype == torch.float32
    first = torch.tensor(palettes[0][0]["rgb"], dtype=torch.float32) / 255.0
    assert torch.allclose(strip[0, 0, 0], first)
    last = torch.tensor(palettes[0][2]["rgb"], dtype=torch.float32) / 255.0
    assert torch.allclose(strip[0, -1, -1], last)
    assert torch.allclose(quantized, img, atol=2.5 / 255.0)


def test_deterministic() -> None:
    img = torch.rand(1, 64, 64, 3, generator=torch.Generator().manual_seed(3))
    a = palette_extract(img, colors=5, max_samples=1000)
    b = palette_extract(img, colors=5, max_samples=1000)
    assert a[1] == b[1]
    assert torch.equal(a[2], b[2])


def test_mask_restricts_pixels() -> None:
    img = _three_color_image()
    mask = torch.zeros(1, 96, 128)
    mask[:, :, 64:] = 1.0  # green + blue only
    _, palettes, quantized = palette_extract(img, mask, colors=2)
    shares = {tuple(c["rgb"]): c["share"] for c in palettes[0]}
    assert len(shares) == 2
    assert all(max(abs(a - b) for a, b in zip(rgb, _RED, strict=True)) > 20 for rgb in shares)
    assert sum(shares.values()) == pytest.approx(1.0, abs=0.01)
    assert quantized.shape == img.shape


def test_empty_mask_raises() -> None:
    with pytest.raises(ValueError, match="no pixels"):
        palette_extract(_three_color_image(), torch.zeros(1, 96, 128), colors=3)


def test_batch_and_mask_broadcast() -> None:
    img = torch.cat([_three_color_image(), 1.0 - _three_color_image()])
    strip, palettes, quantized = palette_extract(img, torch.ones(1, 96, 128), colors=3)
    assert strip.shape[0] == 2 and quantized.shape[0] == 2 and len(palettes) == 2
    assert palettes[0][0]["rgb"] != palettes[1][0]["rgb"]


def test_fewer_distinct_colors_than_clusters() -> None:
    img = torch.full((1, 32, 32, 3), 0.5)
    _, palettes, quantized = palette_extract(img, colors=4)
    assert palettes[0][0]["share"] == pytest.approx(1.0)
    assert torch.allclose(quantized, img, atol=1.5 / 255.0)


@pytest.mark.parametrize(
    "kwargs",
    [{"colors": 1}, {"colors": 17}, {"colors": True}, {"colors": 3.0}, {"max_samples": 10}],
)
def test_parameter_validation(kwargs) -> None:
    with pytest.raises(ValueError):
        palette_extract(_three_color_image(), **kwargs)


def test_input_validation() -> None:
    with pytest.raises(TypeError):
        palette_extract("x")
    with pytest.raises(ValueError):
        palette_extract(torch.rand(1, 8, 8))
    with pytest.raises(ValueError):
        palette_extract(torch.zeros(1, 8, 8, 3, dtype=torch.long))
    with pytest.raises(ValueError):
        palette_extract(torch.rand(1, 8, 8, 3), torch.ones(1, 4, 4))
    with pytest.raises(ValueError):
        palette_extract(torch.rand(2, 8, 8, 3), torch.ones(3, 8, 8))
    with pytest.raises(TypeError):
        palette_extract(torch.rand(1, 8, 8, 3), "m")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_speed_and_memory_2k_cuda() -> None:
    img = torch.rand(1, 2048, 2048, 3, device="cuda")
    palette_extract(img, colors=6)  # warm-up (kernel init)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    palette_extract(img, colors=6)
    torch.cuda.synchronize()
    assert time.perf_counter() - start < 1.0
    assert torch.cuda.max_memory_allocated() < 2 * 1024**3


def test_empty_mask_for_one_item_falls_back_to_full_image(caplog) -> None:
    img = torch.cat([_three_color_image(), _three_color_image()])
    mask = torch.zeros(2, 96, 128)
    mask[1, :, 64:] = 1.0  # item 0 selects nothing, item 1 only green + blue
    with caplog.at_level("WARNING"):
        _, palettes, _ = palette_extract(img, mask, colors=3)
    assert len(palettes) == 2
    assert [c["share"] for c in palettes[0]] == pytest.approx([0.5, 0.3, 0.2], abs=0.01)
    assert "empty mask" in caplog.text
    assert palettes[1][0]["share"] != pytest.approx(0.5, abs=0.05)
