from __future__ import annotations

import pytest
import torch
from kornia.color import ColorMap, ColorMapType, apply_colormap

from core.mask_preview import COLORMAPS, COLORS, mask_preview


def _img(b=1, h=32, w=32):
    g = torch.Generator().manual_seed(0)
    return torch.rand(b, h, w, 3, generator=g)


def _mask(b=1, h=32, w=32):
    m = torch.zeros(b, h, w)
    m[:, 8:24, 8:24] = 1.0
    return m


def test_colormap_list_matches_kornia() -> None:
    assert set(COLORMAPS) == {m.name for m in ColorMapType}
    assert "turbo" in COLORMAPS


def test_overlay_opacity_zero_is_image() -> None:
    img = _img()
    out = mask_preview(img, _mask(), opacity=0.0)
    assert torch.allclose(out, img)


def test_overlay_mask_zero_is_image() -> None:
    img = _img()
    out = mask_preview(img, torch.zeros(1, 32, 32))
    assert torch.allclose(out, img)


@pytest.mark.parametrize("color", list(COLORS))
def test_overlay_full_is_solid_colour(color) -> None:
    out = mask_preview(_img(), torch.ones(1, 32, 32), color=color, opacity=1.0)
    expected = torch.tensor(COLORS[color]).expand(1, 32, 32, 3)
    assert torch.allclose(out, expected)


def test_overlay_half() -> None:
    img = torch.zeros(1, 4, 4, 3)
    out = mask_preview(img, torch.ones(1, 4, 4), color="white", opacity=0.5)
    assert torch.allclose(out, torch.full((1, 4, 4, 3), 0.5))


def test_overlay_image_none_is_grey() -> None:
    out = mask_preview(None, torch.zeros(1, 16, 16))
    assert out.shape == (1, 16, 16, 3)
    assert torch.allclose(out, torch.full_like(out, 0.5))


def test_heatmap_matches_apply_colormap() -> None:
    mask = torch.linspace(0, 1, 64).reshape(1, 8, 8)
    out = mask_preview(None, mask, mode="heatmap", colormap="viridis")
    cmap = ColorMap(base=ColorMapType.viridis, num_colors=256)
    expected = apply_colormap(mask.unsqueeze(1), cmap).permute(0, 2, 3, 1)
    assert torch.allclose(out, expected)
    assert out.std() > 0.05


def test_heatmap_with_image_lerps() -> None:
    img = _img(h=8, w=8)
    mask = torch.rand(1, 8, 8)
    heat = mask_preview(None, mask, mode="heatmap")
    assert torch.allclose(mask_preview(img, mask, mode="heatmap", opacity=1.0), heat, atol=1e-6)
    assert torch.allclose(mask_preview(img, mask, mode="heatmap", opacity=0.0), img, atol=1e-6)


def test_heatmap_all_colormaps_run() -> None:
    mask = torch.rand(1, 8, 8)
    for name in COLORMAPS:
        out = mask_preview(None, mask, mode="heatmap", colormap=name)
        assert out.shape == (1, 8, 8, 3)
        assert torch.isfinite(out).all()


def test_mask_only_exact() -> None:
    mask = torch.rand(2, 8, 8)
    out = mask_preview(_img(2, 8, 8), mask, mode="mask_only")
    assert torch.equal(out, mask.unsqueeze(-1).expand(2, 8, 8, 3))


def test_cutout_checkerboard_where_mask_zero() -> None:
    img = torch.ones(1, 32, 32, 3)
    mask = torch.zeros(1, 32, 32)
    mask[:, :, 16:] = 1.0
    out = mask_preview(img, mask, mode="cutout")
    assert torch.allclose(out[0, 0, 0], torch.full((3,), 0.4))
    assert torch.allclose(out[0, 0, 8], torch.full((3,), 0.6))
    assert torch.allclose(out[0, 8, 0], torch.full((3,), 0.6))
    assert torch.allclose(out[:, :, 16:, :], img[:, :, 16:, :])


def test_edges_only_at_boundary() -> None:
    img = torch.zeros(1, 32, 32, 3)
    out = mask_preview(img, _mask(), color="green", opacity=0.0, show_edges=True)
    changed = (out - img).abs().sum(-1)[0] > 0
    assert changed.any()
    assert not changed[12:20, 12:20].any()  # deep interior
    assert not changed[:5, :].any()
    assert not changed[:, :5].any()  # far outside
    assert torch.allclose(out[0, 7, 16], torch.tensor([0.0, 1.0, 0.0]))
    assert changed.sum() <= 4 * 18 * 2 + 16


def test_no_edges_for_uniform_mask() -> None:
    img = _img()
    out = mask_preview(img, torch.zeros(1, 32, 32), show_edges=True)
    assert torch.allclose(out, img)


def test_batch_broadcast() -> None:
    out = mask_preview(_img(1), _mask(3))
    assert out.shape == (3, 32, 32, 3)
    out = mask_preview(_img(3), _mask(1))
    assert out.shape == (3, 32, 32, 3)
    assert out.dtype == torch.float32


def test_four_channel_image_uses_rgb() -> None:
    img = torch.rand(1, 8, 8, 4)
    out = mask_preview(img, torch.zeros(1, 8, 8))
    assert out.shape[-1] == 3
    assert torch.allclose(out, img[..., :3])


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"mode": "nope"}, ValueError),
        ({"color": "pink"}, ValueError),
        ({"colormap": "nope"}, ValueError),
        ({"opacity": 1.5}, ValueError),
        ({"opacity": True}, TypeError),
        ({"show_edges": 1}, TypeError),
    ],
)
def test_bad_args(kwargs, exc) -> None:
    with pytest.raises(exc):
        mask_preview(_img(), _mask(), **kwargs)


def test_shape_errors() -> None:
    with pytest.raises(ValueError, match="match mask"):
        mask_preview(_img(h=16), _mask())
    with pytest.raises(ValueError, match="batch"):
        mask_preview(_img(2), _mask(3))
    with pytest.raises(ValueError, match="mask must be"):
        mask_preview(None, torch.zeros(1, 1, 8, 8))
    with pytest.raises(ValueError, match="image must be"):
        mask_preview(torch.zeros(1, 32, 32), _mask())
    with pytest.raises(TypeError):
        mask_preview(None, "x")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        mask_preview("x", _mask())  # type: ignore[arg-type]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_heatmap() -> None:
    out = mask_preview(None, _mask().cuda(), mode="heatmap", show_edges=True)
    assert out.device.type == "cuda"


def test_bool_mask_matches_float_mask() -> None:
    mask = torch.zeros(1, 8, 8, dtype=torch.bool)
    mask[:, 2:6, 2:6] = True
    image = torch.rand(1, 8, 8, 3)
    assert torch.equal(mask_preview(image, mask), mask_preview(image, mask.float()))
