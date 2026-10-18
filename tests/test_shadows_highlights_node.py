from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from core.shadows_highlights import shadows_highlights

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_shadows_highlights"


def _load_pack():
    if _PACK_MODULE_NAME in sys.modules:
        return sys.modules[_PACK_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(
        _PACK_MODULE_NAME,
        _PACK_ROOT / "__init__.py",
        submodule_search_locations=[str(_PACK_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PACK_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def node_cls():
    return _load_pack().NODE_CLASS_MAPPINGS["JHPixelProShadowsHighlights"]


def _scene() -> torch.Tensor:
    img = torch.full((1, 64, 96, 3), 0.12)
    img[:, :, 48:] = 0.85
    img[:, 20:40, 10:30] = torch.tensor([0.2, 0.1, 0.05])
    img[:, 52:, :12] = 0.0  # real black and white points (>1 % of pixels), as photos have
    img[:, 52:, -12:] = 1.0
    return img


def test_node_registered() -> None:
    pack = _load_pack()
    assert (
        pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProShadowsHighlights"]
        == "Color: Shadows/Highlights Pro"
    )


def test_node_metadata_is_simple(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/color"
    assert node_cls.RETURN_TYPES == ("IMAGE",)
    inputs = node_cls.INPUT_TYPES()
    assert list(inputs["required"]) == ["image", "shadows", "highlights", "mode"]
    assert inputs["required"]["mode"][1]["default"] == "camera_raw"
    assert all(key == "mask" or key.startswith("ps_") for key in inputs["optional"])


def _textured() -> torch.Tensor:
    """Photo-like scene: textured dark half, textured bright half, real black/white points."""

    torch.manual_seed(0)
    img = torch.full((1, 96, 128, 3), 0.15)
    img[:, :, 64:] = 0.8
    img = img + 0.06 * torch.rand(1, 96, 128, 1)
    img[:, 80:, :8] = 0.0
    img[:, 80:, -8:] = 1.0
    return img.clamp(0.0, 1.0)


@pytest.mark.parametrize("mode", ["camera_raw", "photoshop"])
def test_shadows_lift_dark_side(node_cls, mode: str) -> None:
    img = _textured()
    (out,) = node_cls().run(img, 50.0, 0.0, mode)
    assert out.shape == img.shape and out.dtype == torch.float32
    assert out[:, :70, :56].mean() > img[:, :70, :56].mean() + 0.005


@pytest.mark.parametrize("mode", ["camera_raw", "photoshop"])
def test_highlights_lower_bright_side(node_cls, mode: str) -> None:
    img = _textured()
    (out,) = node_cls().run(img, 0.0, -50.0, mode)
    assert out[:, :70, 72:120].mean() < img[:, :70, 72:120].mean() - 0.005


def test_zero_sliders_identity(node_cls) -> None:
    img = _scene()
    (out,) = node_cls().run(img, 0.0, 0.0, "photoshop")
    assert torch.equal(out, img)


def test_mask_limits_effect(node_cls) -> None:
    img = _scene()
    mask = torch.zeros(1, 64, 96)
    mask[:, :32] = 1.0
    (out,) = node_cls().run(img, 60.0, 0.0, "camera_raw", mask=mask)
    assert torch.allclose(out[:, 32:], img[:, 32:], atol=1e-6)
    assert not torch.allclose(out[:, :32], img[:, :32], atol=1e-3)


def test_ps_options_are_passed(node_cls) -> None:
    img = _scene()
    (a,) = node_cls().run(img, 50.0, 0.0, "photoshop", ps_shadow_radius=2.0)
    (b,) = node_cls().run(img, 50.0, 0.0, "photoshop", ps_shadow_radius=60.0)
    assert not torch.allclose(a, b)


def test_photoshop_ignores_unsupported_signs() -> None:
    img = _scene()
    out = shadows_highlights(img, mode="photoshop", shadows=-50.0, highlights=50.0)
    assert torch.equal(out, img)  # both dialog amounts map to 0 %


def test_validation() -> None:
    img = _scene()
    with pytest.raises(ValueError):
        shadows_highlights(img, mode="lightroom")
    with pytest.raises(ValueError):
        shadows_highlights(img, ps_options={"radius": 3.0})
    with pytest.raises(ValueError):
        shadows_highlights(img, mask=torch.ones(1, 10, 10))
