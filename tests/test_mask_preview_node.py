from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_mask_preview"


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
    _load_pack()
    mod = importlib.import_module(f"{_PACK_MODULE_NAME}.nodes.mask_preview_node")
    return mod.JHPixelProMaskPreview


def test_node_metadata(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/mask"
    assert node_cls.RETURN_TYPES == ("IMAGE",)
    assert node_cls.RETURN_NAMES == ("preview",)
    assert node_cls.FUNCTION == "preview"


def test_input_types(node_cls) -> None:
    spec = node_cls.INPUT_TYPES()
    req = spec["required"]
    assert list(req) == ["mask", "mode", "color", "opacity", "colormap", "show_edges"]
    assert req["mode"][1]["default"] == "overlay"
    assert req["colormap"][1]["default"] == "turbo"
    assert req["opacity"][1]["default"] == 0.5
    assert req["show_edges"][1]["default"] is False
    assert "image" in spec["optional"]
    for _, cfg in [*req.values(), *spec["optional"].values()]:
        assert "tooltip" in cfg


def test_run_with_and_without_image(node_cls) -> None:
    node = node_cls()
    mask = torch.zeros(1, 16, 16)
    mask[:, 4:12, 4:12] = 1.0
    img = torch.rand(1, 16, 16, 3)
    (out,) = node.preview(mask, "overlay", "red", 0.5, "turbo", True, image=img)
    assert out.shape == (1, 16, 16, 3)
    assert out.dtype == torch.float32
    (out,) = node.preview(mask, "heatmap", "red", 0.5, "turbo", False)
    assert out.shape == (1, 16, 16, 3)


def test_run_bad_mode_raises(node_cls) -> None:
    with pytest.raises(ValueError, match="mode"):
        node_cls().preview(torch.zeros(1, 8, 8), "bad", "red", 0.5, "turbo", False)


def test_node_registered() -> None:
    pack = _load_pack()
    assert pack.NODE_CLASS_MAPPINGS["JHPixelProMaskPreview"] is node_cls_from_module()
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProMaskPreview"] == "Mask: Preview Overlay"


def node_cls_from_module():
    module = importlib.import_module(f"{_PACK_MODULE_NAME}.nodes.mask_preview_node")
    return module.JHPixelProMaskPreview
