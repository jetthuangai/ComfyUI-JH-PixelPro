from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_sharpen"


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
    module = importlib.import_module(f"{_PACK_MODULE_NAME}.nodes.sharpen_node")
    return module.JHPixelProSmartSharpen


def test_node_metadata(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/filters"
    assert node_cls.RETURN_TYPES == ("IMAGE",)
    assert node_cls.RETURN_NAMES == ("image",)
    assert node_cls.FUNCTION == "sharpen"


def test_input_types(node_cls) -> None:
    spec = node_cls.INPUT_TYPES()
    req, opt = spec["required"], spec["optional"]
    assert list(req) == ["image", "amount", "radius", "threshold", "luminance_only"]
    assert req["amount"][1]["default"] == 1.0 and req["amount"][1]["step"] == 0.05
    assert req["radius"][1]["default"] == 1.0 and req["radius"][1]["step"] == 0.1
    assert req["threshold"][1]["default"] == 0.0 and req["threshold"][1]["step"] == 0.005
    assert req["luminance_only"][0] == "BOOLEAN"
    assert req["luminance_only"][1]["default"] is True
    assert opt["mask"][0] == "MASK"
    for name in ("amount", "radius", "threshold", "luminance_only"):
        assert "tooltip" in req[name][1]


def test_run(node_cls) -> None:
    img = torch.rand(2, 40, 40, 3)
    (out,) = node_cls().sharpen(img, 1.0, 1.0, 0.0, True)
    assert out.shape == img.shape and out.dtype == torch.float32
    assert not torch.allclose(out, img)


def test_run_mask_and_identity(node_cls) -> None:
    img = torch.rand(1, 40, 40, 3)
    (out,) = node_cls().sharpen(img, 1.0, 1.0, 0.0, False, mask=torch.zeros(1, 40, 40))
    assert torch.allclose(out, img, atol=1e-6)
    (out0,) = node_cls().sharpen(img, 0.0, 1.0, 0.0, True)
    assert torch.equal(out0, img)


def test_run_invalid(node_cls) -> None:
    with pytest.raises(ValueError):
        node_cls().sharpen(torch.rand(1, 40, 40, 3), 9.0, 1.0, 0.0, True)


def test_node_registered() -> None:
    pack = _load_pack()
    assert pack.NODE_CLASS_MAPPINGS["JHPixelProSmartSharpen"] is node_cls_from_module()
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProSmartSharpen"] == "Filters: Smart Sharpen"


def node_cls_from_module():
    module = importlib.import_module(f"{_PACK_MODULE_NAME}.nodes.sharpen_node")
    return module.JHPixelProSmartSharpen
