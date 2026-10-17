from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_pyramid_blend"


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
    return _load_pack().NODE_CLASS_MAPPINGS["JHPixelProPyramidBlend"]


def test_registered() -> None:
    pack = _load_pack()
    name = pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProPyramidBlend"]
    assert name == "Compositing: Pyramid Blend"


def test_metadata(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/compositing"
    assert node_cls.RETURN_TYPES == ("IMAGE",)
    assert node_cls.RETURN_NAMES == ("image",)
    assert node_cls.FUNCTION == "blend"


def test_input_types(node_cls) -> None:
    required = node_cls.INPUT_TYPES()["required"]
    assert list(required) == ["image_a", "image_b", "mask", "levels"]
    assert required["image_a"][0] == "IMAGE"
    assert required["mask"][0] == "MASK"
    kind, opts = required["levels"]
    assert kind == "INT"
    assert (opts["default"], opts["min"], opts["max"]) == (5, 1, 10)
    assert all("tooltip" in spec[1] for spec in required.values())


def test_run(node_cls) -> None:
    a, b = torch.rand(1, 48, 64, 3), torch.rand(1, 48, 64, 3)
    m = torch.ones(1, 48, 64)
    (out,) = node_cls().blend(a, b, m, 4)
    assert out.shape == (1, 48, 64, 3)
    assert out.dtype == torch.float32
    assert torch.allclose(out, b, atol=1e-5)


def test_run_invalid_levels(node_cls) -> None:
    a = torch.rand(1, 16, 16, 3)
    with pytest.raises(ValueError, match="levels"):
        node_cls().blend(a, a, torch.zeros(1, 16, 16), 0)
