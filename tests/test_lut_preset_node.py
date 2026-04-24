from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_lut_preset"


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
    return _load_pack().NODE_CLASS_MAPPINGS["JHPixelProLUTPreset"]


def test_node_registered() -> None:
    pack = _load_pack()
    assert "JHPixelProLUTPreset" in pack.NODE_CLASS_MAPPINGS
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProLUTPreset"] == "Color: LUT Preset"


def test_node_metadata(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/color"
    assert node_cls.RETURN_TYPES == ("IMAGE",)
    assert node_cls.RETURN_NAMES == ("image",)
    assert node_cls.FUNCTION == "apply"


def test_input_types(node_cls) -> None:
    required = node_cls.INPUT_TYPES()["required"]
    assert required["image"] == ("IMAGE",)
    assert required["preset"][1]["default"] == "neutral-identity"
    assert "warm-portrait" in required["preset"][0]
    assert required["intensity"][0] == "FLOAT"
    assert required["intensity"][1]["default"] == 1.0


def test_node_apply_identity(node_cls) -> None:
    node = node_cls()
    image = torch.rand((1, 16, 16, 3), dtype=torch.float32)
    (out,) = node.apply(image, "neutral-identity", 1.0)
    assert torch.allclose(out, image, atol=1e-6)
