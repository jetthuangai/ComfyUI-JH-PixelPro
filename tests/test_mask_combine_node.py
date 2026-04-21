from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_mask_combine"


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
    return _load_pack().NODE_CLASS_MAPPINGS["JHPixelProMaskCombine"]


def test_node_registered() -> None:
    pack = _load_pack()
    assert "JHPixelProMaskCombine" in pack.NODE_CLASS_MAPPINGS
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProMaskCombine"] == "Mask: Combine"


def test_node_metadata(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/mask"
    assert node_cls.RETURN_TYPES == ("MASK",)
    assert node_cls.RETURN_NAMES == ("mask",)
    assert node_cls.FUNCTION == "combine"


def test_input_types(node_cls) -> None:
    required = node_cls.INPUT_TYPES()["required"]
    assert set(required) == {
        "mask_a",
        "mask_b",
        "operation",
        "blend_mode",
        "opacity",
        "feather_sigma",
    }
    assert required["mask_a"] == ("MASK",)
    assert "union" in required["operation"][0]
    assert "soft_feather" in required["blend_mode"][0]


def test_combine_runs(node_cls) -> None:
    node = node_cls()
    mask_a = torch.zeros((1, 24, 24), dtype=torch.float32)
    mask_b = torch.zeros((1, 24, 24), dtype=torch.float32)
    mask_a[:, 6:16, 6:16] = 1.0
    mask_b[:, 10:20, 10:20] = 1.0
    (out,) = node.combine(
        mask_a,
        mask_b,
        operation="union",
        blend_mode="hard",
        opacity=1.0,
        feather_sigma=0.0,
    )
    assert out.shape == mask_a.shape
    assert out.sum().item() > mask_a.sum().item()
