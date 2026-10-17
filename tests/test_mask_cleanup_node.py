from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_mask_cleanup"


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
    mod = importlib.import_module(f"{_PACK_MODULE_NAME}.nodes.mask_cleanup_node")
    return mod.JHPixelProMaskCleanup


def test_node_metadata(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/mask"
    assert node_cls.RETURN_TYPES == ("MASK",)
    assert node_cls.RETURN_NAMES == ("mask",)
    assert node_cls.FUNCTION == "cleanup"


def test_input_types(node_cls) -> None:
    required = node_cls.INPUT_TYPES()["required"]
    assert list(required) == [
        "mask",
        "threshold",
        "min_island_area",
        "max_hole_area",
        "feather_px",
        "connectivity",
    ]
    assert required["threshold"][1]["default"] == 0.5
    assert required["min_island_area"][1]["default"] == 64
    assert required["max_hole_area"][1]["default"] == 64
    assert required["feather_px"][1]["default"] == 0.0
    assert required["feather_px"][1]["step"] == 0.5
    assert required["connectivity"][0] == ["8", "4"]
    assert required["connectivity"][1]["default"] == "8"
    for name, spec in required.items():
        if name != "mask":
            assert "tooltip" in spec[1], name


def test_cleanup_runs(node_cls) -> None:
    m = torch.zeros(1, 64, 64)
    m[0, 10:30, 10:30] = 1.0
    m[0, 50, 50] = 1.0
    (out,) = node_cls().cleanup(m, 0.5, 64, 64, 0.0, "8")
    assert out.shape == m.shape and out.dtype == torch.float32
    assert out.sum().item() == 400


def test_connectivity_string_converted(node_cls) -> None:
    m = torch.zeros(1, 16, 16)
    m[0, 4, 4] = m[0, 5, 5] = m[0, 6, 6] = 1.0
    node = node_cls()
    assert node.cleanup(m, 0.5, 3, 0, 0.0, "8")[0].sum().item() == 3
    assert node.cleanup(m, 0.5, 3, 0, 0.0, "4")[0].sum().item() == 0


def test_feather_through_node(node_cls) -> None:
    m = torch.zeros(1, 16, 32)
    m[0, :, :16] = 1.0
    (out,) = node_cls().cleanup(m, 0.5, 0, 0, 4.0, "8")
    assert 0.0 < out[0, 8, 16].item() < 0.5 < out[0, 8, 15].item() < 1.0


def test_node_registered() -> None:
    pack = _load_pack()
    assert pack.NODE_CLASS_MAPPINGS["JHPixelProMaskCleanup"] is node_cls_from_module()
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProMaskCleanup"] == "Mask: Cleanup"


def node_cls_from_module():
    module = importlib.import_module(f"{_PACK_MODULE_NAME}.nodes.mask_cleanup_node")
    return module.JHPixelProMaskCleanup
