from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_mask_edge_smooth"


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
    return _load_pack().NODE_CLASS_MAPPINGS["JHPixelProMaskEdgeSmoother"]


def test_node_registered() -> None:
    pack = _load_pack()
    assert "JHPixelProMaskEdgeSmoother" in pack.NODE_CLASS_MAPPINGS
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProMaskEdgeSmoother"] == "Mask: Edge Smoother"


def test_node_metadata(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/mask"
    assert node_cls.RETURN_TYPES == ("MASK",)
    assert node_cls.RETURN_NAMES == ("mask",)
    assert node_cls.FUNCTION == "smooth"


def test_input_types(node_cls) -> None:
    input_types = node_cls.INPUT_TYPES()
    required = input_types["required"]
    optional = input_types["optional"]
    assert set(required) == {"mask", "sigma_spatial", "sigma_range", "iterations"}
    assert set(optional) == {"guide"}
    assert required["mask"] == ("MASK",)
    assert optional["guide"] == ("IMAGE",)


def test_smooth_runs_without_guide(node_cls) -> None:
    node = node_cls()
    mask = torch.zeros((1, 24, 24), dtype=torch.float32)
    mask[:, 6:18, 6:18] = 1.0
    (out,) = node.smooth(mask, sigma_spatial=3.0, sigma_range=0.1, iterations=1)
    assert out.shape == mask.shape
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_smooth_runs_with_guide(node_cls) -> None:
    node = node_cls()
    mask = torch.zeros((1, 24, 24), dtype=torch.float32)
    mask[:, 6:18, 6:18] = 1.0
    guide = torch.rand((1, 24, 24, 3), dtype=torch.float32)
    (out,) = node.smooth(mask, sigma_spatial=3.0, sigma_range=0.1, iterations=1, guide=guide)
    assert out.shape == mask.shape
