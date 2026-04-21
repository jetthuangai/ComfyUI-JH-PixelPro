from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_mask_edge_refine"


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
    return _load_pack().NODE_CLASS_MAPPINGS["JHPixelProEdgeAwareMaskRefiner"]


def test_node_registered() -> None:
    pack = _load_pack()
    assert "JHPixelProEdgeAwareMaskRefiner" in pack.NODE_CLASS_MAPPINGS
    assert (
        pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProEdgeAwareMaskRefiner"]
        == "Mask: Edge-Aware Refiner"
    )


def test_node_metadata(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/mask"
    assert node_cls.RETURN_TYPES == ("MASK",)
    assert node_cls.RETURN_NAMES == ("refined_mask",)
    assert node_cls.FUNCTION == "refine"


def test_input_types(node_cls) -> None:
    required = node_cls.INPUT_TYPES()["required"]
    assert set(required) == {"mask", "guide", "radius", "eps", "feather_sigma"}
    assert required["mask"] == ("MASK",)
    assert required["guide"] == ("IMAGE",)
    assert required["radius"][0] == "INT"
    assert required["eps"][0] == "FLOAT"


def test_refine_runs(node_cls) -> None:
    node = node_cls()
    mask = torch.zeros((1, 32, 32), dtype=torch.float32)
    mask[:, 8:24, 8:24] = 1.0
    guide = torch.rand((1, 32, 32, 3), dtype=torch.float32)
    (out,) = node.refine(mask, guide, radius=2, eps=1e-3, feather_sigma=0.0)
    assert out.shape == mask.shape
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0
