from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_mask_trimap"


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
    return _load_pack().NODE_CLASS_MAPPINGS["JHPixelProTrimapBuilder"]


def test_node_registered() -> None:
    pack = _load_pack()
    assert "JHPixelProTrimapBuilder" in pack.NODE_CLASS_MAPPINGS
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProTrimapBuilder"] == "Mask: Trimap Builder"


def test_node_metadata(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/mask"
    assert node_cls.RETURN_TYPES == ("MASK",)
    assert node_cls.RETURN_NAMES == ("trimap",)
    assert node_cls.FUNCTION == "build"


def test_input_types(node_cls) -> None:
    required = node_cls.INPUT_TYPES()["required"]
    assert set(required) == {"mask", "fg_radius", "bg_radius", "smoothing"}
    assert required["mask"][0] == "MASK"
    assert required["fg_radius"][0] == "INT"


def test_build_runs(node_cls) -> None:
    node = node_cls()
    mask = torch.zeros((1, 24, 24), dtype=torch.float32)
    mask[:, 6:18, 6:18] = 1.0
    (trimap,) = node.build(mask, fg_radius=2, bg_radius=4, smoothing=0.0)
    assert trimap.shape == mask.shape
    assert set(torch.unique(trimap).tolist()) <= {0.0, 0.5, 1.0}
