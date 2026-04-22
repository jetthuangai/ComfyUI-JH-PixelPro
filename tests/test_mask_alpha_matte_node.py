from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_mask_alpha_matte"


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
    return _load_pack().NODE_CLASS_MAPPINGS["JHPixelProAlphaMatteExtractor"]


def test_node_registered() -> None:
    pack = _load_pack()
    assert "JHPixelProAlphaMatteExtractor" in pack.NODE_CLASS_MAPPINGS
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProAlphaMatteExtractor"] == "Mask: Alpha Matte"


def test_node_metadata(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/mask"
    assert node_cls.RETURN_TYPES == ("MASK",)
    assert node_cls.RETURN_NAMES == ("alpha",)
    assert node_cls.FUNCTION == "extract"


def test_input_types(node_cls) -> None:
    required = node_cls.INPUT_TYPES()["required"]
    assert set(required) == {"trimap", "guide", "epsilon", "window_radius", "lambda_constraint"}
    assert required["trimap"][0] == "MASK"
    assert required["guide"] == ("IMAGE",)
    assert required["epsilon"][0] == "FLOAT"
    assert required["window_radius"][0] == "INT"
    assert required["lambda_constraint"][0] == "FLOAT"
    assert required["lambda_constraint"][1]["default"] == 100.0


def test_extract_runs(node_cls) -> None:
    node = node_cls()
    trimap = torch.zeros((1, 16, 16), dtype=torch.float32)
    trimap[:, 4:12, 4:12] = 0.5
    trimap[:, 6:10, 6:10] = 1.0
    guide = torch.rand((1, 16, 16, 3), dtype=torch.float32)
    (alpha,) = node.extract(trimap, guide, epsilon=1e-7, window_radius=1, lambda_constraint=100.0)
    assert alpha.shape == trimap.shape
    assert alpha.min().item() >= 0.0
    assert alpha.max().item() <= 1.0


def test_extract_rejects_bad_lambda_constraint(node_cls) -> None:
    node = node_cls()
    trimap = torch.zeros((1, 16, 16), dtype=torch.float32)
    trimap[:, 4:12, 4:12] = 0.5
    trimap[:, 6:10, 6:10] = 1.0
    guide = torch.rand((1, 16, 16, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="lambda_constraint"):
        node.extract(trimap, guide, epsilon=1e-7, window_radius=1, lambda_constraint=0.5)
