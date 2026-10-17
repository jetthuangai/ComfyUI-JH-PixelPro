from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_auto_align"


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
    return _load_pack().NODE_CLASS_MAPPINGS["JHPixelProAutoAlign"]


def test_node_registered() -> None:
    pack = _load_pack()
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProAutoAlign"] == "Geometry: Auto Align"


def test_node_metadata(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/geometry"
    assert node_cls.RETURN_TYPES == ("IMAGE", "MASK", "STRING", "INT")
    assert node_cls.RETURN_NAMES == ("aligned", "valid_mask", "matrix_json", "inliers")
    assert node_cls.FUNCTION == "run"


def test_input_types(node_cls) -> None:
    required = node_cls.INPUT_TYPES()["required"]
    assert list(required) == [
        "image",
        "reference",
        "method",
        "model",
        "max_side",
        "ransac_threshold",
        "min_inliers",
    ]
    assert required["method"][0] == ["sift", "loftr"]
    assert required["method"][1]["default"] == "sift"
    assert required["model"][0] == ["homography", "affine"]
    assert required["max_side"][1] == {
        "default": 1024,
        "min": 256,
        "max": 2048,
        "step": 64,
        "tooltip": required["max_side"][1]["tooltip"],
    }
    assert required["ransac_threshold"][1]["default"] == 3.0
    assert required["min_inliers"][1]["default"] == 12
    for spec in required.values():
        if len(spec) > 1:
            assert spec[1].get("tooltip")


def test_run_flat_fallback(node_cls) -> None:
    image = torch.full((2, 64, 64, 3), 0.5)
    reference = torch.full((1, 48, 80, 3), 0.5)
    aligned, mask, matrix_json, inliers = node_cls().run(
        image, reference, "sift", "homography", 256, 3.0, 12
    )
    assert aligned.shape == (2, 48, 80, 3) and aligned.dtype == torch.float32
    assert mask.shape == (2, 48, 80)
    matrices = json.loads(matrix_json)
    assert len(matrices) == 2 and len(matrices[0]) == 3 and len(matrices[0][0]) == 3
    assert isinstance(inliers, int) and inliers < 12
