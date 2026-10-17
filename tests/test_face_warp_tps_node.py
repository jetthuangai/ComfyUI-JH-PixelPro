from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_face_warp_tps"


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
    module = importlib.import_module(f"{_PACK_MODULE_NAME}.nodes.face_warp_tps_node")
    return module.JHPixelProFaceWarpTPS


def _landmarks(size: int = 64) -> torch.Tensor:
    xs = torch.linspace(0.25, 0.75, 18) * (size - 1)
    ys = torch.linspace(0.25, 0.75, 26) * (size - 1)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1).unsqueeze(0)


def test_node_metadata(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/face"
    assert node_cls.RETURN_TYPES == ("IMAGE",)
    assert node_cls.RETURN_NAMES == ("warped",)
    assert node_cls.FUNCTION == "warp"


def test_input_types(node_cls) -> None:
    required = node_cls.INPUT_TYPES()["required"]
    assert required["image"] == ("IMAGE",)
    assert required["src_landmarks"] == ("LANDMARKS",)
    assert required["dst_landmarks"] == ("LANDMARKS",)
    assert required["max_points"][0] == "INT"
    assert required["max_points"][1]["default"] == 468
    assert required["max_points"][1]["min"] == 16
    assert required["max_points"][1]["max"] == 468
    assert required["anchor_border"][0] == "BOOLEAN"
    assert required["anchor_border"][1]["default"] is True
    optional = node_cls.INPUT_TYPES()["optional"]
    assert optional["smoothness"][0] == "FLOAT"
    assert optional["smoothness"][1]["default"] == 1.0
    assert "tooltip" in optional["smoothness"][1]
    for name in ("max_points", "anchor_border"):
        assert "tooltip" in required[name][1]


def test_identity_warp_with_3d_landmarks(node_cls) -> None:
    image = torch.rand(1, 64, 64, 3)
    lm = _landmarks()
    (out,) = node_cls().warp(image, lm, lm.clone(), 128, True)
    assert out.shape == image.shape and out.dtype == torch.float32
    assert torch.allclose(out, image, atol=1e-4)


def test_accepts_4d_landmarks_uses_first_face(node_cls) -> None:
    image = torch.rand(1, 64, 64, 3)
    lm = _landmarks().unsqueeze(1)  # (B,F,468,2)
    (out,) = node_cls().warp(image, lm, lm + 3.0, 64, False)
    assert out.shape == image.shape
    assert torch.isfinite(out).all()


def test_bad_landmark_rank_raises(node_cls) -> None:
    image = torch.rand(1, 64, 64, 3)
    with pytest.raises(ValueError):
        node_cls().warp(image, torch.zeros(468, 2), torch.zeros(468, 2), 128, True)


def test_node_registered() -> None:
    pack = _load_pack()
    assert pack.NODE_CLASS_MAPPINGS["JHPixelProFaceWarpTPS"] is node_cls_from_module()
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProFaceWarpTPS"] == "Face Warp (TPS smooth)"


def node_cls_from_module():
    module = importlib.import_module(f"{_PACK_MODULE_NAME}.nodes.face_warp_tps_node")
    return module.JHPixelProFaceWarpTPS
