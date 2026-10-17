from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_face_detect_lite"
_NAME = "JHPixelProFaceDetectLite"


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
    return _load_pack().NODE_CLASS_MAPPINGS[_NAME]


def test_registered() -> None:
    pack = _load_pack()
    assert pack.NODE_DISPLAY_NAME_MAPPINGS[_NAME] == "Face: Detect Lite (YuNet GPU)"


def test_schema(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/face"
    assert node_cls.RETURN_TYPES == ("STRING", "STRING", "INT", "MASK")
    assert node_cls.RETURN_NAMES == ("landmarks_json", "bbox_json", "face_count", "face_mask")
    required = node_cls.INPUT_TYPES()["required"]
    assert list(required) == [
        "image",
        "mode",
        "max_faces",
        "confidence_threshold",
        "nms_threshold",
        "detect_size",
    ]
    assert required["detect_size"][1]["default"] == 640
    assert required["detect_size"][1]["step"] == 32
    assert required["confidence_threshold"][1]["default"] == 0.6
    for name, spec in required.items():
        if name != "image":
            assert "tooltip" in spec[1]


def test_run_blank_image(node_cls) -> None:
    from core.face_detect_lite import _get_detector, _pick_device

    try:
        _get_detector(_pick_device(), 0.6, 0.3)
    except RuntimeError as exc:
        pytest.skip(f"YuNet weights unavailable: {exc}")
    image = torch.full((2, 128, 160, 3), 0.5)
    lm, bb, count, mask = node_cls().run(image, "single_largest", 1, 0.6, 0.3, 640)
    assert json.loads(lm) == [] and json.loads(bb) == [] and count == 0
    assert mask.shape == (2, 128, 160) and mask.dtype == torch.float32
