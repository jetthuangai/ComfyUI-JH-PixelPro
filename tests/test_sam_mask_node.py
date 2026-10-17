"""Tests for the SAM One-Click ComfyUI wrapper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_sam_mask"


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
    return _load_pack().NODE_CLASS_MAPPINGS["JHPixelProSAMMask"]


def test_node_registered() -> None:
    pack = _load_pack()
    assert "JHPixelProSAMMask" in pack.NODE_CLASS_MAPPINGS
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProSAMMask"] == "Mask: SAM One-Click"


def test_schema(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/mask"
    assert node_cls.RETURN_TYPES == ("MASK", "FLOAT", "IMAGE")
    assert node_cls.RETURN_NAMES == ("mask", "score", "preview")
    spec = node_cls.INPUT_TYPES()
    for group in spec.values():
        for _name, (_type, opts) in group.items():
            assert "tooltip" in opts
    assert spec["required"]["model"][0] == ["vit_b", "mobile_sam", "vit_l", "vit_h"]
    assert spec["required"]["model"][1]["default"] == "vit_b"


def test_bad_prompt_raises(node_cls) -> None:
    with pytest.raises(ValueError):
        node_cls().segment(torch.rand(1, 16, 16, 3), "vit_b", "[[", "", True, 0.0)


@pytest.mark.skipif(
    not (_PACK_ROOT.parents[1] / "models" / "sams" / "mobile_sam.pt").exists(),
    reason="mobile_sam.pt not in models/sams",
)
def test_run_mobile_sam(node_cls) -> None:
    image = torch.zeros(1, 96, 96, 3) + 0.2
    image[:, 30:70, 30:70, :] = 0.9
    mask, score, preview = node_cls().segment(
        image, "mobile_sam", "[[50, 50]]", "", True, 0.0, bbox_json="", mask_hint=None
    )
    assert mask.shape == (1, 96, 96)
    assert isinstance(score, float)
    assert preview.shape == (1, 96, 96, 3) and preview.dtype == torch.float32


def test_keep_model_loaded_toggle(node_cls, monkeypatch) -> None:
    module = sys.modules[f"{_PACK_MODULE_NAME}.nodes.sam_mask_node"]
    calls = []
    monkeypatch.setattr(
        module, "sam_mask", lambda image, **k: (torch.zeros(image.shape[:3]), torch.zeros(1))
    )
    monkeypatch.setattr(module.sam_core, "unload", lambda: calls.append(1))
    assert node_cls.INPUT_TYPES()["optional"]["keep_model_loaded"][1]["default"] is True
    node_cls().segment(torch.rand(1, 8, 8, 3), "vit_b", "", "", True, 0.0)
    assert calls == []
    node_cls().segment(torch.rand(1, 8, 8, 3), "vit_b", "", "", True, 0.0, keep_model_loaded=False)
    assert calls == [1]
