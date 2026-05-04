from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_color_matcher_region"


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
    return _load_pack().NODE_CLASS_MAPPINGS["JHPixelProColorMatcherRegion"]


def test_node_registered() -> None:
    pack = _load_pack()
    assert "JHPixelProColorMatcherRegion" in pack.NODE_CLASS_MAPPINGS
    assert (
        pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProColorMatcherRegion"]
        == "Color Matcher Region (LAB)"
    )


def test_node_metadata(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/color"
    assert node_cls.RETURN_TYPES == ("IMAGE",)
    assert node_cls.RETURN_NAMES == ("image_matched",)
    assert node_cls.FUNCTION == "run"


def test_input_types(node_cls) -> None:
    spec = node_cls.INPUT_TYPES()
    assert spec["required"]["image_target"] == ("IMAGE",)
    assert spec["required"]["image_reference"] == ("IMAGE",)
    assert spec["required"]["channels"][0] == ["ab", "lab"]
    assert spec["required"]["channels"][1]["default"] == "ab"
    assert spec["required"]["strength"][1]["default"] == 1.0
    assert spec["optional"]["target_mask"][0] == "MASK"
    assert spec["optional"]["reference_mask"][0] == "MASK"


def test_node_runs_with_different_reference_size_and_target_mask(node_cls) -> None:
    target = torch.zeros((1, 32, 32, 3), dtype=torch.float32)
    target[:, :, :, 0] = 0.8
    reference = torch.zeros((1, 18, 22, 3), dtype=torch.float32)
    reference[:, :, :, 1] = 0.8
    target_mask = torch.zeros((1, 32, 32), dtype=torch.float32)
    target_mask[:, :, :16] = 1.0

    output = node_cls().run(
        target,
        reference,
        "lab",
        1.0,
        target_mask=target_mask,
    )[0]

    assert output.shape == target.shape
    assert output[:, :, :16, 1].mean().item() > output[:, :, :16, 0].mean().item()
    assert torch.equal(output[:, :, 16:, :], target[:, :, 16:, :])
