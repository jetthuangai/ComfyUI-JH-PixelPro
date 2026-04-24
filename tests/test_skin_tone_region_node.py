from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_skin_tone_region"


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
    return _load_pack().NODE_CLASS_MAPPINGS["JHPixelProSkinToneTriRegion"]


def test_node_registered() -> None:
    pack = _load_pack()
    assert "JHPixelProSkinToneTriRegion" in pack.NODE_CLASS_MAPPINGS
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProSkinToneTriRegion"] == (
        "Face: Skin Tone Tri-Region"
    )


def test_node_metadata(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/face"
    assert node_cls.RETURN_TYPES == ("MASK", "MASK", "MASK")
    assert node_cls.RETURN_NAMES == ("shadow_mask", "midtone_mask", "highlight_mask")
    assert node_cls.FUNCTION == "split"


def test_input_types(node_cls) -> None:
    spec = node_cls.INPUT_TYPES()
    assert spec["required"]["image"] == ("IMAGE",)
    assert spec["required"]["shadow_cutoff"][1]["default"] == 0.33
    assert spec["required"]["highlight_cutoff"][1]["default"] == 0.66
    assert spec["required"]["soft_sigma"][1]["default"] == 1.0
    assert spec["optional"]["skin_mask"] == ("MASK",)


def test_node_split_runs_bhwc(node_cls) -> None:
    node = node_cls()
    ramp = torch.linspace(0.0, 1.0, 24, dtype=torch.float32)
    image = ramp.view(1, 1, 24, 1).expand(1, 12, 24, 3).clone()
    masks = node.split(image, shadow_cutoff=0.33, highlight_cutoff=0.66, soft_sigma=1.0)

    assert len(masks) == 3
    assert all(mask.shape == (1, 12, 24) for mask in masks)
    assert torch.allclose(sum(masks), torch.ones((1, 12, 24)), atol=1e-4)
