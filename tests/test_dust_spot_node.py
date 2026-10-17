from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_dust_spot"


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
    return _load_pack().NODE_CLASS_MAPPINGS["JHPixelProDustSpotRemoval"]


def test_node_registered() -> None:
    pack = _load_pack()
    name = pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProDustSpotRemoval"]
    assert name == "Filters: Dust & Spot Removal"


def test_node_metadata(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/filters"
    assert node_cls.RETURN_TYPES == ("IMAGE", "MASK")
    assert node_cls.RETURN_NAMES == ("image", "spot_mask")
    spec = node_cls.INPUT_TYPES()
    assert spec["required"]["spot_size"][1]["default"] == 12
    assert spec["required"]["sensitivity"][1]["default"] == 0.5
    assert spec["required"]["mode"][0] == ["dark", "bright", "both"]
    assert spec["required"]["mode"][1]["default"] == "dark"
    assert spec["optional"]["mask"][0] == "MASK"
    assert spec["optional"]["protect_mask"][0] == "MASK"
    for _, cfg in (*spec["required"].values(), *spec["optional"].values()):
        assert "tooltip" in cfg


def test_node_removes_dark_dot(node_cls) -> None:
    img = torch.full((1, 64, 64, 3), 0.6)
    img[:, 30:35, 30:35, :] = 0.3
    out, spot_mask = node_cls().remove(img, 12, 0.5, "dark")
    assert out.shape == img.shape and spot_mask.shape == (1, 64, 64)
    assert (out - 0.6).abs().max() < 0.01
    assert spot_mask[0, 32, 32] == 1.0


def test_node_with_masks(node_cls) -> None:
    img = torch.full((1, 64, 64, 3), 0.6)
    img[:, 30:35, 30:35, :] = 0.3
    out, spot_mask = node_cls().remove(img, 12, 0.5, "dark", mask=torch.zeros(1, 64, 64))
    assert spot_mask.sum() == 0
    assert torch.equal(out, img)
