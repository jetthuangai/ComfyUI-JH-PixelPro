from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_palette"


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
    return _load_pack().NODE_CLASS_MAPPINGS["JHPixelProPaletteExtract"]


def test_node_registered() -> None:
    pack = _load_pack()
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProPaletteExtract"] == "Color: Palette Extract"


def test_node_metadata(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/color"
    assert node_cls.RETURN_TYPES == ("IMAGE", "STRING", "IMAGE")
    assert node_cls.RETURN_NAMES == ("palette", "palette_json", "quantized")
    spec = node_cls.INPUT_TYPES()
    colors_cfg = spec["required"]["colors"][1]
    assert (colors_cfg["default"], colors_cfg["min"], colors_cfg["max"]) == (6, 2, 16)
    assert spec["optional"]["mask"][0] == "MASK"
    for _, cfg in (*spec["required"].values(), *spec["optional"].values()):
        assert "tooltip" in cfg


def test_node_runs_and_returns_json(node_cls) -> None:
    img = torch.zeros(1, 32, 32, 3)
    img[:, :, :16] = torch.tensor([1.0, 0.0, 0.0])
    img[:, :, 16:] = torch.tensor([0.0, 0.0, 1.0])
    strip, text, quantized = node_cls().extract(img, 2)
    data = json.loads(text)
    assert {c["hex"] for c in data} == {"#ff0000", "#0000ff"}
    assert strip.shape == (1, 192, 1024, 3)
    assert quantized.shape == img.shape


def test_node_batch_json_is_nested(node_cls) -> None:
    img = torch.rand(2, 16, 16, 3)
    _, text, _ = node_cls().extract(img, 3)
    data = json.loads(text)
    assert len(data) == 2 and len(data[0]) == 3
