from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_clahe"


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
    return _load_pack().NODE_CLASS_MAPPINGS["JHPixelProCLAHE"]


def test_node_registered() -> None:
    pack = _load_pack()
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProCLAHE"] == "Color: CLAHE Local Contrast"


def test_node_metadata(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/color"
    assert node_cls.RETURN_TYPES == ("IMAGE",)
    assert node_cls.RETURN_NAMES == ("image",)
    assert node_cls.FUNCTION == "run"


def test_input_types(node_cls) -> None:
    spec = node_cls.INPUT_TYPES()
    required = spec["required"]
    assert required["image"][0] == "IMAGE"
    assert required["clip_limit"][1]["default"] == 2.0
    assert required["clip_limit"][1]["step"] == 0.1
    assert required["grid_size"][1]["default"] == 8
    assert required["strength"][1]["default"] == 1.0
    assert required["strength"][1]["step"] == 0.05
    assert spec["optional"]["mask"][0] == "MASK"
    for entry in {**required, **spec["optional"]}.values():
        assert "tooltip" in entry[1]


def test_run(node_cls) -> None:
    image = torch.rand(1, 64, 64, 3)
    (out,) = node_cls().run(image, 2.0, 4, 1.0)
    assert out.shape == image.shape and out.dtype == torch.float32


def test_run_with_mask_and_strength_zero(node_cls) -> None:
    image = torch.rand(2, 64, 64, 3)
    (out,) = node_cls().run(image, 2.0, 4, 0.0, mask=torch.ones(1, 64, 64))
    assert torch.allclose(out, image, atol=1e-6)


def test_run_rejects_bad_input(node_cls) -> None:
    with pytest.raises(ValueError):
        node_cls().run(torch.rand(1, 64, 64, 3), 2.0, 1, 1.0)
